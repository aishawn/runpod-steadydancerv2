# SteadyDancer 使用指南

## 概述

SteadyDancer 是一个基于 Image-to-Video 的动画框架，用于根据参考图像和控制视频的姿态信息生成动画。它确保第一帧的完美保持，适合人物动画生成。

## 启动应用

### 1. 通过 Docker 启动（推荐）

```bash
# 构建镜像
docker build -t wan2gp-steadydancer .

# 运行容器（需要 GPU 支持）
docker run --gpus all -p 7860:7860 wan2gp-steadydancer
```

### 2. 直接启动

```bash
# 进入工作目录
cd /workspace

# 启动应用（监听所有网络接口）
python3 wgp.py --listen

# 或者指定端口
python3 wgp.py --listen --server-port 7860
```

## 在 Web UI 中使用 SteadyDancer

### 步骤 1: 选择模型

1. 打开浏览器访问 `http://localhost:7860`（或容器映射的端口）
2. 在模型选择下拉菜单中选择 **"Wan2.1 Steady Dancer 14B"** 或 **"steadydancer"**

### 步骤 2: 准备输入文件

SteadyDancer 需要以下输入：

#### 必需输入：
- **参考图像 (Image Start)**: 包含要动画化的人物/对象的图像
- **控制视频 (Control Video)**: 包含姿态动作的视频，用于提取姿态信息

#### 可选输入：
- **视频掩码 (Video Mask)**: 用于过滤控制视频中特定区域的掩码视频
- **文本提示词 (Prompt)**: 描述生成视频内容的文本

### 步骤 3: 配置参数

#### 基本参数：
- **Resolution (分辨率)**: 例如 `480x832` 或 `512x768`（必须是 16 的倍数）
- **Video Length (视频长度)**: 生成的帧数，例如 `81` 帧（约 5 秒，16fps）
- **Seed (随机种子)**: 用于复现结果，例如 `42`

#### SteadyDancer 专用参数：
- **Condition Guidance (条件引导)**: 默认 `2.0`，控制姿态引导的强度
  - 值越大，姿态引导越强
  - 设置为 `1.0` 可以加快处理速度（但质量可能略低）
- **Video Prompt Type (视频提示类型)**: 
  - `"V"`: 使用控制视频姿态来动画化开始图像中的人物
  - `"VA"`: 使用控制视频姿态（通过掩码视频过滤）来动画化人物
- **Image Prompt Type (图像提示类型)**: 设置为 `"S"`（使用开始图像）

### 步骤 4: 生成视频

1. 点击 **"Generate"** 按钮
2. 等待处理完成（包括姿态检测、对齐和视频生成）
3. 生成的视频将显示在输出区域

## 通过命令行使用（CLI 模式）

### 创建队列文件

首先在 Web UI 中配置好所有参数，然后保存队列：

1. 在 Web UI 中配置 SteadyDancer 参数
2. 点击 **"Save Queue"** 按钮，保存为 `.zip` 文件

### 处理队列

```bash
# 处理保存的队列文件
python3 wgp.py --process saved_queue.zip

# 指定输出目录
python3 wgp.py --process saved_queue.zip --output-dir ./outputs

# 验证队列文件（不实际生成）
python3 wgp.py --process saved_queue.zip --dry-run
```

## 通过 Python API 调用

### 基本示例

```python
from models.wan import WanAny2V
from models.wan.configs import WAN_CONFIGS
import torch
from PIL import Image

# 加载模型
cfg = WAN_CONFIGS['i2v-14B']
wan_model = WanAny2V(
    config=cfg,
    checkpoint_dir="ckpts",
    model_filename="wan2.1_steadydancer_14B_mbf16.safetensors",
    model_type="steadydancer",
    base_model_type="steadydancer",
)

# 准备输入
image_start = Image.open("reference_image.jpg")  # 参考图像
video_guide = load_video("control_video.mp4")     # 控制视频（姿态视频）
prompt = "a person dancing"                       # 文本提示词

# 生成视频
samples = wan_model.generate(
    input_prompt=prompt,
    image_start=image_start,
    input_video=video_guide,  # 控制视频
    video_mask=video_mask,     # 可选：视频掩码
    height=832,
    width=480,
    frame_num=81,
    sampling_steps=50,
    guide_scale=5.0,
    alt_guide_scale=2.0,       # Condition Guidance
    seed=42,
    video_prompt_type="VA",    # 或 "V"
    image_prompt_type="S",
)
```

## 关键参数说明

### SteadyDancer 特定参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `alt_guide_scale` | 条件引导强度（姿态引导） | 2.0 | 1.0 - 3.0 |
| `video_prompt_type` | 视频提示类型 | "VA" | "V" 或 "VA" |
| `image_prompt_type` | 图像提示类型 | "S" | "S" |
| `denoising_strength` | 去噪强度 | 1.0 | 0.7 - 1.0 |
| `sampling_steps` | 采样步数 | 50 | 30 - 80 |

### 姿态对齐参数

SteadyDancer 会自动进行姿态对齐，相关参数在 `wan_handler.py` 中配置：

- `expand_scale`: 掩码扩展比例（默认 0）
- `max_workers`: 并行处理的工作线程数（默认 1）

## 工作流程

1. **姿态检测**: 使用 DWPose 和 YOLOX 检测参考图像和控制视频中的姿态
2. **姿态对齐**: 将控制视频的姿态对齐到参考图像的人物
3. **条件编码**: 编码对齐后的姿态信息作为条件
4. **视频生成**: 使用 DC-CFG（Dual Condition Classifier-Free Guidance）生成视频
   - 条件引导仅在去噪步骤的 10%-50% 区间应用
   - 结合文本引导和姿态条件引导

## 最佳实践

### 1. 输入准备
- **参考图像**: 使用清晰、正面的人物图像，最好包含完整身体
- **控制视频**: 使用包含清晰姿态动作的视频，人物大小和位置尽量一致
- **分辨率**: 推荐使用 `480x832` 或 `512x768`，确保是 16 的倍数

### 2. 参数调优
- **快速处理**: 设置 `Condition Guidance = 1.0`
- **高质量**: 设置 `Condition Guidance = 2.0-3.0`，增加 `sampling_steps` 到 60-80
- **长视频**: 使用滑动窗口（Sliding Window）功能，设置适当的 `overlap`

### 3. 性能优化
- 使用量化模型（`quanto_int8`）可以减少显存占用
- 减小分辨率可以加快生成速度
- 使用 `batch_size=1` 以避免 OOM 错误

## 故障排除

### 问题 1: 姿态检测失败
- **原因**: 参考图像或控制视频中无法检测到人物
- **解决**: 确保输入图像/视频包含清晰可见的人物

### 问题 2: 第一帧不匹配
- **原因**: 姿态对齐失败
- **解决**: 检查控制视频的第一帧是否包含有效姿态，尝试调整 `expand_scale`

### 问题 3: GPU 内存不足 (OOM)
- **原因**: 分辨率或帧数过大
- **解决**: 
  - 减小分辨率（例如从 832x480 降到 512x384）
  - 减少帧数（例如从 81 降到 49）
  - 使用量化模型

### 问题 4: 生成速度慢
- **原因**: 采样步数过多或条件引导过强
- **解决**: 
  - 设置 `Condition Guidance = 1.0`
  - 减少 `sampling_steps` 到 30-40
  - 使用更小的分辨率

## 示例工作流

### 示例 1: 基础人物动画

```json
{
  "model_type": "steadydancer",
  "image_start": "person.jpg",
  "video_guide": "dance_video.mp4",
  "prompt": "a person dancing gracefully",
  "resolution": "480x832",
  "video_length": 81,
  "alt_guidance_scale": 2.0,
  "video_prompt_type": "V",
  "image_prompt_type": "S"
}
```

### 示例 2: 带掩码的精确控制

```json
{
  "model_type": "steadydancer",
  "image_start": "person.jpg",
  "video_guide": "dance_video.mp4",
  "video_mask": "mask_video.mp4",
  "prompt": "a person dancing",
  "resolution": "512x768",
  "video_length": 65,
  "alt_guidance_scale": 2.5,
  "video_prompt_type": "VA",
  "image_prompt_type": "S"
}
```

## 相关文件

- 模型配置: `defaults/steadydancer.json`
- 处理器: `models/wan/wan_handler.py` (第 309-325 行)
- 姿态对齐: `models/wan/steadydancer/pose_align.py`
- 模型实现: `models/wan/modules/model.py` (第 1497-1515 行)
- 生成逻辑: `models/wan/any2video.py` (第 665-1150 行)

## 参考文档

- [Wan2GP 总览](OVERVIEW.md)
- [命令行参考](CLI.md)
- [模型文档](MODELS.md)
- [故障排除](TROUBLESHOOTING.md)

