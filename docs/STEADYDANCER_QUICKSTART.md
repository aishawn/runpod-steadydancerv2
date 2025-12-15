# SteadyDancer 快速开始

## 🚀 快速启动

```bash
# 启动应用
python3 wgp.py --listen

# 访问 Web UI
# 浏览器打开: http://localhost:7860
```

## 📋 必需输入

1. **参考图像** (Image Start): 包含要动画化的人物图像
2. **控制视频** (Control Video): 包含姿态动作的视频

## ⚙️ 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **Model** | 选择模型 | `Wan2.1 Steady Dancer 14B` |
| **Resolution** | 分辨率 | `480x832` 或 `512x768` |
| **Video Length** | 视频长度 | `81` 帧（约 5 秒） |
| **Condition Guidance** | 条件引导 | `2.0`（快速用 `1.0`） |
| **Video Prompt Type** | 视频提示类型 | `V` 或 `VA` |
| **Image Prompt Type** | 图像提示类型 | `S` |

## 🎯 使用步骤

### Web UI 方式

1. **选择模型**: 下拉菜单选择 `Wan2.1 Steady Dancer 14B`
2. **上传参考图像**: 点击 "Image Start" 上传人物图像
3. **上传控制视频**: 点击 "Control Video" 上传姿态视频
4. **输入提示词**: 例如 "a person dancing gracefully"
5. **设置参数**: 
   - Resolution: `480x832`
   - Video Length: `81`
   - Condition Guidance: `2.0`
6. **生成**: 点击 "Generate" 按钮

### 命令行方式

```bash
# 1. 在 Web UI 中配置好参数并保存队列
# 2. 处理队列
python3 wgp.py --process saved_queue.zip --output-dir ./outputs
```

### Python API 方式

```python
from examples.steadydancer_example import generate_steadydancer_video

generate_steadydancer_video(
    image_start_path="person.jpg",
    video_guide_path="dance.mp4",
    prompt="a person dancing",
    output_path="result.mp4",
    resolution=(480, 832),
    video_length=81,
    alt_guidance_scale=2.0,
)
```

## 💡 提示

### ✅ 最佳实践

- **参考图像**: 使用清晰、正面的人物图像
- **控制视频**: 使用包含清晰姿态动作的视频
- **分辨率**: 确保是 16 的倍数（如 480x832）
- **快速测试**: 设置 `Condition Guidance = 1.0`

### ⚠️ 常见问题

- **OOM 错误**: 减小分辨率或帧数
- **姿态检测失败**: 确保图像/视频包含清晰可见的人物
- **生成速度慢**: 降低 `Condition Guidance` 或减少 `sampling_steps`

## 📚 更多信息

详细文档请参考: [SteadyDancer 完整使用指南](STEADYDANCER_USAGE.md)

