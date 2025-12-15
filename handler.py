import runpod
from runpod.serverless.utils import rp_upload
import os
import sys
import base64
import json
import uuid
import logging
import tempfile
import traceback
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保工作目录正确（RunPod 容器中应该是 /workspace）
workspace_dir = Path("/workspace")
if workspace_dir.exists() and workspace_dir.is_dir():
    os.chdir(workspace_dir)
    logger_workspace = logging.getLogger("workspace")
    logger_workspace.info(f"工作目录设置为: {os.getcwd()}")
else:
    # 如果 /workspace 不存在，使用项目根目录
    os.chdir(project_root)
    logger_workspace = logging.getLogger("workspace")
    logger_workspace.info(f"工作目录设置为: {os.getcwd()}")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入 Wan2GP 相关模块
# 注意：需要先初始化 wgp.py 的全局变量
os.environ["GRADIO_LANG"] = "en"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# 导入 wgp 模块以初始化全局变量
# 注意：wgp.py 在导入时会执行初始化代码，包括解析参数和加载配置
import wgp

# 等待 wgp 模块初始化完成
import time
time.sleep(0.1)  # 给初始化一点时间

# 从 wgp 导入必要的函数和变量
from wgp import (
    load_models, get_model_def, get_base_model_type, get_model_handler,
    get_model_filename, get_local_model_filename, download_models,
    transformer_quantization, transformer_dtype_policy, server_config,
    model_types_handlers, models_def, args
)
from shared.utils.utils import convert_image_to_tensor, save_video, convert_tensor_to_image
from shared.utils import files_locator as fl

# 全局变量存储模型
wan_model = None
offloadobj = None
transformer_type = None

def to_nearest_multiple_of_16(value):
    """将值调整为 16 的倍数"""
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height 值必须是数字: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted

def process_input(input_data, temp_dir, output_filename, input_type):
    """处理输入数据（路径、URL 或 base64）"""
    if input_type == "path":
        logger.info(f"📁 路径输入处理: {input_data}")
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"文件不存在: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 URL 输入处理: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        import urllib.request
        urllib.request.urlretrieve(input_data, file_path)
        return file_path
    elif input_type == "base64":
        logger.info(f"🔢 Base64 输入处理")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"不支持的输入类型: {input_type}")

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """将 Base64 数据保存为文件"""
    try:
        # 处理 data URI 格式 (data:image/jpeg;base64,...)
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Base64 输入已保存到: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ Base64 解码失败: {e}")
        raise Exception(f"Base64 解码失败: {e}")

def load_video_frames(video_path, max_frames=None):
    """加载视频帧为 tensor"""
    try:
        import cv2
    except ImportError:
        raise ImportError("需要安装 opencv-python: pip install opencv-python")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and len(frames) >= max_frames:
            break
        
        # 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        frame_tensor = convert_image_to_tensor(frame_pil)
        frames.append(frame_tensor)
    
    cap.release()
    
    if not frames:
        raise ValueError(f"无法从视频中读取帧: {video_path}")
    
    # 堆叠为 (C, T, H, W)
    video_tensor = torch.stack(frames, dim=1)
    return video_tensor

def initialize_model(model_type="steadydancer"):
    """初始化 SteadyDancer 模型"""
    global wan_model, offloadobj, transformer_type
    
    if wan_model is not None and transformer_type == model_type:
        logger.info("模型已加载，跳过初始化")
        return wan_model
    
    logger.info(f"📦 加载模型: {model_type}")
    
    # 确保模型定义存在
    model_def = get_model_def(model_type)
    if model_def is None:
        raise ValueError(f"模型类型 '{model_type}' 未找到。请确保模型已正确配置。")
    
    # 加载模型
    wan_model, offloadobj = load_models(model_type, override_profile=-1)
    transformer_type = model_type
    
    logger.info("✅ 模型加载完成")
    return wan_model

def generate_steadydancer_video(
    image_start_path,
    video_guide_path,
    prompt,
    output_path,
    video_mask_path=None,
    negative_prompt="",
    resolution=(480, 832),
    video_length=81,
    seed=42,
    sampling_steps=50,
    guidance_scale=5.0,
    alt_guidance_scale=2.0,
    video_prompt_type="VA",
    image_prompt_type="S",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """使用 SteadyDancer 生成视频"""
    global wan_model
    
    logger.info(f"🚀 开始 SteadyDancer 视频生成...")
    logger.info(f"   参考图像: {image_start_path}")
    logger.info(f"   控制视频: {video_guide_path}")
    logger.info(f"   提示词: {prompt}")
    logger.info(f"   分辨率: {resolution[0]}x{resolution[1]}")
    logger.info(f"   视频长度: {video_length} 帧")
    
    # 确保模型已加载
    if wan_model is None:
        initialize_model("steadydancer")
    
    # 获取模型处理器
    base_model_type = get_base_model_type("steadydancer")
    model_handler = get_model_handler("steadydancer")
    
    # 加载输入
    logger.info("📂 加载输入文件...")
    image_start = Image.open(image_start_path).convert("RGB")
    image_start_tensor = convert_image_to_tensor(image_start).to(device)
    
    # 加载控制视频（原始格式：C, T, H, W）
    video_guide_raw = load_video_frames(video_guide_path).to(device)
    logger.info(f"   控制视频帧数: {video_guide_raw.shape[1]}")
    
    # 加载视频掩码（如果有）
    video_mask_raw = None
    if video_mask_path:
        video_mask_raw = load_video_frames(video_mask_path).to(device)
        logger.info(f"   视频掩码帧数: {video_mask_raw.shape[1]}")
    
    logger.info("✅ 输入文件加载完成")
    
    # SteadyDancer 需要先进行姿态对齐预处理
    logger.info("🔄 进行姿态对齐预处理...")
    
    # 准备 pre_video_guide：参考图像需要添加时间维度 [C, 1, H, W]
    pre_video_guide = image_start_tensor.unsqueeze(1)  # [C, 1, H, W]
    
    # 转换视频格式：custom_preprocess_video_with_mask 期望的格式
    # 根据 wgp.py 的 custom_preprocess_video_with_mask 函数：
    # - video_guide 应该是 [C, T, H, W] 格式，值在 [-1, 1] 范围
    # - 函数内部会转换为 [T, H, W, C] 并归一化
    # 但我们直接调用 custom_preprocess，它期望的格式是：
    # - video_guide: [C, T, H, W] 在 [-1, 1] 范围（根据代码分析）
    # - pre_video_guide: [C, 1, H, W] 在 [-1, 1] 范围
    
    # 调用 custom_preprocess 进行姿态对齐
    # 注意：custom_preprocess 内部会处理格式转换
    try:
        # 根据 wan_handler.py，custom_preprocess 期望：
        # - pre_video_guide: [C, T, H, W] tensor（参考图像）
        # - video_guide: 视频帧（格式由内部处理）
        # 但看代码，custom_preprocess 内部调用 PoseAligner.align，它期望 frames 是 List[np.ndarray]
        # 所以我们需要使用 custom_preprocess_video_with_mask 函数
        
        from wgp import custom_preprocess_video_with_mask
        
        # 准备参数：custom_preprocess_video_with_mask 期望 video_guide 是 [C, T, H, W] 在 [-1, 1]
        video_guide_for_preprocess = video_guide_raw  # 已经是 [C, T, H, W] 在 [-1, 1]
        video_mask_for_preprocess = video_mask_raw  # 如果有，也是 [C, T, H, W] 在 [-1, 1]
        
        # 调用预处理函数
        video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2 = custom_preprocess_video_with_mask(
            model_handler=model_handler,
            base_model_type=base_model_type,
            pre_video_guide=pre_video_guide,
            video_guide=video_guide_for_preprocess,
            video_mask=video_mask_for_preprocess,
            height=resolution[1],
            width=resolution[0],
            max_frames=video_guide_raw.shape[1],  # 使用所有帧
            start_frame=0,
            fit_canvas=None,
            fit_crop=None,
            target_fps=16,
            block_size=16,
            expand_scale=0,
        )
        
        if video_guide_processed is None or video_guide_processed.numel() == 0:
            raise ValueError("姿态对齐预处理失败：返回的视频为空")
        
        logger.info(f"✅ 姿态对齐完成: {video_guide_processed.shape}")
        
        # custom_preprocess_video_with_mask 返回的格式应该是 [C, T, H, W] 在 [-1, 1]
        input_frames = video_guide_processed.to(device)
        input_frames2 = video_guide_processed2.to(device) if video_guide_processed2 is not None else None
        
    except Exception as e:
        logger.error(f"❌ 姿态对齐预处理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # 准备 input_video：参考图像 [C, 1, H, W]
    input_video = pre_video_guide  # [C, 1, H, W]
    
    # 生成视频
    logger.info("🎬 开始生成视频...")
    logger.info(f"   采样步数: {sampling_steps}")
    logger.info(f"   文本引导: {guidance_scale}")
    logger.info(f"   条件引导: {alt_guidance_scale}")
    
    with torch.no_grad():
        samples = wan_model.generate(
            input_prompt=prompt,
            n_prompt=negative_prompt,
            image_start=None,  # SteadyDancer 使用 input_video 而不是 image_start
            input_video=input_video,  # 参考图像 [C, 1, H, W]
            input_frames=input_frames,  # 姿态对齐后的控制视频 [C, T, H, W]
            input_frames2=input_frames2,  # 增强版本（可选）
            height=resolution[1],
            width=resolution[0],
            frame_num=video_length,
            sampling_steps=sampling_steps,
            guide_scale=guidance_scale,
            alt_guide_scale=alt_guidance_scale,
            seed=seed,
            video_prompt_type=video_prompt_type,
            image_prompt_type=image_prompt_type,
        )
    
    logger.info("✅ 视频生成完成")
    
    # 保存视频
    logger.info(f"💾 保存视频到: {output_path}")
    save_video(samples, output_path, fps=16)
    logger.info("✅ 视频保存完成")
    
    return output_path

def handler(job):
    """
    RunPod handler for SteadyDancer video generation
    
    支持的输入参数:
    - model_type: 模型类型 (默认: "steadydancer")
    - prompt: 文本提示词 (必需)
    - image_start: 参考图像 (路径、URL 或 base64) (必需)
    - video_guide: 控制视频 (路径、URL 或 base64) (必需)
    - video_mask: 视频掩码 (路径、URL 或 base64) (可选)
    - negative_prompt: 负面提示词 (可选)
    - resolution: 分辨率，格式 "WIDTHxHEIGHT" (默认: "480x832")
    - video_length: 视频长度/帧数 (默认: 81)
    - seed: 随机种子 (默认: 42)
    - sampling_steps: 采样步数 (默认: 50)
    - guidance_scale: 文本引导强度 (默认: 5.0)
    - alt_guidance_scale: 条件引导强度/姿态引导 (默认: 2.0)
    - video_prompt_type: 视频提示类型 "V" 或 "VA" (默认: "VA")
    - image_prompt_type: 图像提示类型 (默认: "S")
    """
    job_input = job.get("input", {})
    
    # 记录输入（排除 base64 数据）
    log_input = {k: v for k, v in job_input.items() 
                 if k not in ["image_start", "video_guide", "video_mask"] or not isinstance(v, str) or len(v) < 100}
    logger.info(f"收到任务输入: {log_input}")
    
    task_id = f"task_{uuid.uuid4()}"
    temp_dir = os.path.join("/tmp", task_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 获取模型类型
        model_type = job_input.get("model_type", "steadydancer")
        if model_type != "steadydancer":
            logger.warning(f"模型类型 '{model_type}' 不是 steadydancer，将使用 steadydancer")
            model_type = "steadydancer"
        
        # 处理参考图像
        image_start = None
        if "image_start" in job_input:
            image_input = job_input["image_start"]
            if isinstance(image_input, str):
                # 判断是路径、URL 还是 base64
                if image_input.startswith("http://") or image_input.startswith("https://"):
                    input_type = "url"
                elif image_input.startswith("data:") or len(image_input) > 100:
                    input_type = "base64"
                else:
                    input_type = "path"
            else:
                raise ValueError("image_start 必须是字符串（路径、URL 或 base64）")
            
            image_start = process_input(image_input, temp_dir, "input_image.jpg", input_type)
        else:
            raise ValueError("缺少必需参数: image_start (参考图像)")
        
        # 处理控制视频
        video_guide = None
        if "video_guide" in job_input:
            video_input = job_input["video_guide"]
            if isinstance(video_input, str):
                if video_input.startswith("http://") or video_input.startswith("https://"):
                    input_type = "url"
                elif video_input.startswith("data:") or len(video_input) > 100:
                    input_type = "base64"
                else:
                    input_type = "path"
            else:
                raise ValueError("video_guide 必须是字符串（路径、URL 或 base64）")
            
            video_guide = process_input(video_input, temp_dir, "control_video.mp4", input_type)
        else:
            raise ValueError("缺少必需参数: video_guide (控制视频)")
        
        # 处理视频掩码（可选）
        video_mask = None
        if "video_mask" in job_input and job_input["video_mask"]:
            mask_input = job_input["video_mask"]
            if isinstance(mask_input, str):
                if mask_input.startswith("http://") or mask_input.startswith("https://"):
                    input_type = "url"
                elif mask_input.startswith("data:") or len(mask_input) > 100:
                    input_type = "base64"
                else:
                    input_type = "path"
                video_mask = process_input(mask_input, temp_dir, "video_mask.mp4", input_type)
        
        # 获取其他参数
        prompt = job_input.get("prompt", "a person dancing")
        negative_prompt = job_input.get("negative_prompt", "")
        
        # 解析分辨率
        resolution_str = job_input.get("resolution", "480x832")
        width, height = map(int, resolution_str.split('x'))
        width = to_nearest_multiple_of_16(width)
        height = to_nearest_multiple_of_16(height)
        resolution = (width, height)
        
        video_length = job_input.get("video_length", 81)
        seed = job_input.get("seed", 42)
        sampling_steps = job_input.get("sampling_steps", 50)
        guidance_scale = job_input.get("guidance_scale", 5.0)
        alt_guidance_scale = job_input.get("alt_guidance_scale", 2.0)
        video_prompt_type = job_input.get("video_prompt_type", "VA" if video_mask else "V")
        image_prompt_type = job_input.get("image_prompt_type", "S")
        
        # 生成输出路径
        output_path = os.path.join(temp_dir, "output_video.mp4")
        
        # 生成视频
        generate_steadydancer_video(
            image_start_path=image_start,
            video_guide_path=video_guide,
            prompt=prompt,
            output_path=output_path,
            video_mask_path=video_mask,
            negative_prompt=negative_prompt,
            resolution=resolution,
            video_length=video_length,
            seed=seed,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
            alt_guidance_scale=alt_guidance_scale,
            video_prompt_type=video_prompt_type,
            image_prompt_type=image_prompt_type,
        )
        
        # 读取生成的视频并转换为 base64
        logger.info("📤 准备返回视频...")
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info("✅ 任务完成")
        return {
            "video": video_base64,
            "format": "mp4",
            "resolution": f"{width}x{height}",
            "frames": video_length,
        }
        
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"❌ 生成失败: {error_message}")
        logger.error(f"错误详情:\n{error_traceback}")
        
        # 清理临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "error": error_message,
            "traceback": error_traceback
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
