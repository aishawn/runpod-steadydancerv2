#!/usr/bin/env python3
"""
SteadyDancer 使用示例

这个脚本演示了如何通过 Python API 调用 SteadyDancer 生成视频。

使用方法:
    python examples/steadydancer_example.py \
        --image_start path/to/reference_image.jpg \
        --video_guide path/to/control_video.mp4 \
        --prompt "a person dancing" \
        --output output_video.mp4
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
import numpy as np
from shared.utils.utils import convert_image_to_tensor, save_video


def load_video_frames(video_path, max_frames=None):
    """
    加载视频帧
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数（None 表示加载所有帧）
    
    Returns:
        torch.Tensor: 形状为 (C, T, H, W) 的视频张量，值范围 [-1, 1]
    """
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
        # 转换为 PIL Image 然后到 tensor
        frame_pil = Image.fromarray(frame)
        frame_tensor = convert_image_to_tensor(frame_pil)
        frames.append(frame_tensor)
    
    cap.release()
    
    if not frames:
        raise ValueError(f"无法从视频中读取帧: {video_path}")
    
    # 堆叠为 (C, T, H, W)
    video_tensor = torch.stack(frames, dim=1)
    return video_tensor


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
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    使用 SteadyDancer 生成视频
    
    Args:
        image_start_path: 参考图像路径
        video_guide_path: 控制视频路径（包含姿态动作）
        prompt: 文本提示词
        output_path: 输出视频路径
        video_mask_path: 可选的视频掩码路径
        negative_prompt: 负面提示词
        resolution: 分辨率 (width, height)
        video_length: 视频长度（帧数）
        seed: 随机种子
        sampling_steps: 采样步数
        guidance_scale: 文本引导强度
        alt_guidance_scale: 条件引导强度（姿态引导）
        device: 计算设备
    """
    print(f"🚀 开始 SteadyDancer 视频生成...")
    print(f"   参考图像: {image_start_path}")
    print(f"   控制视频: {video_guide_path}")
    print(f"   提示词: {prompt}")
    print(f"   分辨率: {resolution[0]}x{resolution[1]}")
    print(f"   视频长度: {video_length} 帧")
    
    # 加载模型
    print("\n📦 加载模型...")
    from models.wan import WanAny2V
    from models.wan.configs import WAN_CONFIGS
    from models.wan.wan_handler import family_handler
    
    cfg = WAN_CONFIGS['i2v-14B']
    model_filename = "wan2.1_steadydancer_14B_mbf16.safetensors"
    
    wan_model = WanAny2V(
        config=cfg,
        checkpoint_dir="ckpts",
        model_filename=model_filename,
        model_type="steadydancer",
        base_model_type="steadydancer",
        dtype=torch.bfloat16,
    )
    wan_model.model.to(device)
    print("✅ 模型加载完成")
    
    # 加载输入
    print("\n📂 加载输入文件...")
    image_start = Image.open(image_start_path).convert("RGB")
    image_start_tensor = convert_image_to_tensor(image_start).to(device)
    
    video_guide = load_video_frames(video_guide_path).to(device)
    print(f"   控制视频帧数: {video_guide.shape[1]}")
    
    video_mask = None
    if video_mask_path:
        video_mask = load_video_frames(video_mask_path).to(device)
        print(f"   视频掩码帧数: {video_mask.shape[1]}")
    
    print("✅ 输入文件加载完成")
    
    # 生成视频
    print("\n🎬 开始生成视频...")
    print(f"   采样步数: {sampling_steps}")
    print(f"   文本引导: {guidance_scale}")
    print(f"   条件引导: {alt_guidance_scale}")
    
    with torch.no_grad():
        samples = wan_model.generate(
            input_prompt=prompt,
            n_prompt=negative_prompt,
            image_start=image_start_tensor,
            input_video=video_guide,
            video_mask=video_mask,
            height=resolution[1],
            width=resolution[0],
            frame_num=video_length,
            sampling_steps=sampling_steps,
            guide_scale=guidance_scale,
            alt_guide_scale=alt_guide_scale,
            seed=seed,
            video_prompt_type="VA" if video_mask else "V",
            image_prompt_type="S",
        )
    
    print("✅ 视频生成完成")
    
    # 保存视频
    print(f"\n💾 保存视频到: {output_path}")
    save_video(samples, output_path, fps=16)
    print("✅ 视频保存完成")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="SteadyDancer 视频生成示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础用法
  python examples/steadydancer_example.py \\
      --image_start person.jpg \\
      --video_guide dance.mp4 \\
      --prompt "a person dancing gracefully" \\
      --output result.mp4

  # 带掩码的精确控制
  python examples/steadydancer_example.py \\
      --image_start person.jpg \\
      --video_guide dance.mp4 \\
      --video_mask mask.mp4 \\
      --prompt "a person dancing" \\
      --output result.mp4 \\
      --alt_guidance_scale 2.5
        """
    )
    
    parser.add_argument(
        "--image_start",
        type=str,
        required=True,
        help="参考图像路径（包含要动画化的人物）"
    )
    parser.add_argument(
        "--video_guide",
        type=str,
        required=True,
        help="控制视频路径（包含姿态动作）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文本提示词"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="steadydancer_output.mp4",
        help="输出视频路径（默认: steadydancer_output.mp4）"
    )
    parser.add_argument(
        "--video_mask",
        type=str,
        default=None,
        help="可选的视频掩码路径"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="负面提示词"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480x832",
        help="分辨率，格式: WIDTHxHEIGHT（默认: 480x832）"
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=81,
        help="视频长度（帧数，默认: 81）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="采样步数（默认: 50）"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="文本引导强度（默认: 5.0）"
    )
    parser.add_argument(
        "--alt_guidance_scale",
        type=float,
        default=2.0,
        help="条件引导强度/姿态引导（默认: 2.0）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（默认: 自动检测）"
    )
    
    args = parser.parse_args()
    
    # 解析分辨率
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # 确定设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🖥️  使用设备: {device}")
    if device == "cpu":
        print("⚠️  警告: 使用 CPU 模式，生成速度会很慢")
    
    # 检查输入文件
    if not os.path.exists(args.image_start):
        print(f"❌ 错误: 参考图像不存在: {args.image_start}")
        sys.exit(1)
    
    if not os.path.exists(args.video_guide):
        print(f"❌ 错误: 控制视频不存在: {args.video_guide}")
        sys.exit(1)
    
    if args.video_mask and not os.path.exists(args.video_mask):
        print(f"❌ 错误: 视频掩码不存在: {args.video_mask}")
        sys.exit(1)
    
    # 生成视频
    try:
        output_path = generate_steadydancer_video(
            image_start_path=args.image_start,
            video_guide_path=args.video_guide,
            prompt=args.prompt,
            output_path=args.output,
            video_mask_path=args.video_mask,
            negative_prompt=args.negative_prompt,
            resolution=resolution,
            video_length=args.video_length,
            seed=args.seed,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            alt_guidance_scale=args.alt_guidance_scale,
            device=device,
        )
        print(f"\n🎉 成功！视频已保存到: {output_path}")
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

