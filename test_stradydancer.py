#!/usr/bin/env python3
"""
SteadyDancer 视频生成测试脚本
基于 MCG-NJU/SteadyDancer 模型进行人体动画生成
"""

import os
import sys
import base64
import requests
import json
import time
from pathlib import Path

# 加载环境变量
def load_env():
    env_file = os.path.join(os.path.dirname(__file__), '.env.local')
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ 已加载环境变量文件: {env_file}")
    else:
        print(f"⚠️ 未找到 .env 文件: {env_file}")

load_env()


class SteadyDancerGenerator:
    """SteadyDancer 视频生成器"""
    
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.api_endpoint = os.getenv("RUNPOD_API_ENDPOINT_STEADYDANCER", "")
        
        if not self.api_key:
            raise ValueError("❌ 请设置 RUNPOD_API_KEY 环境变量")
        if not self.api_endpoint:
            raise ValueError("❌ 请设置 RUNPOD_API_ENDPOINT_STEADYDANCER 环境变量")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"🔧 API Endpoint: {self.api_endpoint}")
    
    def encode_file_to_base64(self, file_path: str) -> str:
        """将文件编码为 base64"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def generate_video(
        self,
        image_path: str,
        video_path: str,
        prompt: str = "A person dancing gracefully",
        negative_prompt: str = "",
        width: int = 480,
        height: int = 832,
        video_length: int = 81,
        seed: int = 42,
        sampling_steps: int = 50,
        guidance_scale: float = 5.0,
        alt_guidance_scale: float = 2.0,
        video_mask_path: str = None,
        video_prompt_type: str = "VA",
        image_prompt_type: str = "S",
    ):
        """
        生成 SteadyDancer 动画视频
        
        参数:
            image_path: 参考图像路径（起始帧）
            video_path: 输入视频路径（用于姿态检测）
            prompt: 提示词
            negative_prompt: 负面提示词
            width: 视频宽度（必须是16的倍数，会自动调整）
            height: 视频高度（必须是16的倍数，会自动调整）
            video_length: 视频长度（帧数）
            seed: 随机种子
            sampling_steps: 采样步数
            guidance_scale: 文本引导强度（CFG scale）
            alt_guidance_scale: 条件引导强度/姿态引导
            video_mask_path: 可选的视频掩码路径
            video_prompt_type: 视频提示类型 "V" 或 "VA"（默认: "VA"，如果有掩码则自动使用 "VA"）
            image_prompt_type: 图像提示类型（默认: "S"）
        """
        
        print(f"🚀 开始生成 SteadyDancer 视频...")
        print(f"📷 参考图像: {image_path}")
        print(f"🎬 输入视频: {video_path}")
        print(f"📝 提示词: {prompt}")
        print(f"📐 尺寸: {width}x{height}")
        print(f"🎞️ 长度: {video_length} 帧 (约 {video_length/16:.1f} 秒)")
        
        # 检查文件存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 图像文件不存在: {image_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ 视频文件不存在: {video_path}")
        
        # 编码文件为 base64
        print("🔄 编码文件...")
        image_base64 = self.encode_file_to_base64(image_path)
        video_base64 = self.encode_file_to_base64(video_path)
        print(f"✅ 图像编码完成: {len(image_base64)} 字符")
        print(f"✅ 视频编码完成: {len(video_base64)} 字符")
        
        # 如果有视频掩码，也编码
        video_mask_base64 = None
        if video_mask_path:
            if not os.path.exists(video_mask_path):
                print(f"⚠️ 视频掩码文件不存在: {video_mask_path}，将忽略")
            else:
                video_mask_base64 = self.encode_file_to_base64(video_mask_path)
                print(f"✅ 视频掩码编码完成: {len(video_mask_base64)} 字符")
                # 如果有掩码，自动使用 "VA" 类型
                if video_prompt_type == "V":
                    video_prompt_type = "VA"
        
        # 准备请求数据（匹配 handler.py 的接口）
        payload = {
            "input": {
                "model_type": "steadydancer",  # 指定使用 SteadyDancer 模型
                "image_start": image_base64,  # handler.py 期望的参数名
                "video_guide": video_base64,  # handler.py 期望的参数名
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "resolution": f"{width}x{height}",  # handler.py 期望的格式
                "video_length": video_length,
                "seed": seed,
                "sampling_steps": sampling_steps,
                "guidance_scale": guidance_scale,
                "alt_guidance_scale": alt_guidance_scale,
                "video_prompt_type": video_prompt_type,
                "image_prompt_type": image_prompt_type,
            }
        }
        
        # 如果有视频掩码，添加到请求中
        if video_mask_base64:
            payload["input"]["video_mask"] = video_mask_base64
        
        try:
            # 发送请求
            print(f"\n📤 提交任务到 RunPod...")
            # RunPod serverless 使用 /run 端点
            # 如果 endpoint 已经包含 /run，直接使用；否则添加
            base_url = self.api_endpoint.rstrip('/')
            if not base_url.endswith('/run'):
                submit_url = f"{base_url}/run"
            else:
                submit_url = base_url
            
            print(f"📡 请求 URL: {submit_url}")
            response = requests.post(
                submit_url,
                headers=self.headers,
                json=payload,
                timeout=(10, 30)  # 连接超时10秒，读取超时30秒
            )
            response.raise_for_status()
            
            result = response.json()
            job_id = result.get('id')
            
            if not job_id:
                print(f"❌ 未返回任务ID: {result}")
                return None
            
            print(f"✅ 任务已提交!")
            print(f"🆔 任务ID: {job_id}")
            
            return job_id
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应内容: {e.response.text}")
            return None
    
    def check_status(self, job_id: str, max_retries: int = 3, retry_delay: int = 2):
        """
        检查任务状态，带重试机制
        
        参数:
            job_id: 任务ID
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        # RunPod API: /run endpoint 用于提交，/status endpoint 用于查询
        # 构建状态查询URL
        base_url = self.api_endpoint.rstrip('/run').rstrip('/runsync').rstrip('/')
        # RunPod serverless 使用 /status/{job_id} 端点
        status_url = f"{base_url}/status/{job_id}"
        
        # 可重试的异常类型（网络错误）
        retryable_exceptions = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
        )
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                # 设置超时：连接超时5秒，读取超时10秒
                response = requests.get(
                    status_url,
                    headers=self.headers,
                    timeout=(5, 10)
                )
                response.raise_for_status()
                return response.json()
            except retryable_exceptions as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # 指数退避
                    print(f"⚠️ 网络错误（尝试 {attempt + 1}/{max_retries}）: {e}")
                    print(f"⏳ {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 检查状态失败（已重试 {max_retries} 次）: {e}")
                    raise
            except requests.exceptions.RequestException as e:
                # 非网络错误（如4xx, 5xx），不重试
                print(f"❌ 检查状态失败: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"响应内容: {e.response.text}")
                raise
        
        # 如果所有重试都失败
        if last_exception:
            raise last_exception
    
    def wait_for_completion(self, job_id: str, check_interval: int = 10, max_wait_time: int = 3600):
        """
        等待任务完成
        
        参数:
            job_id: 任务ID
            check_interval: 检查间隔（秒）
            max_wait_time: 最大等待时间（秒）
        """
        print(f"\n⏳ 等待任务 {job_id} 完成...")
        print(f"🔄 检查间隔: {check_interval} 秒")
        print(f"⏱️ 最大等待时间: {max_wait_time} 秒")
        
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 5  # 连续错误的最大次数
        
        while True:
            elapsed = time.time() - start_time
            
            # 检查是否超时
            if elapsed > max_wait_time:
                raise TimeoutError(f"等待任务完成超时（已等待 {int(elapsed)} 秒）")
            
            try:
                # 查询状态
                result = self.check_status(job_id)
                consecutive_errors = 0  # 重置连续错误计数
                
                status = result.get('status', 'UNKNOWN')
                
                # 打印进度
                print(f"📊 状态: {status} (已等待 {int(elapsed)} 秒)", end='\r')
                
                # 检查是否完成
                if status == 'COMPLETED':
                    print(f"\n\n✅ 任务完成!")
                    return result
                elif status == 'FAILED':
                    print(f"\n\n❌ 任务失败!")
                    error = result.get('error', '未知错误')
                    print(f"错误信息: {error}")
                    return result
                elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                    # 继续等待
                    print(f"\n⏳ 任务进行中，等待 {check_interval} 秒...")
                    time.sleep(check_interval)
                else:
                    print(f"\n⚠️ 未知状态: {status}")
                    time.sleep(check_interval)
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise Exception(f"连续 {max_consecutive_errors} 次检查状态失败，可能网络连接有问题: {e}")
                
                # 等待一段时间后重试
                wait_time = min(check_interval, 30)  # 最多等待30秒
                print(f"\n⚠️ 网络错误，{wait_time} 秒后重试... (连续错误: {consecutive_errors}/{max_consecutive_errors})")
                time.sleep(wait_time)
            except requests.exceptions.RequestException as e:
                # 其他请求错误（如4xx, 5xx），不重试
                print(f"\n❌ 检查状态失败: {e}")
                raise
    
    def download_result(self, result: dict, output_dir: str = "./outputs"):
        """
        下载生成的视频
        
        参数:
            result: 任务结果
            output_dir: 输出目录
        """
        if result.get('status') != 'COMPLETED':
            print(f"❌ 任务未完成，无法下载")
            return None
        
        output_data = result.get('output', {})
        video_base64 = output_data.get('video')
        
        if not video_base64:
            print(f"❌ 未找到视频数据")
            return None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"steadydancer_{timestamp}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        # 解码并保存视频
        try:
            print(f"\n📥 下载视频...")
            video_data = base64.b64decode(video_base64)
            
            with open(output_path, 'wb') as f:
                f.write(video_data)
            
            file_size_mb = len(video_data) / (1024 * 1024)
            print(f"✅ 视频已保存: {output_path}")
            print(f"📦 文件大小: {file_size_mb:.2f} MB")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 保存视频失败: {e}")
            return None


def test_steadydancer():
    """测试 SteadyDancer 视频生成"""
    
    print("🧪 测试 SteadyDancer 视频生成器")
    print("=" * 60)
    
    # 配置参数
    image_path = r"C:\shawn\1code\1project-cur\video-edit\code\runpod\steadydancer\SteadyDancer\data\images\00001.png"
    video_path = r"C:\shawn\1code\1project-cur\video-edit\code\runpod\steadydancer\SteadyDancer\data\videos\00002\video.mp4"
    
    prompt = "A person dancing gracefully with smooth movements"
    negative_prompt = "static, blurry, low quality, distorted, bad anatomy"
    
    # 视频参数（匹配 handler.py 的接口）
    width = 480  # 必须是16的倍数（handler.py 会自动调整）
    height = 832  # 必须是16的倍数（handler.py 会自动调整）
    video_length = 81  # 约5秒 (81帧 / 16fps)
    seed = 42
    
    # 采样参数（匹配 handler.py 的接口）
    sampling_steps = 50  # 采样步数
    guidance_scale = 5.0  # 文本引导强度（CFG scale）
    alt_guidance_scale = 2.0  # 条件引导强度/姿态引导
    
    # 视频提示类型
    video_prompt_type = "VA"  # "V" 或 "VA"（如果有掩码建议用 "VA"）
    image_prompt_type = "S"  # 图像提示类型
    
    # 可选的视频掩码路径
    video_mask_path = None  # 如果有掩码视频，设置路径
    
    # 检查文件
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        print("请修改 image_path 变量为实际图像路径")
        return False
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        print("请修改 video_path 变量为实际视频路径")
        return False
    
    try:
        # 创建生成器
        generator = SteadyDancerGenerator()
        
        print(f"\n📋 配置参数:")
        print(f"  参考图像: {image_path}")
        print(f"  输入视频: {video_path}")
        if video_mask_path:
            print(f"  视频掩码: {video_mask_path}")
        print(f"  提示词: {prompt}")
        print(f"  尺寸: {width}x{height} (会自动调整为16的倍数)")
        print(f"  长度: {video_length} 帧")
        print(f"  采样步数: {sampling_steps}")
        print(f"  文本引导: {guidance_scale}")
        print(f"  条件引导: {alt_guidance_scale}")
        print(f"  视频提示类型: {video_prompt_type}")
        print(f"  图像提示类型: {image_prompt_type}")
        print(f"  种子: {seed}")
        
        # 生成视频
        job_id = generator.generate_video(
            image_path=image_path,
            video_path=video_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            video_length=video_length,
            seed=seed,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
            alt_guidance_scale=alt_guidance_scale,
            video_mask_path=video_mask_path,
            video_prompt_type=video_prompt_type,
            image_prompt_type=image_prompt_type,
        )
        
        if not job_id:
            print("❌ 未返回任务ID")
            return False
        
        # 等待完成
        final_result = generator.wait_for_completion(job_id)
        
        if final_result.get('status') == 'COMPLETED':
            print("🎉 视频生成成功!")
            
            # 下载视频
            video_path_out = generator.download_result(final_result)
            if video_path_out:
                print(f"📁 完整路径: {os.path.abspath(video_path_out)}")
                return True
            else:
                return False
        else:
            print(f"❌ 视频生成失败: {final_result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🔧 SteadyDancer 视频生成测试")
    print("=" * 60)
    
    # 检查环境变量
    if not os.getenv("RUNPOD_API_KEY"):
        print("❌ 请设置 RUNPOD_API_KEY 环境变量")
        return 1
    
    if not os.getenv("RUNPOD_API_ENDPOINT_STEADYDANCER"):
        print("❌ 请设置 RUNPOD_API_ENDPOINT_STEADYDANCER 环境变量")
        print("提示: 在 .env 文件中添加 RUNPOD_API_ENDPOINT_STEADYDANCER=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID")
        print("注意: 不需要包含 /run 或 /runsync，脚本会自动添加")
        return 1
    
    # 运行测试
    success = test_steadydancer()
    
    if success:
        print("\n🎊 测试成功!")
        return 0
    else:
        print("\n💔 测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

