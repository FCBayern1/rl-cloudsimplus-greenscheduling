"""
CUDA兼容性诊断脚本
检查RTX 5080与PyTorch的兼容性
"""

import sys
import subprocess

print("=" * 60)
print("CUDA兼容性诊断报告")
print("=" * 60)
print()

# 1. 检查nvidia-smi
print("[1/5] 检查NVIDIA驱动和CUDA版本...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    for line in lines[:10]:  # 前10行通常包含驱动和CUDA信息
        if 'Driver Version' in line or 'CUDA Version' in line:
            print(f"  {line.strip()}")
    print()
except Exception as e:
    print(f"  ❌ 无法运行nvidia-smi: {e}")
    print()

# 2. 检查PyTorch
print("[2/5] 检查PyTorch版本和CUDA支持...")
try:
    import torch
    print(f"  ✓ PyTorch版本: {torch.__version__}")
    print(f"  ✓ CUDA是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  ✓ CUDA版本: {torch.version.cuda}")
        print(f"  ✓ cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  ✓ 可用GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
            print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("  ❌ CUDA不可用")
        print()

        # 尝试获取错误信息
        try:
            _ = torch.zeros(1).cuda()
        except Exception as e:
            print(f"  ❌ CUDA错误: {e}")
    print()

except ImportError:
    print("  ❌ PyTorch未安装")
    print()

# 3. 检查PyTorch支持的compute capabilities
print("[3/5] 检查PyTorch支持的CUDA Compute Capabilities...")
try:
    import torch
    if hasattr(torch.cuda, 'get_arch_list'):
        archs = torch.cuda.get_arch_list()
        print(f"  PyTorch编译时支持的架构: {archs}")

        # 检查是否支持sm_120
        if 'sm_120' in archs or '12.0' in archs:
            print("  ✓ 支持sm_120 (RTX 5080)")
        else:
            print("  ❌ 不支持sm_120 (RTX 5080)")
            print(f"  支持的最高版本: {max(archs) if archs else 'Unknown'}")
    print()
except:
    print("  无法获取架构列表")
    print()

# 4. 检查stable-baselines3
print("[4/5] 检查stable-baselines3...")
try:
    import stable_baselines3
    print(f"  ✓ stable-baselines3版本: {stable_baselines3.__version__}")
    print()
except ImportError:
    print("  ❌ stable-baselines3未安装")
    print()

# 5. 推荐方案
print("[5/5] 诊断结果和推荐方案")
print("-" * 60)

try:
    import torch
    if torch.cuda.is_available():
        print("✓ CUDA可用，GPU训练已就绪！")
    else:
        print("❌ CUDA不可用，建议采取以下措施：")
        print()
        print("方案1: 安装PyTorch Nightly版本")
        print("  pip uninstall torch torchvision torchaudio")
        print("  pip install --pre torch torchvision torchaudio \\")
        print("    --index-url https://download.pytorch.org/whl/nightly/cu124")
        print()
        print("方案2: 检查CUDA Toolkit版本")
        print("  运行: nvidia-smi")
        print("  确保CUDA Version >= 12.4")
        print("  如果版本太低，下载安装最新CUDA Toolkit:")
        print("  https://developer.nvidia.com/cuda-downloads")
        print()
        print("方案3: 暂时使用CPU训练")
        print("  在config.yml中设置: device: 'cpu'")
        print("  或设置环境变量: $env:DEVICE = 'cpu'")
except ImportError:
    print("❌ PyTorch未安装")
    print("请先安装PyTorch:")
    print("  pip install torch torchvision torchaudio")

print()
print("=" * 60)
print("诊断完成！")
print("=" * 60)
