import torch

print("=" * 60)
print("PyTorch CUDA Test")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print("=" * 60)

if torch.cuda.is_available():
    print("GPU Information:")
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print("=" * 60)
    print("SUCCESS: RTX 5080 CUDA is working!")
else:
    print("CUDA NOT AVAILABLE")
    print("Possible reasons:")
    print("  1. PyTorch nightly does not support RTX 5080 (sm_120) yet")
    print("  2. NVIDIA driver version is too old")
    print("  3. CUDA Toolkit version mismatch")
    print()
    print("Recommendation: Use CPU training for now")

print("=" * 60)
