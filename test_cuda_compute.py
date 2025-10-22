import torch
import time

print("=" * 60)
print("RTX 5080 Real Compute Test")
print("=" * 60)

if not torch.cuda.is_available():
    print("CUDA not available. Exiting.")
    exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")
print()

# Test 1: Simple tensor operation
print("[Test 1] Simple tensor creation on GPU...")
try:
    x = torch.randn(1000, 1000, device='cuda')
    print("  SUCCESS: Tensor created on GPU")
except Exception as e:
    print(f"  FAILED: {e}")
    exit(1)

# Test 2: Matrix multiplication
print("[Test 2] Matrix multiplication on GPU...")
try:
    a = torch.randn(2000, 2000, device='cuda')
    b = torch.randn(2000, 2000, device='cuda')
    torch.cuda.synchronize()

    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"  SUCCESS: GPU computation time: {gpu_time:.4f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    exit(1)

# Test 3: CPU vs GPU speed comparison
print("[Test 3] Speed comparison (CPU vs GPU)...")
try:
    # CPU test
    a_cpu = torch.randn(3000, 3000)
    b_cpu = torch.randn(3000, 3000)
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start

    # GPU test
    a_gpu = torch.randn(3000, 3000, device='cuda')
    b_gpu = torch.randn(3000, 3000, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"  CPU time: {cpu_time:.4f}s")
    print(f"  GPU time: {gpu_time:.4f}s")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")

    if gpu_time < cpu_time:
        print("  SUCCESS: GPU is faster than CPU!")
    else:
        print("  WARNING: GPU is slower than CPU (might not be working properly)")

except Exception as e:
    print(f"  FAILED: {e}")
    exit(1)

print()
print("=" * 60)
print("FINAL RESULT:")

if gpu_time < cpu_time * 0.5:  # GPU should be at least 2x faster
    print("VERDICT: RTX 5080 is WORKING for training!")
    print("You can use device='cuda' in config.yml")
else:
    print("VERDICT: RTX 5080 detected but may not work properly")
    print("Recommendation: Use device='cpu' for now")

print("=" * 60)
