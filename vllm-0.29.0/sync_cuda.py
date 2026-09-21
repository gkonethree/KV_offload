import subprocess
import sys
from importlib.metadata import requires

# 1. Extract PyTorch's exact required NVIDIA packages
raw_reqs = requires('torch')
nvidia_reqs = []
runtime_version = None

for req in raw_reqs:
    if 'nvidia' in req.lower():
        # Clean formatting, e.g., 'nvidia-cuda-runtime (== 13.0.88)' -> 'nvidia-cuda-runtime==13.0.88'
        clean_req = req.split(';')[0].strip().replace('(', '').replace(')', '').replace(' ', '')
        nvidia_reqs.append(clean_req)
        if 'runtime' in clean_req:
            runtime_version = clean_req.split('==')[1]

print(f"Restoring PyTorch's perfectly matched dependencies:\n{nvidia_reqs}\n")

# 2. Force reinstall all required runtime and cublas packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall"] + nvidia_reqs)

# 3. Install matching nvcc compiler
if runtime_version:
    print(f"\nInstalling NVCC compiler version {runtime_version} to match the runtime perfectly...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"nvidia-cuda-nvcc=={runtime_version}"])
    except subprocess.CalledProcessError:
        print("\nFallback: trying with -cu13 suffix...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"nvidia-cuda-nvcc-cu13=={runtime_version}"])
