# python
import subprocess, sys
import torch

print("Python:", sys.version.splitlines()[0])
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
print("torch.backends.cudnn.version():", torch.backends.cudnn.version())

# run nvidia-smi (will fail if driver/utility not found)
try:
    out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    print("\n--- nvidia-smi output ---\n", out)
except Exception as e:
    print("\n--- nvidia-smi not available or failed ---\n", str(e))