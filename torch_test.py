import torch

# Check CUDA availability and details
print("CUDA available:", torch.cuda.is_available())
#print("CUDA device count:", torch.cuda.device_count())
print("PyTorch Version:", torch.__version__)  # Check PyTorch version
print("PyTorch CUDA Version:",torch.version.cuda)  # Check CUDA version PyTorch was built with

if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print(x)
else:
    print("CUDA is not available.")