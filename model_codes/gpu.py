import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU.")
