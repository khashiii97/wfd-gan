import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Print whether CUDA is available
print(f"CUDA available: {cuda_available}")

# If CUDA is available, print the number of GPUs and their names
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")
