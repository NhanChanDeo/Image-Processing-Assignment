import torch; 
print('PyTorch version:', torch.__version__); 
print('CUDA available:', torch.cuda.is_available()); 
print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); 
print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')

# Check cuBLAS performance
print(f"cuBLAS Enabled: {torch.backends.cuda.is_built()}")
print(f"cuDNN Enabled: {torch.backends.cudnn.is_available()}")