import torch
import time

def check_gpu():
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available! Found {device_count} GPU(s).")

        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")


        try:
            print("\n Running a test computation on GPU...")
            device = torch.device("cuda")
            x = torch.rand(10000, 10000).to(device)
            y = torch.rand(10000, 10000).to(device)

            start_time = time.time()
            z = torch.matmul(x, y)
            end_time = time.time()

            print(f"   Success! Computation took {end_time - start_time:.4f} seconds.")
            print("   Your GPU is ready for training.")

        except Exception as e:
            print(f" Error during computation: {e}")

    else:
        print(" CUDA is NOT available. PyTorch is running on CPU.")
        print("   Run this command to install the correct version:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu()