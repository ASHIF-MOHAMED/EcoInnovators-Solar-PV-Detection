import os
import torch
from ultralytics import YOLO
from roboflow import Roboflow

# ==============================================================================
# 🛠️ CONFIGURATION
# ==============================================================================
# GPU SETTINGS:
# Batch size 4 is safe for 'yolo11l' on most GPUs (8GB VRAM). 
# If you have 16GB+ VRAM, you can try batch=8. 
# If you get "CUDA Out of Memory", change to batch=2.
BATCH_SIZE = 4 
EPOCHS = 50
IMAGE_SIZE = 640

def check_gpu():
    """Verifies that the GPU is correctly detected."""
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Usage: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        return 0 # Device ID 0
    else:
        print("❌ GPU NOT Detected. Training will be VERY slow on CPU.")
        return 'cpu'

def main():
    print("🚀 Starting Solar Panel Verification Training Pipeline...")
    device = check_gpu()


    print("\nDownloading Dataset...")
    

    rf = Roboflow(api_key="QFGilUHQjFBmh2ZptIds")
    project = rf.workspace("ashif-projects2").project("source1_usable-0bb5e")
    version = project.version(2)
    dataset = version.download("yolov11")

    print(f"✅ Dataset Path: {dataset.location}")


    print("\n🛠️ Verifying Folder Structure...")
    
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    # Ensure train/valid/test folders exist even if empty
    required_folders = ['train/images', 'valid/images', 'test/images']
    for folder in required_folders:
        folder_path = os.path.join(dataset.location, folder)
        if not os.path.exists(folder_path):
            print(f"   ⚠️ Creating missing folder: {folder_path}")
            os.makedirs(folder_path, exist_ok=True)
        else:
            print(f"   OK: {folder}")

    print("\n🧠 Loading YOLOv11-Large Segmentation Model...")
    model = YOLO('yolo11l-seg.pt') 

    print("\n STARTING TRAINING LOOP...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Device: {device}")
    
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        plots=True,       # Saves training graphs
        save=True,        # Saves best.pt
        
        # WINDOWS SPECIFIC SETTINGS:
        workers=2,        # Prevents "Page File" crash on Windows
        exist_ok=True     # Allows overwriting previous run folder if needed
    )

    print("\n✅ Training Complete!")
    print(f"   Best Model Saved at: runs/segment/train/weights/best.pt")


if __name__ == '__main__':
    main()