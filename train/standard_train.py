
import tqdm
from ultralytics.utils import LOGGER
LOGGER.setLevel(60)

# disable tqdm globally after Ultralytics has loaded
tqdm.tqdm = lambda *args, **kwargs: iter(args[0])

import os
import torch
from ultralytics.models.yolo.model import YOLO
import argparse
from tqdm import tqdm  # this still works

def main(num_epochs=500, user_store_dir='tmp'):

    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. Training will run on CPU.")

    train_dir = os.path.dirname(os.path.abspath(__file__))
    megatron_dir = os.path.dirname(train_dir)
    data_dir = os.path.join(megatron_dir, "ProcessedData")
    
    results_top_dir = os.path.join(megatron_dir, "Results")
    results_dir = os.path.join(results_top_dir, user_store_dir) # run-specific directory
    ckp_dir = os.path.join(results_dir, "checkpoints")

    os.makedirs(results_top_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckp_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO('yolov8n.pt')
    model.to(device)

    data_yaml_path = os.path.join(data_dir, 'DeepSight-2d-Mammogram', 'data.yaml')
    imgsz = 512

    # === Train and evaluate in manual loop ===
    print(f"\n Begin training YOLO for {num_epochs} epochs.")

    model.train(
        data=data_yaml_path,
        epochs=num_epochs,
        imgsz=imgsz,
        batch=100,
        device=device,
        optimizer='Adam',
        lr0=0.001,
        weight_decay=0.0005,
        patience=0,
        workers=2,
        task='detect',
        save=True,
        project="train", name="train_res"
    )

    # === Evaluate on val set ===
    metrics = model.val(
        data=data_yaml_path,
        imgsz=imgsz,
        device=device,
        batch=10,
        workers=2,
        verbose=False,
        save=True, project="train", name="val_res" # saved in same dir as above
    )

    map50 = metrics.box.map50
    map = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    print(f"mAP50={map50:.4f} | mAP50-95={map:.4f} | P={precision:.4f} | R={recall:.4f} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, help='Path to training results folder')
    args = parser.parse_args()

    # Then use args.data_folder in your code
    print(f"Using data folder: {args.data_folder}")

    main(num_epochs=50, user_store_dir=args.data_folder)