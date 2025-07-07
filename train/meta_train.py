import tqdm
from ultralytics.utils import LOGGER
LOGGER.setLevel(60)

# disable tqdm globally after Ultralytics has loaded
tqdm.tqdm = lambda *args, **kwargs: iter(args[0])

import warnings
warnings.filterwarnings("ignore")

import argparse
import os, pickle, gc
import yaml, json
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # this still works
from ultralytics.models.yolo.model import YOLO
from torch.utils.data import DataLoader
import tempfile
from collections import defaultdict
import tempfile
from pathlib import Path

from sampling import create_episode_configs
from head_manager import HeadManager


def safe_clone_model(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "temp_model.pt")
        model.save(model_path)
        cloned_model = YOLO(model_path)
    return cloned_model.to(model.device)

def reptile_update(base_model, episode_model, meta_lr):
    base_params = dict(base_model.model.named_parameters())
    episode_params = dict(episode_model.model.named_parameters())

    for name, base_param in base_params.items():
        if name in episode_params and base_param.shape == episode_params[name].shape:
            episode_param = episode_params[name].data.to(base_param.device)
            base_param.data += meta_lr * (episode_param - base_param.data)


def write_episode_data_yaml(support_imgs, query_imgs, yaml_path):
    if not support_imgs and not query_imgs:
        raise ValueError("Both support and query image lists are empty.")
    
    support_txt = os.path.join(os.path.dirname(yaml_path), "support.txt")
    query_txt = os.path.join(os.path.dirname(yaml_path), "query.txt")

    with open(support_txt, 'w') as f:
        for path in support_imgs:
            f.write(f"{path}\n")

    with open(query_txt, 'w') as f:
        for path in query_imgs:
            f.write(f"{path}\n")

    # Use either support or query image to get the task path
    example_img = support_imgs[0] if support_imgs else query_imgs[0]
    task_dir = example_img.split("/images/")[0]
    dataset_yaml_path = os.path.join(task_dir, "data.yaml")

    # Load nc and names from that task's data.yaml
    with open(dataset_yaml_path, 'r') as f:
        base_yaml = yaml.safe_load(f)
    nc = base_yaml.get("nc", 1)
    names = base_yaml.get("names", ["tumor"])

    with open(yaml_path, 'w') as f:
        f.write(f"train: {support_txt}\n")
        f.write(f"val: {query_txt}\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

def load_or_create_episodes(episode_dir, data_dir, yaml_config_dir, label='train', support_size=10, query_size=10, seed=None, force_regen=False):
    os.makedirs(episode_dir, exist_ok=True)
    path = os.path.join(episode_dir,f"{label}_episodes.pkl")

    if not force_regen and os.path.exists(path):
        print("Loaded cached episode configs.")
        with open(path, "rb") as f:
            eps = pickle.load(f)
    else:
        #print("Generating new episode configs...")
        eps = create_episode_configs(data_dir, yaml_config_dir, label=label, support_size=support_size, query_size=query_size, with_replacement=False, seed=seed)
        with open(path, "wb") as f:
            pickle.dump(eps, f)

    return eps

def meta_episode_collate(batch):
        return batch  # list of episode dicts

def main(num_epochs=10, meta_lr=0.05, user_store_dir='/tmp/'):
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
    yaml_config_dir = os.path.join(results_dir, "config")
    ckp_dir = os.path.join(results_dir, "checkpoints")
    head_dir = os.path.join(results_dir, "heads")
    plot_dir = os.path.join(results_dir, "plots")
    metrics_path = os.path.join(results_dir, "epoch_val_metrics.json")

    os.makedirs(results_top_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(yaml_config_dir, exist_ok=True)
    os.makedirs(ckp_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)

    # TODO: add a line to copy utils/dataset_config.yaml to yaml_config_dir
    # and read the tasks from the this file.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = YOLO('yolov8n.pt'). 
    base_model.to(device)

    best_avg_map50 = 0.0
    head_manager = HeadManager(head_dir, save_every=20)
    task_list= ['CBIS-DDSM-Mammogram', 'MRI-Brain-Tumor', 'DeepSight-2d-Mammogram',
                'Breast-Ultrasound', 'Advanced-MRI-Breast-Lesion', 
                'RSNA-pneumonia', 'Chest-xray'] 
                
    print(f"\nBegin Meta-Training: total epochs={num_epochs}, meta_lr={meta_lr}, tasks={task_list}")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1} ===")

        for task in task_list:
            print(f"    Now training on task [{task}]")
            yaml_path = os.path.join(data_dir, task, 'data.yaml')
            with open(yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
            num_classes = data_yaml['nc']   

                  
            head_manager.get_or_create_head(base_model, task, num_classes)
            episode_model = safe_clone_model(base_model)

            with tempfile.TemporaryDirectory() as tmpdir:
                support_images, query_images = sample_support_and_query_images(
                    image_dir=f"{data_dir}/{task}/images/train", 
                    support_size=800, 
                    query_size=200, 
                    split_ratio=0.98,
                    seed=epoch)

                joint_yaml = create_joint_episode_yaml(
                            base_yaml_path=yaml_path,
                            output_yaml_path=os.path.join(tmpdir, f"{task}_joint.yaml"),
                            support_images=support_images,
                            query_images=query_images,
                            split="train")

                episode_model.train(
                    data=str(joint_yaml),  
                    epochs=30, batch=100, patience=0, optimizer='Adam',
                    lr0=0.001,
                    momentum=0.9,
                    weight_decay=0.0005,
                    warmup_epochs=3, 
                    imgsz=512,
                    device=device.index if device.type == 'cuda' else None,
                    verbose=False, save=False, plots=False,
                    workers=2,
                    task='detect', project="train", name="train_res"
                )

                head_manager.update_after_episode(episode_model, task)
        
            reptile_update(base_model, episode_model, meta_lr)

            del support_images, query_images
            del episode_model
            torch.cuda.empty_cache()
            gc.collect()

        task_metrics = defaultdict(list)
        # Evaluate on a val set per task to track meta-level generalization across epochs 
        for task in task_list:
            yaml_path = os.path.join(data_dir, task, 'data.yaml')
            with open(yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
            num_classes = data_yaml['nc']   

            head_manager.get_or_create_head(base_model, task, num_classes)
            val_model = safe_clone_model(base_model)
            val_model.model.names = data_yaml['names']
            
            with tempfile.TemporaryDirectory() as tmpdir:
                support_images, atemp_images = sample_support_and_query_images(
                    image_dir=f"{data_dir}/{task}/images/train", 
                    support_size=800, 
                    query_size=200, 
                    split_ratio=0.98,
                    seed=epoch)

                train_yaml = create_joint_episode_yaml(
                    base_yaml_path=yaml_path,
                    output_yaml_path=os.path.join(tmpdir, f"{task}_meta_tr.yaml"),
                    support_images=support_images,
                    query_images=atemp_images,
                    split="train")
                
                # Without this step, we're unfairly evaluating Reptile on a task 
                # it wasn’t trained for (i.e., zero-shot generalization).
                val_model.train(data=str(train_yaml), epochs=5, batch=100, imgsz=512, 
                                device=device.index if device.type == 'cuda' else None,
                                warmup_epochs=3,
                                verbose=False,
                                save=False,
                                plots=False, 
                                project="meta", name="train_res",) # newly added
            
            del support_images, atemp_images

            with tempfile.TemporaryDirectory() as tmpdir:
                val_images, btemp_images = sample_support_and_query_images(
                    image_dir=f"{data_dir}/{task}/images/val",
                    support_size=500,
                    query_size=1,
                    split_ratio=0.95,  # all images to to val_images
                    seed=epoch
                )

                val_yaml = create_joint_episode_yaml(
                    base_yaml_path=yaml_path,
                    output_yaml_path=os.path.join(tmpdir, f"{task}_meta_val.yaml"),
                    support_images=btemp_images,
                    query_images=val_images,
                    split="val")
           
                print(f"    Now evaluating on val set for [{task}]")
                try:
                    metrics = val_model.val(
                        data=val_yaml,
                        imgsz=512,
                        batch=64,
                        device=device.index if device.type == 'cuda' else None,
                        verbose=False,
                        workers=2,
                        task='detect',
                        project="meta", name="val_res",
                        save=False, plots=False
                    )
                    print(f"    Finished evaluating on val set for [{task}]")
                except RuntimeError as e:
                    print(f"    Skipping episode validation for {task} due to empty stats: {e}")
                    metrics = None

                del val_images, btemp_images
                del val_model
                torch.cuda.empty_cache()
                gc.collect()

                if metrics is not None:
                    task_metrics[task].append({
                        'map50': metrics.box.map50,
                        'map': metrics.box.map,
                        'precision': metrics.box.mp,
                        'recall': metrics.box.mr
                    })

        # Load existing metrics if available
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                all_epoch_metrics = json.load(f)
        else:
            all_epoch_metrics = {}

        # --- Compute and display per-task aggregated summary ---
        print(f"\n    Meta-Val Aggregate Results:")
        epoch_record = {}
        for task, episodes in task_metrics.items():
            n = len(episodes)
            avg_map50 = sum(e['map50'] for e in episodes) / n
            avg_map = sum(e['map'] for e in episodes) / n
            avg_p = sum(e['precision'] for e in episodes) / n
            avg_r = sum(e['recall'] for e in episodes) / n
            avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r + 1e-6)

            # Print summary
            print(f"    [{task}] n={n} | mAP50={avg_map50:.4f} | mAP50-95={avg_map:.4f} | P={avg_p:.4f} | R={avg_r:.4f} | F1={avg_f1:.4f}")

            # Save to epoch record
            epoch_record[task] = {
                "mAP50": avg_map50,
                "mAP": avg_map,
                "precision": avg_p,
                "recall": avg_r,
                "f1": avg_f1
            }
        
        # Save using current epoch (1-based)
        all_epoch_metrics[str(epoch + 1)] = epoch_record

        with open(metrics_path, "w") as f:
            json.dump(all_epoch_metrics, f, indent=2)
        
        # if (epoch + 1) % 5 == 0:
        #     plot_val_metrics_over_epochs(metrics_path, plot_dir, smooth_window=1)

        if (epoch + 1) % 20 == 0:
            save_path = f"{ckp_dir}/model_epoch{epoch+1}.pt"
            base_model.save(save_path)
        
        # Compute average mAP50 across all tasks
        avg_map50_across_tasks = sum(task['mAP50'] for task in epoch_record.values()) / len(epoch_record)
        print(f"    Avg mAP50 across tasks: {avg_map50_across_tasks:.4f}")

        # Save best model based on avg mAP50
        if avg_map50_across_tasks > best_avg_map50:
            best_avg_map50 = avg_map50_across_tasks
            best_path = f"{ckp_dir}/best_model_epoch{epoch+1}.pt"
            base_model.save(best_path)
            print(f"    Best model saved at epoch {epoch + 1} with mAP50={best_avg_map50:.4f}")

        print(f"\n    Saved meta-val metrics, plots and models to {results_dir}")
    
    print("\nMeta-training complete.")

def plot_val_metrics_over_epochs(metrics_path, output_dir, smooth_window=3):
    """
    Plots validation metrics vs # of epochs per task

    Args:
        metrics_path (str): Path to JSON file storing per-epoch metrics.
        output_dir (str): Directory where plots will be saved.
        smooth_window (int): Window size for moving average smoothing.
    """
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, "r") as f:
        all_epoch_metrics = json.load(f)

    if not all_epoch_metrics:
        print("No metrics to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract task names and metric keys from the first epoch
    sample_epoch = next(iter(all_epoch_metrics.values()))
    tasks = list(sample_epoch.keys())
    metric_keys = list(sample_epoch[tasks[0]].keys())

    for metric in metric_keys:
        plt.figure()
        for task in tasks:
            x = sorted(int(e) for e in all_epoch_metrics.keys())
            y = np.array([all_epoch_metrics[str(e)][task][metric] for e in x], dtype=np.float32)

            def moving_avg(arr, k):
                return np.convolve(arr, np.ones(k) / k, mode='valid')

            def moving_std(arr, k):
                return np.array([np.std(arr[i:i+k]) for i in range(len(arr) - k + 1)], dtype=np.float32)

            if len(y) >= smooth_window:
                y_smooth = moving_avg(y, smooth_window)
                y_std = moving_std(y, smooth_window)
                x_smooth = x[smooth_window - 1:]

                y_smooth = np.asarray(y_smooth, dtype=np.float32)
                y_std = np.asarray(y_std, dtype=np.float32)

                plt.plot(x_smooth, y_smooth, label=task)
                plt.fill_between(x_smooth, y_smooth - y_std, y_smooth + y_std, alpha=0.2)
            else:
                plt.plot(x, y, label=task)

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} over epochs on the Val Set")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric}_over_epochs.png")
        plt.savefig(plot_path)
        plt.close()

def sample_support_and_query_images(image_dir, support_size=1000, query_size=200, split_ratio=0.8, seed=5):
    image_dir = Path(image_dir)
    label_dir = image_dir.parent.parent / "labels" / image_dir.name

    all_images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    total_available = len(all_images)

    random.seed(seed)
    random.shuffle(all_images)

    # Fallback to split if not enough images
    if total_available < support_size + query_size:
        support_end = int(split_ratio * total_available)
        query_end = total_available
        support_images = all_images[:support_end]
        query_images = all_images[support_end:query_end]
        return support_images, query_images

    # Separate labeled and unlabeled images
    labeled_images = [img for img in all_images if (label_dir / (img.stem + ".txt")).stat().st_size > 0]
    unlabeled_images = [img for img in all_images if (label_dir / (img.stem + ".txt")).stat().st_size == 0]

    # Support set: 95% labeled, 5% unlabeled
    support_labeled_count = int(0.95 * support_size)
    support_unlabeled_count = support_size - support_labeled_count

    if len(labeled_images) < support_labeled_count:
        raise ValueError("Not enough labeled images for the support set.")

    support_labeled = random.sample(labeled_images, support_labeled_count)
    support_unlabeled = random.sample(unlabeled_images, min(support_unlabeled_count, len(unlabeled_images)))
    support_images = support_labeled + support_unlabeled
    random.shuffle(support_images)

    # Query set: 95% labeled, 5% unlabeled (avoid overlap with support_labeled)
    remaining_labeled = list(set(labeled_images) - set(support_labeled))
    if len(remaining_labeled) < int(0.95 * query_size):
        raise ValueError("Not enough labeled images for the query set.")

    query_labeled = random.sample(remaining_labeled, int(0.95 * query_size))
    query_unlabeled = random.sample(unlabeled_images, min(int(0.05 * query_size), len(unlabeled_images)))
    query_images = query_labeled + query_unlabeled
    random.shuffle(query_images)

    return support_images, query_images

def create_joint_episode_yaml(base_yaml_path, output_yaml_path, support_images, query_images, split='train'):
    with open(base_yaml_path, 'r') as f:
        base_yaml = yaml.safe_load(f)

    root_dir = Path(output_yaml_path).parent

    # Create dirs for support and query episodes
    support_img_dir = root_dir / "images" / "train"
    query_img_dir   = root_dir / "images" / "val"
    support_lbl_dir = root_dir / "labels" / "train"
    query_lbl_dir   = root_dir / "labels" / "val"

    support_img_dir.mkdir(parents=True, exist_ok=True)
    query_img_dir.mkdir(parents=True, exist_ok=True)
    support_lbl_dir.mkdir(parents=True, exist_ok=True)
    query_lbl_dir.mkdir(parents=True, exist_ok=True)

    base_yaml['train'] = os.path.realpath(base_yaml['train'])
    base_yaml['val'] = os.path.realpath(base_yaml['val'])

    # Determine original label path
    # train_img_root = Path(base_yaml_path).parent / "images" / split
    label_root = Path(base_yaml_path).parent / "labels" / split

    def link_images_and_labels(images, img_dir, lbl_dir):
        for img_path in images:
            img_path = Path(img_path)
            lbl_path = label_root / (img_path.stem + ".txt")

            dst_img = img_dir / img_path.name
            dst_lbl = lbl_dir / lbl_path.name

            if not dst_img.exists():
                dst_img.symlink_to(img_path.resolve())
            if lbl_path.exists() and not dst_lbl.exists():
                dst_lbl.symlink_to(lbl_path.resolve())

    link_images_and_labels(support_images, support_img_dir, support_lbl_dir)
    link_images_and_labels(query_images, query_img_dir, query_lbl_dir)

    # Update YAML with new paths
    base_yaml['train'] = str(support_img_dir.resolve())
    base_yaml['val'] = str(query_img_dir.resolve())

    with open(output_yaml_path, "w") as f:
        yaml.dump(base_yaml, f)

    return output_yaml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, help='Path to training results folder')
    args = parser.parse_args()

    # Then use args.data_folder in your code
    print(f"Using data folder: {args.data_folder}")

    main(num_epochs=40, user_store_dir=args.data_folder)
