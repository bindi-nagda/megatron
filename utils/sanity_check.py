import os
import random
import cv2
import argparse
from glob import glob

def sanity_check(processed_dir, sanity_dir, dataset_name, num_samples=20):
    """
       Perform a sanity check on the processed dataset by visualizing randomly 
       selected images and their bounding boxes.
    """
    print(f"Starting sanity check for {dataset_name}")

    images_dir = os.path.join(processed_dir, dataset_name, "images", "train")
    labels_dir = os.path.join(processed_dir, dataset_name, "labels", "train")
    output_dir = os.path.join(sanity_dir, dataset_name)
    
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]

    if len(image_files) == 0:
        print("No images found!")
        return

    # Randomly select images
    selected_images = random.sample(image_files, min(num_samples, len(image_files)))

    for img_file in selected_images:
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        height, width = img.shape[:2]

        # Read labels
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found.")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Warning: Label file {label_path} is empty. Skipping drawing.")
            continue

        # Draw bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping invalid label line: {line}")
                continue  # skip invalid lines

            class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

            if class_id == 1:
                print(f"\nclass_id = {class_id}")

            # Convert normalized to absolute coordinates
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)
            print(f" bbox=({x_min}, {y_min}), ({x_max},{y_max})")

            # Draw rectangle and class id
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, str(int(class_id)), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save image
        save_path = os.path.join(output_dir, img_file)
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")

    print(f"🎯 Sanity check complete! Check {output_dir}")


def count_files_with_class_0(labels_dir):
    txt_files = glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)
    count = 0

    for file_path in txt_files:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip().startswith("0 "):  # class ID 0
                    count += 1
                    break  # count each file only once

    print(f"Found {count} label files containing class ID 0 (unknown) in {labels_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sanity check on a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., DeepSight-2d-Mammogram)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples for sanity check",
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    utils_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(utils_dir)
    processed_dir = os.path.join(base_dir, "ProcessedData")
    sanity_dir = os.path.join(base_dir, "SanityCheck")

    sanity_check(processed_dir, sanity_dir, dataset_name, num_samples=args.num_samples)
