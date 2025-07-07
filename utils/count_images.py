import os

def count_labels_in_processed_data(processed_data_dir):
    total_images = 0
    total_labeled = 0
    total_unlabeled = 0

    print(f"🔍 Scanning datasets in: {processed_data_dir}\n")

    for dataset_name in os.listdir(processed_data_dir):
        dataset_path = os.path.join(processed_data_dir, dataset_name)
        labels_train_path = os.path.join(dataset_path, "labels", "train")

        if not os.path.isdir(labels_train_path):
            print(f"⚠️ Skipping {dataset_name}: 'labels/val/' folder not found.")
            continue
       
        num_labeled = 0
        num_unlabeled = 0

        for file in os.listdir(labels_train_path):
            file_path = os.path.join(labels_train_path, file)
            if os.path.getsize(file_path) > 0:
                num_labeled += 1
            else:
                num_unlabeled += 1

        count = num_labeled + num_unlabeled
        total_images += count
        total_labeled += num_labeled
        total_unlabeled += num_unlabeled

        print(f"{dataset_name}: {count} images ({num_labeled} labeled, {num_unlabeled} unlabeled)")

    print(f"\nTotal images: {total_images} ({total_labeled} labeled, {total_unlabeled} unlabeled)")

import os

def count_class_0_txt_files(directory):
    """
    Count the number of YOLO-format .txt files in a directory that contain class ID 0.

    Args:
        directory (str): Path to the directory containing YOLO .txt annotation files.

    Returns:
        int: Number of .txt files with at least one line starting with class ID 0.
    """
    count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and stripped.split()[0] == '0':
                        count += 1
                        break  # Only count the file once
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    print(f'Number of labels with class ID 0: {count}')
    return count

def delete_files_with_only_class_0(data_dir, image_exts={'.jpg', '.jpeg', '.png'}):
    """
    Deletes images and labels in both train/ and val/ that contain only class ID 0.

    Args:
        data_dir (str): Base path containing 'images/train', 'images/val', etc.
        image_exts (set): Set of allowed image file extensions.
    """
    for split in ['train', 'val']:
        label_dir = os.path.join(data_dir, 'labels', split)
        image_dir = os.path.join(data_dir, 'images', split)

        deleted = 0
        for fname in os.listdir(label_dir):
            if not fname.endswith('.txt'):
                continue

            label_path = os.path.join(label_dir, fname)
            try:
                with open(label_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    class_ids = [line.split()[0] for line in lines]
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
                continue

            if lines and all(cid == '0' for cid in class_ids):
                os.remove(label_path)

                base_name = os.path.splitext(fname)[0]
                deleted_image = False
                for ext in image_exts:
                    image_path = os.path.join(image_dir, base_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        deleted_image = True
                        break

                deleted += 1
                #print(f"Deleted from {split}: {label_path}" + (f" and image" if deleted_image else ""))
        
        print(f"[{split}] Total deletions: {deleted}")


if __name__ == "__main__":
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    megatron_dir = os.path.dirname(utils_dir)
    processed_base_dir = os.path.join(megatron_dir, "ProcessedData")
    deepsight_dir = os.path.join(processed_base_dir, "DeepSight-2d-Mammogram")
    deepsight_val_dir = os.path.join(processed_base_dir, "DeepSight-2d-Mammogram/labels/val")

    count_labels_in_processed_data(processed_base_dir)
    # delete_files_with_only_class_0(deepsight_dir)
    # count_class_0_txt_files(deepsight_val_dir)
