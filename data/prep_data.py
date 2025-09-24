import os
import random
import shutil
import argparse
import yaml
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import json
import cv2
from process_advanced_mri import process_all_patients

### =============== HELPER FUNCTIONS ===============

def create_destination_dirs(base_dir, has_val=False):
    if has_val:
        for subfolder in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(base_dir, subfolder), exist_ok=True)
    else:
        for subfolder in ["images/train", "labels/train"]:
            os.makedirs(os.path.join(base_dir, subfolder), exist_ok=True)

import os
from pathlib import Path

def audit_missing_pairs(images_folder, labels_folder, split_name):
    print(f"\n\tAuditing {split_name} set...")
    missing_labels = []
    missing_images = []

    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)

    images = set(f.stem for f in images_folder.glob("*") if f.suffix in [".jpg", ".png"])
    labels = set(f.stem for f in labels_folder.glob("*.txt"))

    for img_name in images:
        if img_name not in labels:
            missing_labels.append(img_name)
            # Delete image file
            for ext in [".jpg", ".png"]:
                img_path = images_folder / f"{img_name}{ext}"
                if img_path.exists():
                    img_path.unlink()
                    break

    for lbl_name in labels:
        if lbl_name not in images:
            missing_images.append(lbl_name)
            # Delete label file
            lbl_path = labels_folder / f"{lbl_name}.txt"
            if lbl_path.exists():
                lbl_path.unlink()

    if not missing_labels and not missing_images:
        print(f"\tNo missing files detected in {split_name} set.")
    else:
        if missing_labels:
            print(f"\t⚠️  Removed {len(missing_labels)} images without labels: {missing_labels}")
        if missing_images:
            print(f"\t⚠️  Removed {len(missing_images)} labels without images: {missing_images}")


def convert_bbox_to_yolo(class_id, x_min, y_min, x_max, y_max, image_width, image_height):
    """
    For DeepSight_2d_Mammogram.
    Convert bbox from (x_min, y_min, x_max, y_max) to YOLO format
    """
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def get_window_value(value, default):
    """
    For DeepSight_2d_Mammogram.
    Handle WindowCenter or WindowWidth safely.
    If it's a list, return first element.
    If it's a float or int, return as is.
    """
    if isinstance(value, (list, tuple)):
        return value[0]
    elif isinstance(value, (float, int)):
        return value
    else:
        return default  # fallback if something weird

def process_split(images_src, masks_src, images_dst, labels_dst, split_name="Split", class_id_lookup=None, use_color_mask=False):
    """
    Process one split (train or val)

    Args:
        images_src: Source images directory
        masks_src: Source masks directory
        images_dst: Destination images directory
        labels_dst: Destination labels directory
        split_name: Name for tqdm progress bar
        class_id_lookup: Optional dict mapping filename to class_id
        use_color_mask: If True, treat mask as color and threshold green regions
    """
    image_files = [f for f in os.listdir(images_src) if f.endswith(".jpg") or f.endswith(".png")]

    for filename in tqdm(image_files, desc=f"        Processing {split_name} images", unit="file", miniters=100):
        name = os.path.splitext(filename)[0]

        image_path = os.path.join(images_src, filename)
        mask_path = os.path.join(masks_src, f"{name}.png")

        shutil.copy2(image_path, os.path.join(images_dst, filename))

        if class_id_lookup:
            if filename not in class_id_lookup:
                print(f"⚠️ Warning: {filename} not found in class_id_lookup. Skipping.")
                continue
            class_id = class_id_lookup[filename]
        else:
            class_id = 0  # Default if no lookup provided

        label_path = os.path.join(labels_dst, f"{name}.txt")

        # If mask does not exist, create empty label
        if not os.path.exists(mask_path):
            print(f"⚠️ Mask not found for {filename}. Creating empty label.")
            with open(label_path, 'w') as f:
                f.write("")
            continue

        # Load the mask
        if use_color_mask:
            mask = cv2.imread(mask_path)

            # Treat any nonzero pixel as tumor
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            with open(label_path, 'w') as f:
                f.write("")
            continue

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        label_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore very small boxes (optional, to clean noise)
            if w < 5 or h < 5:
                continue

            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            bbox_width = w / img_width
            bbox_height = h / img_height

            label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            label_lines.append(label_line)

        with open(label_path, 'w') as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")
            else:
                f.write("")

def create_val_split(processed_base_dir, val_ratio=0.15, seed=2):
    
    print("\nNow processing train-val split for datasets...")

    random.seed(seed)

    for dataset_dir in tqdm(os.listdir(processed_base_dir), "Processing train-val split"):
        dataset_path = os.path.join(processed_base_dir, dataset_dir)
        images_train_dir = os.path.join(dataset_path, "images/train")
        labels_train_dir = os.path.join(dataset_path, "labels/train")
        images_val_dir = os.path.join(dataset_path, "images/val")
        labels_val_dir = os.path.join(dataset_path, "labels/val")

        if os.path.exists(images_val_dir) and os.path.exists(labels_val_dir):
            continue

        os.makedirs(images_val_dir, exist_ok=True)
        os.makedirs(labels_val_dir, exist_ok=True)

        image_files = [f for f in os.listdir(images_train_dir) if f.endswith((".jpg", ".png"))]
        val_count = int(len(image_files) * val_ratio)
        val_files = random.sample(image_files, val_count)

        for img_file in val_files:
            label_file = os.path.splitext(img_file)[0] + ".txt"

            shutil.move(os.path.join(images_train_dir, img_file),
                        os.path.join(images_val_dir, img_file))
            
            shutil.move(os.path.join(labels_train_dir, label_file),
                        os.path.join(labels_val_dir, label_file))

### =============== DATASET-SPECIFIC FUNCTIONS ===============

def reorganize_split_MRI_Brain_Tumor(extracted_dir, processed_dir):
    """
    Reorganize MRI-Brain-Tumor dataset and copy into ProcessedData folder
    """
    print(f"\tStarting reorganization for dataset: {extracted_dir} --> {processed_dir}")
    create_destination_dirs(processed_dir, True)

    dst_images_train = os.path.join(processed_dir, "images", "train")
    dst_images_val = os.path.join(processed_dir, "images", "val")
    dst_labels_train = os.path.join(processed_dir, "labels", "train")
    dst_labels_val = os.path.join(processed_dir, "labels", "val")

    def reorganize_split(split):
        print(f"\n\tProcessing {split} split...")

        split_dir = os.path.join(extracted_dir, split)
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

        image_count, label_count = 0, 0

        for class_name in tqdm(classes, desc=f"        Classes in {split}"):
            class_dir = os.path.join(split_dir, class_name)

            img_src = os.path.join(class_dir, "images")
            lbl_src = os.path.join(class_dir, "labels")

            img_dst = os.path.join(processed_dir, "images", split.lower())
            lbl_dst = os.path.join(processed_dir, "labels", split.lower())

            img_files = os.listdir(img_src) if os.path.exists(img_src) else []
            lbl_files = os.listdir(lbl_src) if os.path.exists(lbl_src) else []

            for img_file in img_files:
                src_img_path = os.path.join(img_src, img_file)
                dst_img_path = os.path.join(img_dst, f"{class_name}_{img_file}")
                shutil.copy2(src_img_path, dst_img_path)
                image_count += 1

            for lbl_file in lbl_files:
                src_lbl_path = os.path.join(lbl_src, lbl_file)
                dst_lbl_path = os.path.join(lbl_dst, f"{class_name}_{lbl_file}")
                shutil.copy2(src_lbl_path, dst_lbl_path)
                label_count += 1

        print(f"\tFinished processing {split}: {image_count} images, {label_count} labels.\n")

    reorganize_split("Train")
    reorganize_split("Val")

    audit_missing_pairs(dst_images_train, dst_labels_train, "Train")
    audit_missing_pairs(dst_images_val, dst_labels_val, "Val")
    print("\n\tDataset restructuring and audit complete! Processed data is ready.")

def reorganize_split_Chest_xray(extracted_dir, processed_dir):
    """
    Reorganize Chest-xray dataset (VinBigData) to YOLOv8 structure
    """
    print(f"\t🚀 Starting reorganization for dataset: {extracted_dir} --> {processed_dir}")
    create_destination_dirs(processed_dir, False)

    images_src = os.path.join(extracted_dir, "train")
    csv_path = os.path.join(extracted_dir, "train.csv")

    images_dst = os.path.join(processed_dir, "images", "train")
    labels_dst = os.path.join(processed_dir, "labels", "train")

    df = pd.read_csv(csv_path)
    grouped = df.groupby('image_id')

    for image_id, group in tqdm(grouped, desc="        Processing Train Images"):
        img_filename = f"{image_id}.jpg"
        src_img_path = os.path.join(images_src, img_filename)
        dst_img_path = os.path.join(images_dst, img_filename)

        if not os.path.exists(src_img_path):
            print(f"⚠️ Warning: Image {img_filename} not found, skipping.")
            continue

        shutil.copy2(src_img_path, dst_img_path)

        # Prepare YOLO label
        label_lines = []
        for _, row in group.iterrows():
            class_id = row['class_id']
            x_min = row['x_min']
            y_min = row['y_min']
            x_max = row['x_max']
            y_max = row['y_max']
            image_width = row['raw_width'] * row['scale_x'] # scale width
            image_height = row['raw_height'] * row['scale_y']  # scaled height

            if pd.isna(x_min) or pd.isna(y_min) or pd.isna(x_max) or pd.isna(y_max):
                # skip if no bbox values
                continue

            # Normalize coordinates
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Normalize to [0, 1] range
            label_line = f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            label_lines.append(label_line)

        # Write YOLO .txt file
        label_filename = os.path.join(labels_dst, f"{image_id}.txt")
        with open(label_filename, 'w') as f:
            if label_lines:
                f.write("\n".join(label_lines))
            else:
                f.write("")  # Write an empty file if there are no labels

    audit_missing_pairs(images_dst, labels_dst, "Train")
    print("\n\tDataset restructuring and audit complete! Processed data is ready.")

def reorganize_split_DeepSight_2d_Mammogram(extracted_dir, processed_dir):
    """
    Reorganize DeepSight 2D Mammogram Dataset to YOLOv8 structure
    """
    print(f"\t🚀 Starting reorganization for dataset: {extracted_dir} --> {processed_dir}")
    create_destination_dirs(processed_dir, False)

    images_dst = os.path.join(processed_dir, "images", "train")
    labels_dst = os.path.join(processed_dir, "labels", "train")

    images_src = os.path.join(extracted_dir, "2d_resized_256/images")   # .npz images
    metadata_src = os.path.join(extracted_dir, "2d_resized_256/metadata") # metadata JSONs
    
    label_to_class_id = {
        "NonCancer": 0,
        "IndexCancer": 1,
        "PreIndexCancer": 2,
        "Unknown": 3
    }

    image_files = [f for f in os.listdir(images_src) if f.endswith(".npz")]

    for filename in tqdm(image_files, desc="        Processing DeepSight 2D Mammogram", unit="file", miniters=100):
        image_path = os.path.join(images_src, filename)
        metadata_filename = os.path.splitext(filename)[0] + ".json"
        metadata_path = os.path.join(metadata_src, metadata_filename)

        if not os.path.exists(metadata_path):
            print(f"⚠️ Metadata not found for {filename}. Skipping.")
            continue

        with open(metadata_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        label = data.get("label", "Unknown")
        if label == "Unknown":
            continue  # Skip Unknown class

        class_id = label_to_class_id[label]
        coords = data.get("coords", None)
        if coords is None:
            print(f"⚠️ No coords found for {filename}. Skipping.")
            continue

        npz_data = np.load(image_path, allow_pickle=True)
        img_array = npz_data['data']

        # Ensure the image is uint8
        if img_array.dtype != np.uint8:
            img_array = (255 * (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))).astype(np.uint8)

        # Save the image as PNG
        name = os.path.splitext(filename)[0]
        output_image_path = os.path.join(images_dst, f"{name}.png")
        img = Image.fromarray(img_array)
        img.save(output_image_path)

        # Prepare label .txt
        x_min, y_min, x_max, y_max = coords
        image_width = img_array.shape[1]
        image_height = img_array.shape[0]
        label_line = convert_bbox_to_yolo(class_id, x_min, y_min, x_max, y_max, image_width, image_height)

        label_output_path = os.path.join(labels_dst, f"{name}.txt")
        with open(label_output_path, 'w') as f:
            f.write(label_line+"\n")  

    audit_missing_pairs(images_dst, labels_dst, "Train")
    print("\nDataset restructuring and audit complete! Processed data is ready.")

def reorganize_split_CBIS_DDSM_Mammogram(extracted_dir, processed_dir):
    """
    Reorganize CBIS-DDSM Mammogram Dataset to YOLOv8 structure
    """
    print(f"\tStarting reorganization for dataset: {extracted_dir} --> {processed_dir}")
    create_destination_dirs(processed_dir, True)

    images_dst_train = os.path.join(processed_dir, "images", "train")
    labels_dst_train = os.path.join(processed_dir, "labels", "train")
    images_dst_val = os.path.join(processed_dir, "images", "val")
    labels_dst_val = os.path.join(processed_dir, "labels", "val")

    train_images_src = os.path.join(extracted_dir, "CBIS-DDSM-Patches/train", "images")   # train images
    train_masks_src = os.path.join(extracted_dir, "CBIS-DDSM-Patches/train", "masks")     # train masks
    val_images_src = os.path.join(extracted_dir, "CBIS-DDSM-Patches/test", "images")      # val images
    val_masks_src = os.path.join(extracted_dir, "CBIS-DDSM-Patches/test", "masks")        # val masks

    process_split(train_images_src, train_masks_src, images_dst_train, labels_dst_train, split_name="Train")
    process_split(val_images_src, val_masks_src, images_dst_val, labels_dst_val, split_name="Validation")

    audit_missing_pairs(images_dst_train, labels_dst_train, "Train")
    audit_missing_pairs(images_dst_val, labels_dst_val, "Validation")
    print("\n\tDataset restructuring and audit complete! Processed data is ready.")

def reorganize_split_Breast_Ultrasound(extracted_dir, processed_dir):
    """
    Reorganize Breast Ultrasound Dataset to YOLOv8 structure
    """
    print(f"\tStarting reorganization for dataset: {extracted_dir} --> {processed_dir}")
    create_destination_dirs(processed_dir, False)

    images_dst_train = os.path.join(processed_dir, "images", "train")
    labels_dst_train = os.path.join(processed_dir, "labels", "train")

    bus_uclm = "BUS-UCLM Breast ultrasound lesion segmentation dataset"
    train_images_src = os.path.join(extracted_dir, bus_uclm, "BUS-UCLM/images")   # train images
    train_masks_src = os.path.join(extracted_dir, bus_uclm, "BUS-UCLM/masks")     # train masks
    info_csv_path = os.path.join(extracted_dir, bus_uclm, "BUS-UCLM/INFO.csv")   
    df_info = pd.read_csv(info_csv_path, sep=";")

    label_mapping = {"Normal": -1, "Benign": 0, "Malignant": 1}
    filename_to_classid = {row['Image']: label_mapping[row['Label']] for _, row in df_info.iterrows()}

    process_split(
        train_images_src,
        train_masks_src,
        images_dst_train,
        labels_dst_train,
        split_name="Train",
        class_id_lookup=filename_to_classid,
        use_color_mask=True
    )

    audit_missing_pairs(images_dst_train, labels_dst_train, "Train")
    print("\n\tDataset restructuring and audit complete! Processed data is ready.")

def reorganize_split_RSNA_pneumonia(extracted_dir, processed_dir):
    """
    Reorganize RSNA Pneumonia Dataset to YOLOv8 structure
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    print(f"\tStarting reorganization for dataset: {extracted_dir} --> {processed_dir}")
    create_destination_dirs(processed_dir, False)

    images_dst_train = os.path.join(processed_dir, "images", "train")
    labels_dst_train = os.path.join(processed_dir, "labels", "train")

    os.makedirs(images_dst_train, exist_ok=True)
    os.makedirs(labels_dst_train, exist_ok=True)

    train_images_src = os.path.join(extracted_dir, "stage_2_train_images")
    csv_path = os.path.join(extracted_dir, "stage_2_train_labels.csv")   

    df = pd.read_csv(csv_path)
    grouped = df.groupby('patientId')

    for patientId, rows in tqdm(grouped, desc="        Processing RSNA images", unit="file", miniters=200):
        image_path = os.path.join(train_images_src, f"{patientId}.dcm")
        image_dst_path = os.path.join(images_dst_train, f"{patientId}.png")
        label_path = os.path.join(labels_dst_train, f"{patientId}.txt")

        if not os.path.exists(image_path):
            print(f"⚠️ DICOM image not found for {patientId}. Skipping.")
            continue

        # Load and convert DICOM image
        dicom = pydicom.dcmread(image_path)
        data = dicom.pixel_array.astype(np.float32)

        voi_lut = True
        fix_monochrome = True

       # Only apply VOI LUT if DICOM has relevant fields
        if voi_lut and hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
            try:
                data = apply_voi_lut(data, dicom)
            except Exception as e:
                print(f"Warning: Failed to apply VOI LUT for {patientId}: {e}")
        # else:
        #     # print(f"No VOI LUT metadata for {patientId}. Skipping VOI LUT.")

        # Handle MONOCHROME1 inversion
        if fix_monochrome and hasattr(dicom, 'PhotometricInterpretation') and dicom.PhotometricInterpretation == "MONOCHROME1":
            print("Fixing monochrome\n")
            data = np.amax(data) - data

        # Normalize to 0–255 uint8
        if np.min(data) != np.max(data):  # Avoid divide-by-zero
            data = data - np.min(data)
            data = data / np.max(data)
            data = (data * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)

        img_array = data

        cv2.imwrite(image_dst_path, img_array)

        height, width = img_array.shape

        # Check if any pneumonia is present
        if rows['Target'].sum() == 0:
            # No pneumonia: create empty label
            with open(label_path, 'w') as f:
                f.write("")
            continue

        # Otherwise, write YOLO bbox(es)
        label_lines = []
        for _, row in rows.iterrows():
            if row['Target'] == 1:
                x = row['x']
                y = row['y']
                w = row['width']
                h = row['height']

                # Normalize bbox to YOLOv8 format
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height

                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(label_path, 'w') as f:
            f.write("\n".join(label_lines) + "\n")

    audit_missing_pairs(images_dst_train, labels_dst_train, "Train")
    print("\n\tDataset restructuring and audit complete! Processed data is ready.")

def reorganize_split_Advanced_MRI_Breast_Lesion(extracted_dir, processed_dir):
    print(f"\tStarting reorganization for dataset: {extracted_dir} --> {processed_dir}")
    images_dst_train = os.path.join(processed_dir, "images", "train")
    labels_dst_train = os.path.join(processed_dir, "labels", "train")

    os.makedirs(images_dst_train, exist_ok=True)
    os.makedirs(labels_dst_train, exist_ok=True)

    process_all_patients(extracted_dir, processed_dir)

    audit_missing_pairs(images_dst_train, labels_dst_train, "Train")
    print("\n\tDataset restructuring and audit complete! Processed data is ready.")

### Add your own processing function in the form: def reorganize_split_<filename_with_underscores>():

### =============== MAIN LOGIC ===============

def load_datasets_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['datasets']

def call_process_method_for_dataset(filename_without_ext, extracted_base_dir, processed_base_dir):
    func_name = f"reorganize_split_{filename_without_ext.replace('-', '_')}"
    if func_name in globals():
        print(f"\n\tCalling function: {func_name}")
        globals()[func_name](extracted_base_dir, processed_base_dir)
    else:
        print(f"\t⚠️ Warning: No processing function found for '{func_name}'. Skipping dataset.")

def main(force=False):
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    megatron_dir = os.path.dirname(utils_dir)
    extracted_base_dir = os.path.join(megatron_dir, "ExtractedData")
    processed_base_dir = os.path.join(megatron_dir, "ProcessedData")
    config_path = os.path.join(utils_dir, 'config', 'datasets_config.yaml')

    os.makedirs(processed_base_dir, exist_ok=True)
    datasets = load_datasets_config(config_path)

    for dataset in datasets:
        filename_no_ext = os.path.splitext(dataset['filename'])[0]
        extracted_dir = os.path.join(extracted_base_dir, filename_no_ext)
        processed_dir = os.path.join(processed_base_dir, filename_no_ext)

        if os.path.exists(processed_dir) and any(os.scandir(processed_dir)):
            if force:
                print(f"\n⚠️  Reprocessing {filename_no_ext}. Removing existing: {processed_dir}")
                shutil.rmtree(processed_dir)
            else:
                print(f"\n✅ Already processed: {processed_dir}. Skipping...")
                continue

        print(f"\n📦 Processing dataset: {filename_no_ext}")
        if os.path.exists(extracted_dir):
            call_process_method_for_dataset(filename_no_ext, extracted_dir, processed_dir)
        else:
            print(f"❌ Skipping {filename_no_ext}: Extracted directory does not exist.\n")
    
    # For datasets that didn't already come with validation set, create train-val split
    create_val_split(processed_base_dir=processed_base_dir, val_ratio=0.15, seed=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force reprocessing of your datasets by deleting existing processed folders")
    args = parser.parse_args()

    main(force=args.force)
