import os
import cv2
import random
import numpy as np
import pydicom
from glob import glob
from highdicom_reader import SeriesDataset
from tqdm import tqdm

def has_intensity_overlap(image_slice, mask_slice, threshold=150, min_overlap=20):
    """
    Optional function. Check if the mask overlaps with high-intensity regions (likely tumor).
    """
    high_intensity = image_slice > threshold
    overlap = np.logical_and(high_intensity, mask_slice > 0)
    return np.count_nonzero(overlap) > min_overlap

def extract_aligned_mask(seg_dicom, series: list[pydicom.Dataset]) -> np.ndarray:
    """
    Extracts a 3D binary mask from a 2D or 3D DICOM SEG file,
    aligned with the given image series (*.dcm images) by SOPInstanceUID.
    """
    seg_array = seg_dicom.pixel_array
    ref_seq = seg_dicom.PerFrameFunctionalGroupsSequence
    uid_to_index = {d.SOPInstanceUID: i for i, d in enumerate(series)}
    num_slices = len(series)

    if seg_array.ndim != 3 or seg_array.shape[0] != len(ref_seq):
        raise ValueError(f"Unsupported SEG shape: {seg_array.shape}")

    height, width = seg_array.shape[1:] # seg_array shape: (frames, H, W)
    aligned_mask = np.zeros((num_slices, height, width), dtype=np.uint8)

    for frame_idx, frame in enumerate(ref_seq):
        try:
            if hasattr(frame, "DerivationImageSequence"):
                uid = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            elif hasattr(frame, "ReferencedImageSequence"):
                uid = frame.ReferencedImageSequence[0].ReferencedSOPInstanceUID
            else:
                continue
        except Exception:
            continue

        if uid not in uid_to_index:
            continue

        slice_idx = uid_to_index[uid]
        aligned_mask[slice_idx] = seg_array[frame_idx]

    return aligned_mask

def save_yolo_labels(mask_np, slice_idx, output_root, patient_id):
    label_path = os.path.join(output_root, "labels", "train")
    os.makedirs(label_path, exist_ok=True)
    filename = os.path.join(label_path, f"{patient_id}_image_{slice_idx:03}.txt")
    mask = mask_np[slice_idx].astype(np.uint8)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(filename, "w") as f:
        if len(contours) == 0:
            return
        for cnt in contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h

            if norm_w < 0.015 or norm_h < 0.015:
                continue  # Skip too-small boxes

            f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def save_all_pngs(image_np, mask_np, slice_idx, output_root, patient_id, force=False):
    image = image_np[slice_idx]
    mask = mask_np[slice_idx]

    if not force:
        if not mask.any() or np.count_nonzero(mask) < 100:
            return False
        # if not has_intensity_overlap(image, mask):
        #     return False

    image_norm = ((image - np.min(image)) / np.ptp(image) * 255).astype(np.uint8)
   
    image_dir = os.path.join(output_root, "images", "train")
    os.makedirs(image_dir, exist_ok=True)
    img_filename = f"{patient_id}_image_{slice_idx:03}.png"

    cv2.imwrite(os.path.join(image_dir, img_filename), image_norm)

    # # Save mask and overlay image for debug purposes
    # mask_dir = os.path.join(output_root, "masks")
    # overlay_dir = os.path.join(output_root, "overlays")
    # os.makedirs(mask_dir, exist_ok=True)
    # os.makedirs(overlay_dir, exist_ok=True)
    # cv2.imwrite(os.path.join(mask_dir, img_filename.replace("image", "mask")), (mask * 255).astype(np.uint8))
   
    # # Create overlay with transparency
    # overlay = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2BGR)
    # green_mask = np.zeros_like(overlay)
    # green_mask[:, :, 1] = 255
    # alpha = 0.20
    # overlay = np.where(mask[:, :, None] > 0, (alpha * green_mask + (1 - alpha) * overlay).astype(np.uint8), overlay)
    # cv2.imwrite(os.path.join(overlay_dir, img_filename.replace("image", "overlay")), overlay)

    save_yolo_labels(mask_np, slice_idx, output_root, patient_id)
    return True

def extract_segmentation(seg_path, image_series_dir, output_root, patient_id, num_background_slices=5):
    try:
        seg_dicom = pydicom.dcmread(seg_path)
        # print("SEG pixel array shape:", seg_dicom.pixel_array.shape)

        reader = SeriesDataset.from_files(image_series_dir)
        series = sorted(reader.get_all_instances(), key=lambda d: int(d.InstanceNumber))
        image_np = np.stack([d.pixel_array for d in series])
        mask_np = extract_aligned_mask(seg_dicom, series)

        min_slices = min(mask_np.shape[0], image_np.shape[0])
        image_np = image_np[:min_slices]
        mask_np = mask_np[:min_slices]

        saved = 0
        for i in range(min_slices):
            if mask_np[i].any() and np.count_nonzero(mask_np[i]) > 100:
                if save_all_pngs(image_np, mask_np, i, output_root, patient_id):
                    saved += 1

        background = [i for i in range(min_slices) if not mask_np[i].any()]
        for i in random.sample(background, min(num_background_slices, len(background))):
            if save_all_pngs(image_np, mask_np, i, output_root, patient_id, force=True):
                saved += 1

        # print(f"{patient_id}: Saved {saved} slices")
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")

def process_all_patients(extracted_base_dir, processed_base_dir, dataset_name="Advanced-MRI-Breast-Lesion"):
    
    input_root = os.path.join(extracted_base_dir)
    output_root = os.path.join(processed_base_dir)
    os.makedirs(output_root, exist_ok=True)

    patient_dirs = sorted(glob(os.path.join(input_root, "AMBL-*")))

    for patient_dir in tqdm(patient_dirs, desc="        Processing MRI dicom images"):
        patient_id = os.path.basename(patient_dir)

        try:
            contrast_dir = glob(os.path.join(patient_dir, "*Delayed contrast*"))[0]
            seg_dir = glob(os.path.join(contrast_dir, "*ROI*"))[0]
            img_dir = glob(os.path.join(contrast_dir, "*MultiPhase*"))[0]

            seg_file = glob(os.path.join(seg_dir, "*.dcm"))[0]
            extract_segmentation(seg_file, img_dir, output_root, patient_id, num_background_slices=0)
       
        except Exception as e: # Skip if directory/ ROI file not found
            print(f"{patient_id} files not found: Skipping patient.")

if __name__ == "__main__":
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    megatron_dir = os.path.dirname(utils_dir)
    extracted_base_dir = os.path.join(megatron_dir, "ExtractedData", "Advanced-MRI-Breast-Lesion")
    processed_base_dir = os.path.join(megatron_dir, "ProcessedData", "Advanced-MRI-Breast-Lesion")

    process_all_patients(extracted_base_dir, processed_base_dir)
