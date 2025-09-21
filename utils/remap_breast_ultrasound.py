import os
from glob import glob

def remap_class_ids(labels_dir):
    txt_files = glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)

    for file_path in txt_files:
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue  # skip empty lines

            class_id = int(parts[0])
            if class_id == 1:
                parts[0] = '0'
            elif class_id == 2:
                parts[0] = '1'
            else:
                continue  # ignore or skip unexpected class IDs

            new_lines.append(" ".join(parts))

        # Overwrite the file
        with open(file_path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")

    print(f"Remapped {len(txt_files)} label files in {labels_dir}")

# Example usage:
labels_base_dir = "/ProcessedData/Breast-Ultrasound/labels/train"  # contains train/ and val/
remap_class_ids(labels_base_dir)
