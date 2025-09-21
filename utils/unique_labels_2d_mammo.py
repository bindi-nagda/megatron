import os
import json

def get_unique_labels(folder_path):
    """Parses JSON files to find and print unique label values as they are retrieved.
       Exits early if more than 5 unique labels are found."""
    unique_labels = set()
    print(f"Parsing JSON files in {folder_path} to extract unique labels...\n")

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if "label" in data:
                        label = data["label"]
                        if label not in unique_labels:
                            unique_labels.add(label)
                            print(f"Found label: {label}")
                            
                        if len(unique_labels) == 4:
                            print("\nMore than 4 unique labels found. Exiting early.")
                            return unique_labels
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Skipping {filename} due to error: {e}")

    return unique_labels

def find_first_index_cancer(folder_path, label):
    """Finds the first JSON file in the folder where the 'label' field is 'IndexCancer'."""
    print("Searching for the first file with label 'IndexCancer'...\n")

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"): 
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if data.get("label") == label:
                        print(data)
                        return filename 
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Skipping {filename} due to error: {e}")

    return None 

import os
import json


def has_multiple_bboxes(metadata):
    """
    Check if the metadata contains multiple bounding boxes.
    
    Args:
        metadata (dict): Loaded JSON metadata for a single image.

    Returns:
        bool: True if multiple bounding boxes are defined, False otherwise.
    """
    coords = metadata.get("coords", None)

    if coords is None:
        return False  # No bounding boxes defined

    # If coords is a list of multiple bounding boxes (list of lists)
    if isinstance(coords, list):
        if len(coords) == 0:
            return False  # No bounding boxes
        elif isinstance(coords[0], (list, tuple)) and len(coords[0]) == 4:
            # Example: [ [x_min, y_min, x_max, y_max], [x_min, y_min, x_max, y_max], ... ]
            return len(coords) > 1
        elif isinstance(coords[0], (int, float)) and len(coords) == 4:
            # Single bounding box: [x_min, y_min, x_max, y_max]
            return False
    return False  # Unknown format, assume single bbox
    
# Optional function available to the user
def remove_images_with_unknown_labels(metadata_folder, images_folder):
    """Removes image files if their corresponding metadata file has the label 'Unknown'."""
    print("Scanning metadata files for 'Unknown' labels...\n")
    removed_count = 0

    for filename in os.listdir(metadata_folder):
        if filename.endswith(".json"): 
            metadata_path = os.path.join(metadata_folder, filename)
            image_filename = os.path.splitext(filename)[0] + ".npz" 
            image_path = os.path.join(images_folder, image_filename)

            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    
                if data.get("label") == "Unknown":
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        removed_count += 1
                        print(f"Removed: {image_filename}")
                    else:
                        print(f"Image not found: {image_filename}")
                    
                    # Also delete metadata file 
                    os.remove(metadata_path)

                # Detect number of bounding boxes
                if has_multiple_bboxes(data):
                    print(f"Multiple bounding boxes found for {filename}")
                # else:
                #     print(f"Single bound box found for {filename}")

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Skipping {filename} due to error: {e}")

    print(f"\nTotal images removed: {removed_count}")

def main():
    
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    exc_data_dir = os.path.join(parent_directory, 'ExtractedData')

    metadata_folder = os.path.join(exc_data_dir, 'DeepSight-2d-Mammogram/2d_resized_256/metadata')
    images_folder = os.path.join(exc_data_dir, 'DeepSight-2d-Mammogram/2d_resized_256/images')

    # remove_images_with_unknown_labels(metadata_folder, images_folder)

    label = "Unknown"
    # Get and print unique labels
    get_unique_labels(metadata_folder)

    # Find and print the first file with label 'IndexCancer'
    first_match = find_first_index_cancer(metadata_folder, label)
    # if first_match:
    #     print(f"\nFirst file with label 'IndexCancer': {first_match}")
    # else:
    #     print("\nNo file found with label 'IndexCancer'.")

if __name__ == '__main__':
    main()
