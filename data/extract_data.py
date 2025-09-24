import os
import yaml
import zipfile
import tarfile
from tqdm import tqdm

def load_config(config_file='datasets_config.yaml'):
    """Load the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def unzip_dataset(zip_path, extract_to):
    """Unzip the specified zip file to the given location with a progress bar."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()

        print(f"Extracting {os.path.basename(zip_path)} (zip)...")
        for member in tqdm(members, desc=f"Extracting {os.path.basename(zip_path)}", unit="file"):
            zip_ref.extract(member, path=extract_to)

def untar_dataset(tar_path, extract_to):
    """Untar the specified tar file to the given location with a progress bar."""
    with tarfile.open(tar_path, 'r') as tar_ref:
        members = tar_ref.getmembers()

        print(f"Extracting {os.path.basename(tar_path)} (tar)...")
        for member in tqdm(members, desc=f"Extracting {os.path.basename(tar_path)}", unit="file"):
            tar_ref.extract(member, path=extract_to)

def main():
    # Paths relative to this script location
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    megatron_dir = os.path.dirname(utils_dir)

    original_data_dir = os.path.join(megatron_dir, 'OriginalData')
    processed_data_dir = os.path.join(megatron_dir, 'ExtractedData')

    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"📂 Processed data directory is: {processed_data_dir}\n")

    config_path = os.path.join(utils_dir, 'config', 'datasets_config.yaml')
    config = load_config(config_path)

    for dataset in config['datasets']:
        filename = dataset['filename']
        dataset_name = dataset['name']

        archive_path = os.path.join(original_data_dir, filename)
        target_extract_dir = os.path.join(processed_data_dir, os.path.splitext(filename)[0])

        if not os.path.exists(archive_path):
            print(f"❌ File not found: {archive_path}. Skipping...\n")
            continue

        if os.path.exists(target_extract_dir) and any(os.scandir(target_extract_dir)):
            print(f"✅ Already extracted: {target_extract_dir}. Skipping...\n")
            continue

        os.makedirs(target_extract_dir, exist_ok=True)

        # Determine whether to unzip or untar
        if filename.endswith('.zip'):
            unzip_dataset(archive_path, target_extract_dir)
        elif filename.endswith('.tar') or filename.endswith('.tar.gz') or filename.endswith('.tgz'):
            untar_dataset(archive_path, target_extract_dir)
        else:
            print(f"⚠️ Unsupported file type: {filename}. Skipping...\n")

    print("\nAll datasets processed.")

if __name__ == "__main__":
    main()
