import os
import subprocess
import shutil
import time
import sys
from tqdm import tqdm
import yaml

def load_config(config_file='datasets_config.yaml'):
    """Load the YAML configuration file"""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def download_dataset(command, target_filename, pbar, dataset_name):
    """Download a dataset and rename the file if necessary."""
    try:
        if os.path.exists(target_filename):
            pbar.write(f"✅ Dataset {target_filename} already exists. Skipping download.")
            return True
        
        # Hide the progress bar temporarily
        pbar.clear()
        
        if command.startswith("kaggle datasets"):
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            dots = 0
            while process.poll() is None:
                dots = (dots % 3) + 1
                status = f"\r⏳ Downloading {dataset_name}" + "." * dots + " " * (3 - dots)
                sys.stdout.write(status)
                sys.stdout.flush()
                time.sleep(0.5)
            
            # Clear the line when done
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            
            dataset_id = command.split('-d ')[1].split('/')[-1]
            shutil.move(f"{dataset_id}.zip", target_filename)
            print(f"✅ Successfully downloaded dataset and renamed to: {target_filename}")
        
        elif command.startswith("kaggle competition"):
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            dots = 0
            while process.poll() is None:
                dots = (dots % 3) + 1
                status = f"\r⏳ Downloading {dataset_name}" + "." * dots + " " * (3 - dots)
                sys.stdout.write(status)
                sys.stdout.flush()
                time.sleep(0.5)
            
            # Clear the line when done
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            
            dataset_id = command.split('-c ')[1].split(' ')[-1]
            shutil.move(f"{dataset_id}.zip", target_filename)
            print(f"✅ Successfully downloaded dataset and renamed to: {target_filename}")
        
        elif command.startswith("wget"):
            command_quiet = f"{command} -O {target_filename} -q"  # Add -q for quiet mode
            
            # Start a subprocess without waiting
            process = subprocess.Popen(
                command_quiet, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            dots = 0
            while process.poll() is None:
                dots = (dots % 3) + 1
                status = f"\r⏳ Downloading {dataset_name}" + "." * dots + " " * (3 - dots)
                sys.stdout.write(status)
                sys.stdout.flush()
                time.sleep(0.5)
            
            # Clear the line when done
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command_quiet)
            
            print(f"✅ Successfully downloaded and renamed to: {target_filename}")
        
        elif command.startswith("n/a"):
            print(f"💻 The {target_filename} requires a manual download due to restrictions.")

        # Display the progress bar again
        pbar.refresh()
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        pbar.refresh()
        return False
    except FileNotFoundError as e:
        print(f"❌ Skipping download. {e}")
        pbar.refresh()
        return False
    except OSError as e:
        print(f"❌ Error moving file: {e}")
        pbar.refresh()
        return False

def main():
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    megatron_dir = os.path.dirname(utils_dir)
    original_data_dir = os.path.join(megatron_dir, 'OriginalData')
    
    config_path = os.path.join(utils_dir, 'config', 'datasets_config.yaml')
    config = load_config(config_path)  # Load the YAML config file
    datasets = config.get('datasets', [])

    if not os.path.exists(original_data_dir):
        os.makedirs(original_data_dir)
        print(f"📁 Created directory: {original_data_dir}") 

    os.chdir(original_data_dir)
    print(f"📂 Changed download directory to: {original_data_dir}")
    print("🚀 Starting downloads...\n")
    
    # Create progress bar for overall progress
    total_datasets = len(datasets)
    with tqdm(total=total_datasets, desc="Overall Progress", unit="dataset", 
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
              dynamic_ncols=True) as pbar:
        for i, dataset in enumerate(datasets):
            dataset_name = dataset['name']
            pbar.write(f"\n🔄 Starting dataset {i+1}/{total_datasets}: {dataset_name}")
            
            filename = dataset['filename']
            command = dataset['command']
            
            success = download_dataset(command, filename, pbar, dataset_name)
            
            # Update progress bar
            pbar.update(1)
    
    print("\n\n ✨ All downloads completed! ✨\n")

if __name__ == '__main__':
    main()