# MEGATRON: Meta-Learning for Next-Generation Advanced Technology Realization & Acceleration

MEGATRON is a meta-learning framework for multimodal medical imaging data processing and model training. This repository provides tools to download and process data from open-source GREI repositories, and train meta-models on these datasets for few-shot object detection, with support for extensible custom datasets and SLURM-based job submission for GPU clusters.

For additional information, please read the accompanying paper at 
<a href="https://earlybyrdteam.github.io/megatron-site/" target="_blank" rel="noopener noreferrer">this link</a>.

---

## System Requirements

Before using MEGATRON, ensure your system meets the following minimum requirements:

| Item         | Minimum Required |
|--------------|-----------------|
| GPU          | CUDA-enabled with compute capability >=3.0 |
| CUDA Toolkit | 11.8            |
| Storage      | 1 TB             |
| vRAM         | 40 GB            |
| Linux Distro | Ubuntu 22.04     |
| Conda        | 25.1.1           |
| Python       | 3.10             |

---

## Installation

### 1. Install Miniconda

To manage Python dependencies, you need Miniconda installed. Follow these steps:

#### Download Miniconda installer for Linux (Python 3.x)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

#### Run the installer
```
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the on-screen prompts and choose the default options. Restart your shell after installation.

### 2. Set up the Conda Environment

Run the provided setup script to create the MEGATRON environment:

```bash
sh setup/setup_env.sh
```

This will create a Conda environment named `meta310`. 

There are two environments available for : 
1. GPU-based processing (meta310) 
2. CPU-based processing (myenv310).

To create the CPU environment, simply replace `environment.yml` to `environmentNoCuda.yml` in the `setup/setup_env.sh` script.

### 3. Activate the Environment

```
conda activate meta310
```

or

```
conda activate myenv310
```

---

## Data Download and Processing

After activating the environment, you can download and process datasets by running the following commands in sequence:

```bash
python data/download_data.py
python data/extract_data.py
python data/prep_data.py
```
The downloaded, exctracted and processed data will be stored in automatically-created folders named `OriginalData`, `ExtractedData`, and `ProcessedData`. The `OriginalData` and `ExtractedData` can be safely discarded after all processed data has been generated.

The code has mechanisms to validate data and perform error-handling. All processes will be logged to terminal, and failed processed can be safely re-initiated without compromising already processed data. 

Running the above commands repeatedly will skip datasets that have already been processed. To force reprocessing of any dataset(s), you can run

```bash
python data/prep_data.py --force
```

For a final sanity check of the processed data, you can run the following utility file by passing in the dataset name. This will generate the processed images with bounding boxes overlays.

```bash
python utils/sanity_check.py --dataset Breast-Ultrasound --num-samples 20
```

**Customizing Datasets:**

* Select which datasets to download, extract, and process by modifying the `config/datasets_config.yaml` file.
* Add your own custom datasets and extend the processing pipeline using the function templates in the `*_data.py` files located in the `data/` directory.

---

## Model Training

### 1. Configure Training

* Edit `batch_job.sh` to set the desired name for your results directory. This is where logs and model outputs will be saved.
* Update paths in `run_job.std` to match your directory locations.
* Specify which GPU nodes you want to use for training in `run_job.std` by modifying the #SBATCH --partition and #SBATCH --partition fields.

### 2. Run Training

Start a training run by executing:

```bash
sh batch_job.sh
```

The script will submit jobs to SLURM and manage the training workflow. The `train.py` script already has mechanisms in place to periodically save model files and will continually save the best performing model as training progresses. All performance metrics will be logged at every epoch in `.json` and `.out` files inside the specified results directory.

### 3. View Results

The training run logs, plots and model outputs can be viewed in the results directory specified in the `batch_job.sh` file.

---

## Notes

* Ensure that CUDA Toolkit 11.8 is properly installed and compatible with your GPU drivers.
* All preprocessing steps, logging, and error handling are handled within the pipeline to facilitate reproducibility.
* You can extend MEGATRON to new datasets or modify processing steps using the templates in the `utils/` directory.

---

## License

This repository is open-source. Please see the [LICENSE](LICENSE) file for details.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---
