# MEGATRON: Meta-Learning for Next-Generation Advanced Technology Realization & Acceleration

MEGATRON is a meta-learning framework for multimodal medical imaging data processing and model training. This repository provides tools to download and process data from open-source GREI repositories, and train meta-models on these datasets for few-shot object detection, with support for extensible custom datasets and SLURM-based job submission for GPU clusters.

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
python utils/download_data.py
python utils/extract_data.py
python utils/prep_data.py
```

The code has mechanisms to validate data and perform error-handling. All processes will be logged, and failed processed and can be safely re-initiated without affected already processed data. 

To force reprocessing of any dataset(s), you can run

```bash
python utils/prep_data.py --force
```

**Customizing Datasets:**

* Select which datasets to download, extract, and process by modifying the `config/datasets_config.yaml` file.
* Add your own custom datasets and extend the processing pipeline using the function templates in the `*_data.py` files located in the `utils/` directory.

---

## Model Training

### 1. Configure Training

* Edit `batch_job.sh` to set the desired name for your results directory. This is where logs and model outputs will be saved.
* Update paths in `run_job.std` to match your directory locations.
* Specify which GPU nodes you want to use for training in `run_job.std`.

### 2. Run Training

Start a training run by executing:

```bash
sh batch_job.sh
```

The script will submit jobs to SLURM and manage the training workflow.

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