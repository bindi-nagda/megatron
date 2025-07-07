#!/bin/bash

# File: setup_env.sh
# Location: megatron/setup_env.sh

# Step 1: Move to the folder where this script is located (megatron/)
cd "$(dirname "$0")"

# Step 2: Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "❌ Conda is not installed. Please install Miniconda first."
    exit 1
fi

# Step 3: Create the environment using environment.yml
if [ -f "environment.yml" ]; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml

    # The line below is optional and will setup an env if you don't have CUDA. 
    # This environment will still support data processing.
    # conda env create -f environmentNoCuda.yml 
else
    echo "❌ environment.yml not found! Make sure it's in the same folder as setup_env.sh."
    exit 1
fi

# Step 4: Activate instructions
echo ""
echo "Environment created successfully!"
echo "To activate it, run:"
echo "conda activate <your-env-name>"
