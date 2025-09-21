DIRNAME=My_Training_Job_$(date +%Y%m%d_%H%M%S)
mkdir -p /<your-path>/megatron/Results/$DIRNAME
cd /<your-path>/megatron/Results/$DIRNAME
cp /<your-path>/megatron/train/meta_train.py .

# # Copy .yaml file so it's used in a job-specific way
# mkdir config
# cp /home1/bnagda2015/megatron/utils/config/datasets_config.yaml config
sbatch --export=ALL,DATA_FOLDER=$DIRNAME /<your-path>/megatron/run_job.std