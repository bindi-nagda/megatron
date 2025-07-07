DIRNAME=FiveTask_MetaLearn_meta_warmup_lr0.05_ie30
mkdir -p /home1/bnagda2015/megatron/Results/$DIRNAME
cd /home1/bnagda2015/megatron/Results/$DIRNAME

# # Copy .yaml file so it's used in a job-specific way
# mkdir config
# cp /home1/bnagda2015/megatron/utils/config/datasets_config.yaml config

sbatch --export=ALL,DATA_FOLDER=$DIRNAME /home1/bnagda2015/megatron/run_job.std