#!/bin/bash

#SBATCH --job-name=my_asynch_femnist_frl
#SBATCH --time=0-40:30:00
#SBATCH --gres=gpu:nvidia-a100:1
#SBATCH --mem-per-cpu=32G
#SBATCH --output="/data/bio-eng-llm/AFL_non_convex_2/logs/%x-%j.log"

# Change directory within the script


# Load any necessary modules, if require

source /data/bio-eng-llm/virtual_envs/dnn_env/bin/activate

# Execute the Python script
python /data/bio-eng-llm/AFL_non_convex_2/src/async_fdl_femnist_delay_track_lr_non_iid_2.py

