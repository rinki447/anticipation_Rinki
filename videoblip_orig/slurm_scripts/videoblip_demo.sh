#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.
#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job
#SBATCH --cpus-per-task=1 # Each task needs only one CPU
#SBATCH --mem=5G # This particular job won't need much memory
#SBATCH --time=1-00:01:00  # 1 day and 1 minute 
#SBATCH --job-name="batch job test"
#SBATCH -p cpu # You could pick other partitions for other jobs
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.


# Place any commands you want to run below

conda activate eilev
model_path="videoblip_orig_cap_model/"
input_path="/data/AmitRoyChowdhury/Anirudh/videoblip_inputs/eaac906a-e4c0-4760-9bca-99a707780645_start_frame_7334_end_frame_7574.pkl"
processor_path=${model_path}
device="cuda"

python3 video_blip_generate_action_narration.py \
  --device ${device} \
  --model ${model_path} \
  --input_path ${input_path} \
  --processor ${processor_path} \
  --prompt "What is the camera wearer doing?"