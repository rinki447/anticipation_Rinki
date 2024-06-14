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
model_path="videoblip_output/checkpoint-999/"
annots_path="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_val.json"
processor_path="Salesforce/blip2-opt-2.7b"
vids_dir="/data/AmitRoyChowdhury/ego4d_data/v2/clips/"
save_dir="/data/AmitRoyChowdhury/Anirudh/videoblip_ram_prompt_attn_generations/"
ram_tags_path="/data/AmitRoyChowdhury/Anirudh/ram_openset_tags_ego4d/"
device="cuda"

python3 compute_videoblip_acc.py \
  --device ${device} \
  --model ${model_path} \
  --vids_dir ${vids_dir} \
  --annots_path ${annots_path} \
  --ram_tags_path ${ram_tags_path} \
  --save_dir ${save_dir} \
  --processor ${processor_path}