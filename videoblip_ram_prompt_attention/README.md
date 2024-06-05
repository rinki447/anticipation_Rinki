**For Shao Yuan: At least 3 GPU required

> Save EGo4D clips in "folder_clips" 

> Save Ram-Tags in "folder_tags" from https://drive.google.com/file/d/1bPdlewf9ICRHOuW3qK14BdUZgPuDGWMi/view?usp=sharing
(same folder Anirudh shared with you before)

> Save Annotations under "your_annot_folder" from https://drive.google.com/drive/folders/1G2_Ow6TjgCoBgevBtUNKWdPDzLM6R9Jx?usp=sharing

>>  eg: for me the folder path is "/data/AmitRoyChowdhury/ego4d_data/v2/annotations/"




> Inside slurm/train_videoblip_rinki.sh, replace the paths:

1> line 31, RAM_TAGS_PATH="folder_name" with your "folder_tags"

2> line 29, VIDS_DIR="folder_name" with your "folder_clips"

3> line 30, replace LTA_ANNOTS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/" with "your_annots_folder" path

4> line 16 and 17:

ego4d_annot_path_train="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json"  =  your_annots_folder+"fho_lta_train.json"

ego4d_annot_path_test="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_val.json"  =  your_annots_folder+"fho_lta_val.json"

5> line 19, 20:

forecast_annot_tr_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_train_annots.json"  =  your_annots_folder path +"anticipation_final_train_annots.json"

forecast_annot_te_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_test_annots.json" =  your_annots_folder path +"anticipation_final_test_annots.json"

6> line 32: SAVE_DIR="" , change the path to your chosen path, to save results

7> line 38: OUT_Dir="" , change this path to your chosen folder, to save checkpoints of the model

> To run the code I used:
sbatch -p vcggpu --gres=gpu:4 --mem=30g --time=07-00:01:00 slurm_scripts/train_videoblip_rinki.sh


####################################################################################################################################################################################




![anticipation](https://github.com/Anirudh257/cluster_backup/assets/132046732/04748d62-93ff-4744-82a3-3a8c9d17e911)
# Architecture

<figure>
  <img src="Plots/VideoBLIP_RAM_Prompt_Attention.png">
  <figcaption>VideoBLIP with fused textual and vision features.</figcaption>
</figure>

# Setup

1. Install the requirements for running the code

```
conda env create -f blip_ram.yml
```

2. Download the Ego4D clips and annotations following https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md:

```
python -m ego4d.cli.cli \
    --output_directory=${EGO4D_DIR} \
    --datasets annotations clips lta_models \
    --benchmarks FHO
    --version v2
```


