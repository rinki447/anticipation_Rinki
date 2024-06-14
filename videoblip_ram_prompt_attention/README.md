**For Shao Yuan: At least 3 GPU required

You may face the same errors you faced running last model. So i have compressed "python3 train_videoblip_rinki.py --*** " line by removing spaces inside "slurm_scripts/train_videoblip_no_slurm.sh", yet there can be spaces, which you might need to remove to avoid argument error in your system.

Also, last time, you had an error of extra "r" at the end of the annotation file, so you had to add 
f=f.strip() kind of line after **[with open(xyz,"r) as f:]** in dataset.py. Similarly here if you face the same issue, you can add similar line under dataset_rinki.py



> Save EGo4D clips in "folder_clips" 

> Save Ram-Tags in "folder_tags" from https://drive.google.com/file/d/1bPdlewf9ICRHOuW3qK14BdUZgPuDGWMi/view?usp=sharing
(same folder Anirudh shared with you before)

> Save Annotations under "your_annot_folder" from https://drive.google.com/drive/folders/1G2_Ow6TjgCoBgevBtUNKWdPDzLM6R9Jx?usp=sharing

>>  eg: for me the folder path is "/data/AmitRoyChowdhury/ego4d_data/v2/annotations/"







> Inside slurm_scripts/train_videoblip_no_slurm.sh, replace the paths:

1> RAM_TAGS_PATH="folder_name" with your "folder_tags"

2> VIDS_DIR="folder_name" with your "folder_clips"

3> LTA_ANNOTS_DIR="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/" with "your_annots_folder" path

4>

ego4d_annot_path_train="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json"  =  your_annots_folder+"fho_lta_train.json"

ego4d_annot_path_test="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_val.json"  =  your_annots_folder+"fho_lta_val.json"

5> 

forecast_annot_tr_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_train_annots.json"  =  your_annots_folder path +"anticipation_final_train_annots.json"

forecast_annot_te_dir="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/anticipation_final_test_annots.json" =  your_annots_folder path +"anticipation_final_test_annots.json"

6>  SAVE_DIR="" , change the path to your chosen path, to save results

7>  OUT_Dir="" , change this path to your chosen folder, to save checkpoints of the model

8> change the line "conda activate llava" into "conda activate eilev" assuming eilev is your environment




> To run the code I used:
bash slurm_scripts/train_videoblip_no_slurm.sh


####################################################################################################################################################################################



![ram_caption_fusion](https://github.com/rinki447/anticipation_Rinki/assets/132046732/9eaf1ac8-1f77-4023-a248-39858b977006)




