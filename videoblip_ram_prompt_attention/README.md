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

3. Extracting the RAM_tags from [link](https://drive.google.com/file/d/1bPdlewf9ICRHOuW3qK14BdUZgPuDGWMi/view?usp=sharing)

# Training the model

Update the correct paths to the Ego4D clips, annotations and other files in the `slurm_scripts/train_videoblip.sh`.
```
bash slurm_scripts/train_videoblip.sh
```

# Computing the accuracy

```
bash slurm_scripts/compute_videoblip_acc.sh
```
