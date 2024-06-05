import os, sys, tqdm
from transformers import (
    Blip2Processor, 
    TrainingArguments, 
    Trainer,
    HfArgumentParser,
    )
from dataset import SegFileDataset

import multiprocessing as mp
from dataclasses import dataclass, field
from utils import extract_validation_samples

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# NOTE: We can't use 3.10's new X|Y syntax b/c HfArgumentParser doesn't support it.
# https://github.com/huggingface/transformers/issues/20249
@dataclass
class ModelArguments:
    model_name_or_path: str
    save_dir: str

@dataclass
class DataArguments:
    blip_captions_dir: str
    vids_dir: str
    lta_annots_dir: str
    val_gt_caption_path: str
    # blip2_vision_feats_dir:str

@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    save_dir = model_args.save_dir    
    model_name_or_path=model_args.model_name_or_path
    blip_caps_dir=data_args.blip_captions_dir
    ego4d_lta_annots_dir = data_args.lta_annots_dir
    vids_dir = data_args.vids_dir
    val_gt_caption_path = data_args.val_gt_caption_path

    if not os.path.exists(save_dir):
         os.makedirs(save_dir)

    # Don't remove "unused columns" such as clip-related columns
    training_args.remove_unused_columns = False


    processor = Blip2Processor.from_pretrained(
        model_name_or_path
    )

    val_dataset = SegFileDataset(
         blip_caps_dir=blip_caps_dir, 
         annots_path= ego4d_lta_annots_dir + "fho_lta_val.json",
         vids_dir=vids_dir,
         processor=processor,
         gt_caption_path = val_gt_caption_path,
    )    

    print("Length of validation dataset:{}".format(len(val_dataset)))

    extract_validation_samples(val_dataset, save_dir)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
