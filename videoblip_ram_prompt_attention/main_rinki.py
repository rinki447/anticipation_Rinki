import os, sys, tqdm

import pickle
import random
import numpy as np
import wandb
import torch
from dataset_rinki import SegFileDataset #dataset file name changed by Rinki
from transformers import (
    Blip2Processor, 
    TrainingArguments, 
    Trainer,
    HfArgumentParser,
    BitsAndBytesConfig
    )
import multiprocessing as mp
from models_rinki import VideoBlipForConditionalGeneration

from utils import DataCollatorForVideoSeq2Seq

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from dataclasses import dataclass, field

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# NOTE: We can't use 3.10's new X|Y syntax b/c HfArgumentParser doesn't support it.
# https://github.com/huggingface/transformers/issues/20249
@dataclass
class ModelArguments:
    model_name_or_path: str
    save_dir: str
    processor_path: str

@dataclass
class DataArguments:
    vids_dir: str
    lta_annots_dir: str
    ram_tags_path:str
    n_present_video:str ######## added by Rinki
    n_future_video: str ##### added by Rinki
    no_frame: str #### added by Rinki
    forecast_annot_tr_dir: str # added by Rinki
    forecast_annot_te_dir: str #added by Rinki 

@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch")
    NUM_TRAIN_EPOCHS: int=field(default=1)
    # resume_from_checkpoint: bool = True

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    save_dir = model_args.save_dir    
    model_name_or_path=model_args.model_name_or_path
    processor_path = model_args.processor_path
    ego4d_lta_annots_dir = data_args.lta_annots_dir
    vids_dir = data_args.vids_dir
    ram_tags_path = data_args.ram_tags_path
    n_present_video= data_args.n_present_video ######## added by Rinki
    n_future_video= data_args.n_future_video ##### added by Rinki
    no_frame=data_args.no_frame #### added by Rinki
    forecast_annot_tr_dir =data_args.forecast_annot_tr_dir # added by Rinki
    forecast_annot_te_dir =data_args.forecast_annot_te_dir #added by Rinki 

    if not os.path.exists(save_dir):
         os.makedirs(save_dir)

    # Don't remove "unused columns" such as clip-related columns
    training_args.remove_unused_columns = False

    wandb.init(
        
        # set the wandb project where this run will be logged
        project="VideoBLIP_Qformer_RAM_Attention",

        # track hyperparameters and run metadata
        config= training_args
    )

    wandb.run.log_code(".")

    # processor = Blip2Processor.from_pretrained(
    #     model_name_or_path
    # )

    processor = Blip2Processor.from_pretrained(
        processor_path
    )

    # Create training and validation dataset
    '''train_dataset = SegFileDataset(
        annots_path= ego4d_lta_annots_dir + "fho_lta_train.json",
        vids_dir=vids_dir,
        processor=processor,
        ram_tags_path = ram_tags_path
    )'''
    # arguments to dataset changed by Rinki to fit dataset_rinki.py
    train_dataset=SegFileDataset(forecast_annot_tr_dir,ram_tags_path,no_frame,processor,num_query_tokens=32,decoder_only_lm=True,sample_for_test=False) #clip_path is within forecast_annot_tr_dir
    print("Length of training dataset:{}".format(len(train_dataset)))

    '''val_dataset = SegFileDataset(
         annots_path= ego4d_lta_annots_dir + "fho_lta_val.json",
         vids_dir=vids_dir,
         processor=processor,
         ram_tags_path = data_args.ram_tags_path
    )  '''  
    # arguments to dataset changed by Rinki to fit dataset_rinki.py
    val_dataset=SegFileDataset(forecast_annot_te_dir,ram_tags_path,no_frame,processor,num_query_tokens=32,decoder_only_lm=True,sample_for_test=False) #clip_path is within forecast_annot_te_dir
    print("Length of validation dataset:{}".format(len(val_dataset)))

    # rand_ind = random.randint(0, len(val_dataset))
    # rand_sample = val_dataset.__getitem__(rand_ind)

    # with open("rand_sample.pkl", "wb") as f:
    #     pickle.dump(rand_sample, f, pickle.HIGHEST_PROTOCOL)

    # sys.exit()
    #model_args.model_name_or_pathmodel_args.model_name_or_path
    model = VideoBlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        low_cpu_mem_usage=False if is_deepspeed_zero3_enabled() else True,
        device_map='auto'
    )
    
    # model.load_adapter(peft_model_id)
    # model.add_adapter(peft_config)
    
    model.config.text_config.eos_token_id = processor.tokenizer.eos_token_id

    # freeze everything except for qformer
    for param in model.vision_model.parameters():
        param.requires_grad = False
        
    for param in model.language_model.parameters():
        param.requires_grad = False

    for param in model.multihead_attn.parameters():
        param.requires_grad = True

    for param in model.vis_proj.parameters():
        param.requires_grad = True

    for param in model.text_proj.parameters():
        param.requires_grad = True


    # we need to enable input require grads since the vision model (the first layer) is
    # frozen.
    model.enable_input_require_grads()

    print("Total number of trainable parameters:{}".format(count_parameters(model)))
    # model.print_trainable_parameters()

    # Load the best model at the end so we can save it
    training_args.load_best_model_at_end = True
    training_args.report_to=["wandb"]
    #for attr, value in training_args.__dict__.items():
        #print(f"{attr}: {value}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        )    
    )

    #print("no of epochs",training_args.num_train_epochs)
    #print("training_args.resume_from_checkpoint:",training_args.resume_from_checkpoint)
    #print("train batch size per device",training_args.per_device_train_batch_size)
    #print("eval batch size per device",training_args.per_device_eval_batch_size)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)



    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    wandb.finish()


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
