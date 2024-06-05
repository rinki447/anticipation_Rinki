import os, sys, tqdm
import argparse
import pickle
import random
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from dataset import SegFileDataset
from transformers import (
    Blip2Processor, 
    TrainingArguments, 
    Trainer,
    HfArgumentParser,
    AutoModelForCausalLM
    )
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from peft import (
    LoraConfig, 
    prepare_model_for_int8_training, 
    get_peft_model,
    PeftModel
)
import multiprocessing as mp
from models import FramesBlipForConditionalGeneration
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from dataclasses import dataclass, field
from eilev.data.utils import DataCollatorForInterleavedVideoSeq2Seq

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_args_parser():
    parser = argparse.ArgumentParser('Training Interleaved image-caption BLIP model.',
                                     add_help=False)
    parser.add_argument("--seed", default=42, help = "Seed to control randomness")
    parser.add_argument("--num_epochs", default=40, type=int, 
                        help="Number of epochs to train the model")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of workers")
    parser.add_argument("--batch-size", default=256, type=int, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay") 
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate") 
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature") 
    parser.add_argument("--ego4d_lta_annots_dir", required=True, 
                        help="Path to LTA annotations")
    parser.add_argument("--vids_dir", required=True, help="Path to Ego4D videos.")
    parser.add_argument("--blip_captions_dir", required=True, 
                        help="Path to directory containing BLIP captions")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--model_name_or_path", type=str, required=False, 
                        default="Salesforce/blip2-opt-2.7b", 
                        help="Path to the SpaceTimeTransformer checkpoint.")
    parser.add_argument("--train_gt_caption_path", type=str, required=True, 
                        help="Pickle file containing the training segment and \
                        sentence-level narrations")
    parser.add_argument("--val_gt_caption_path", type=str, required=True, 
                        help="Pickle file containing the validation segment and \
                        sentence-level narrations")        
    return parser


def main(args):

    save_dir = args.save_dir    
    model_name_or_path=args.model_name_or_path
    blip_caps_dir=args.blip_captions_dir
    ego4d_lta_annots_dir = args.ego4d_lta_annots_dir
    vids_dir = args.vids_dir
    train_gt_caption_path = args.train_gt_caption_path
    val_gt_caption_path = args.val_gt_caption_path

    if not os.path.exists(save_dir):
         os.makedirs(save_dir)

    wandb.init(
        
        # set the wandb project where this run will be logged
        project="QFormer_captions",

        # track hyperparameters and run metadata
        config= args
    )

    wandb.run.log_code(".")

    processor = Blip2Processor.from_pretrained(
        model_name_or_path
    )

    # Create training and validation dataset
    train_dataset = SegFileDataset(
         blip_caps_dir=blip_caps_dir, 
         annots_path= ego4d_lta_annots_dir + "fho_lta_train.json",
         vids_dir=vids_dir,
         processor=processor,
         gt_caption_path = train_gt_caption_path
    )

    val_dataset = SegFileDataset(
         blip_caps_dir=blip_caps_dir, 
         annots_path= ego4d_lta_annots_dir + "fho_lta_val.json",
         vids_dir=vids_dir,
         processor=processor,
         gt_caption_path = val_gt_caption_path
    )    

    print("Number of training samples: {}".format(len(train_dataset)))
    print("Number of validation samples: {}".format(len(val_dataset)))

    # Define model
    # model = FramesBlipForConditionalGeneration.from_pretrained(
    #     model_name_or_path,
    #     low_cpu_mem_usage=False if is_deepspeed_zero3_enabled() else True,
    #     torch_dtype=torch.float16,
    #     load_in_8bit=True,
    #     device_map="auto"
    # )

    # # Adding PEFT layers on top of the model for efficient finetuning
    # peft_model_id = "ybelkada/opt-350m-lora"
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # peft_adapter = AutoModelForCausalLM.from_pretrained(peft_model_id)
    model = FramesBlipForConditionalGeneration.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=False if is_deepspeed_zero3_enabled() else True,
        load_in_4bit=True,
        device_map="auto"
    )
    
    model.config.text_config.eos_token_id = processor.tokenizer.eos_token_id

    # freeze everything except for qformer
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False

    # we need to enable input require grads since the vision model (the first layer) is
    # frozen.
    model.enable_input_require_grads()

    # Add a PEFT model adapter
    model = PeftModel(model, peft_config)

    # print("Total number of trainable parameters:{}".format(count_parameters(model)))
    model.print_trainable_parameters()

    # Load the best model at the end so we can save it
    training_args.load_best_model_at_end = True
    training_args.report_to=["wandb"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForInterleavedVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        )    
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Interleaved image-caption BLIP model.', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main(args)