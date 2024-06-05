import argparse, os
import pickle, json
from transformers import Blip2Processor
from dataset import SegFileDataset
from models import VideoBlipForConditionalGeneration
from utils import (
    process,
    get_gt_caption,
    DataCollatorForVideoSeq2Seq
)
from tqdm import tqdm
# from random import sample
import torch
from torch.utils.data import DataLoader
import re

def extract_synonyms(part_of_speech):
    word = ""
    synonyms = []

    if "_(" not in part_of_speech:
        return [part_of_speech]

    for ch in part_of_speech:
        if ch.isalpha():
            word += ch
        else:
            if len(word) > 0:
                synonyms.append(word)
            word = ""
    
    if len(word) > 0:
        synonyms.append(word)

    return list(set(synonyms))

def extract_verb_noun(
        generated_text:str
):
    pattern = r'"(.*?)"'
    match = re.search(pattern, generated_text)
    if match:
        substring = match.group(1)
    else:
        print("Substring not found.")

    try:
        verb, noun = substring.split(" ")
    except:
        verb, noun = "NO", "NO"

    return verb, noun

def generate(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    annots_path: str,
    ram_tags_path: str,
    save_dir: str
) -> None:

    noun_cnt = verb_cnt = act_cnt = 0 

    val_dataset = SegFileDataset(
         annots_path= annots_path,
         vids_dir=vids_dir,
         processor=processor,
         ram_tags_path=ram_tags_path,
         sample_for_test=True
    )    

    val_loader = DataLoader(
        dataset=val_dataset,
        collate_fn=DataCollatorForVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of=8
        ),
        batch_size=4,
        shuffle=False    
    )

    with open(annots_path, "r") as f:
        lta_annots = json.load(f)['clips']

    seg_file_generated_text = {}

    num_val_samples = len(val_dataset)
    for ind in tqdm(range(num_val_samples)):
        sample = val_dataset.__getitem__(ind)
        seg_file = sample['seg_file']
        # try:
        prompt = sample['prompt']

        save_path = save_dir + seg_file

        if os.path.exists(save_path):
            continue

        gt_verb, gt_noun = get_gt_caption(annots=lta_annots, seg_file=seg_file)

        # # (1, F, H, W, C)
        # # (1, 8, 1080, 1440, 3)
        frames = sample['frames'].unsqueeze(0)
        
        # # (1, C, F, H, W) 
        # # eg. (1, 3, 8, 1080, 1440)
        frames = frames.permute(0, 4, 1, 2, 3)

        inputs = process(
            processor, 
            video=frames, 
            text=prompt.strip()
        ).to(model.device)

        # dict containing keys:
        #   pixel_values: (B, C, F, H, W) eg. (1, 3, 8, 224, 224)
        #   input_ids:  (B, num_tokens) eg. (1, 8)
        #   attention_mask:  (B, num_tokens) eg. (1, 8)
        #   ram_sent_embeds: (B, F, sent_dim) eg. (1, 8, 384)
        inputs['ram_sent_embeds'] = sample['ram_sent_embeds'].unsqueeze(0).to(model.device)
        torch.save(inputs, "videoblip_inputs.pt")
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=128,
                temperature=0.0000001, # Keeping temperature less as we want a deterministic output.
                top_p=0.9,
                repetition_penalty=1.5,
                do_sample=True,
            )
        
            generated_text = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[
                0
            ].strip()

            seg_file_generated_text[seg_file] = generated_text
            
            pred_verb, pred_noun = extract_verb_noun(generated_text)

            gt_verb_syns = extract_synonyms(gt_verb)
            gt_noun_syns = extract_synonyms(gt_noun)

            verb_cnt += (pred_verb in gt_verb_syns)
            noun_cnt += (pred_noun in gt_noun_syns)
            act_cnt += ((pred_verb in gt_verb_syns) and (pred_noun in gt_noun_syns))

        # with open("generated_text.pkl", "wb") as f:
        #     pickle.dump(generated_text, f, pickle.HIGHEST_PROTOCOL)

            torch.cuda.empty_cache()

            with open(save_dir + seg_file, "wb") as f:
                pickle.dump(generated_text, f, pickle.HIGHEST_PROTOCOL)

            # print(f"Generated_text: {generated_text}")
        # except:
        #     print("Problem with {}".format(seg_file))
        #     continue

    verb_acc = verb_cnt/num_val_samples*100
    noun_acc = noun_cnt/num_val_samples*100
    act_acc = act_cnt/num_val_samples*100
    print("Verb accuracy: {}, Noun accuracy: {}, Action acuracy: {}".format(verb_acc, noun_acc, act_acc))

    with open(save_dir + "seg_file_gen_text.pkl", "wb") as f:
        pickle.dump(seg_file_generated_text, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VideoBLIP responses and compute the overall accuracy"
    )
    parser.add_argument(
        '--vids_dir', 
        type=str, 
        required=True, 
        help='video directory'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        help='Directory to save the VideoBLIP generations.', 
        required=True
    )   
    parser.add_argument(
        '--ram_tags_path', 
        type=str, 
        help='Directory where the RAM tags are saved.', 
        required=True
    )   
    parser.add_argument(
        "--annots_path", 
        required=True, 
        help="Path to Ego4D-annotations."
    ) 
    parser.add_argument("--model", default="kpyu/video-blip-flan-t5-xl-ego4d")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    vids_dir = args.vids_dir
    annots_path = args.annots_path
    save_dir = args.save_dir
    ram_tags_path = args.ram_tags_path

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open("video_blip_args.pkl", "wb") as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    model = VideoBlipForConditionalGeneration.from_pretrained(
        args.model,
        device_map='auto'
        ).to(args.device)
    
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)

    generate(
        model, 
        processor, 
        annots_path, 
        ram_tags_path,
        save_dir
    )
