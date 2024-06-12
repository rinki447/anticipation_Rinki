import argparse, os
import pickle, json
from transformers import Blip2Processor
from dataset_rename import SegFileDataset
from models import VideoBlipForConditionalGeneration
from utils import (
	get_gt_caption,
	DataCollatorForVideoSeq2Seq,generate_input_ids_and_labels,generate_input_ids
)
from tqdm import tqdm
# from random import sample
import torch
from torch.utils.data import DataLoader
import re

PROMPTS = [
    "What are the next 20 future actions that the camera wearer is going to do?",
    "Question: What are the next 20 future actions that the camera wearer is going to do?",
    "What are the next 20 future actions that the camera wearer is going to do? An answer to the question is",
    "Q: What are the next 20 future actions that the camera wearer is going to do? A:",
    "Given the video frames, answer the following question. What are the next 20 future actions that the camera wearer is going to do?",
    "Based on the video frames, respond to this question: What are the next 20 future actions that the camera wearer is going to do? "
    "Answer:",
    "Use the provided video frames to answer the question: WWhat are the next 20 future actions that the camera wearer is going to do?",
    'What is the answer to the following question? "What are the next 20 future actions that the camera wearer is going to do?"',
    'The question "What are the next 20 future actions that the camera wearer is going to do?" can be answered using the video frames. '
    "The answer is",
]

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

'''def extract_verb_noun(
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

	return verb, noun'''

def generate(
	model: VideoBlipForConditionalGeneration,
	processor: Blip2Processor,
	forecast_annot_te_dir: str,
	save_dir: str
) -> None:

	#noun_cnt = verb_cnt = act_cnt = 0 

	'''val_dataset = SegFileDataset(
		 annots_path= annots_path,
		 vids_dir=vids_dir,
		 processor=processor
		)   '''

	val_dataset=SegFileDataset(forecast_annot_te_dir,processor,num_query_tokens=32,decoder_only_lm=True,sample_for_test=False) #clip_path is within forecast_annot_te_dir
	 
	
	val_loader = DataLoader(
		dataset=val_dataset,
		collate_fn=DataCollatorForVideoSeq2Seq(
			processor.tokenizer,
			pad_to_multiple_of=8
		),
		batch_size=4,
		shuffle=False    
	)

	


	seg_file_generated_text = {}

	num_val_samples = len(val_dataset)
	for ind in tqdm(range(num_val_samples)):
		sample = val_dataset.__getitem__(ind) # need to return short name for 2 observed files, in dict sample["seg_file"]
		seg_file = sample['seg_file']

		print("file_name",seg_file)

		prompt = sample['prompt']
		# try:
		#prompt = sample['prompt']
		print("prompt",prompt)

		save_path = save_dir + seg_file #path for the 2 obs clip data output to save

		print("save path",save_path)
		if os.path.exists(save_path):
			continue

		#gt_verb, gt_noun = sample["seg_truth"]

		
		# # (B, N_obs,F, H, W, C)
		# # (1,2, 8, 1080, 1440, 3)
		print("sample['frames'] shape:",sample['frames'].shape)
		frames = sample['frames'].unsqueeze(0)


		batch,n_obs,channel,time,_,_=frames.shape
		print("frames shape",frames.shape)#torch.Size([1, 2, 3, 8, 224, 224])
		
		#inputs={}
		#inputs["pixel_values"]=frames
		#inputs["input_id"]=
		#inputs["attention_mask"]=



		'''inputs = process(
			processor, 
			video=frames, 
			text=prompt.strip()
		).to(model.device)'''


		'''inputs = processor(
		    images=frames, 
		    text=prompt.strip(), 
		    return_tensors="pt"
		)
		_,_,_, height, weight = inputs.pixel_values.size()
		inputs["pixel_values"] = inputs.pixel_values.view(
		            batch, n_obs, time, channel, height, weight
		        ).permute(0, 1,3, 2, 4, 5)'''

		

		# dict containing keys:
		#   pixel_values: (B, C, F, H, W) eg. (1, 3, 8, 224, 224)
		#   input_ids:  (B, num_tokens) eg. (1, 8)
		#   attention_mask:  (B, num_tokens) eg. (1, 8)
		#   ram_sent_embeds: (B, F, sent_dim) eg. (1, 8, 384)
		#inputs['ram_sent_embeds'] = sample['ram_sent_embeds'].unsqueeze(0).to(model.device)
		#torch.save(inputs, "videoblip_inputs.pt")
		

		inputs={}
		pixels=frames.to(model.device)
		inputs["pixel_values"]=pixels

		tokenized_inputs = generate_input_ids(
		        tokenizer=processor.tokenizer,
		        prompt=PROMPTS[0],
		        decoder_only_lm=True
		        )
		
		
		inputs["input_ids"]=tokenized_inputs["input_ids"].unsqueeze(0).to(model.device)
		
		
		with torch.inference_mode():
			generated_ids = model.generate(
				**inputs,
				num_beams=12,
				max_new_tokens=128,
				temperature=0.1, # Keeping temperature less as we want a deterministic output.
				top_p=0.7,
				repetition_penalty=1.5,
				do_sample=True,
				)

			print("generated id shape:",generated_ids.shape)
		
			generated_text = processor.batch_decode(
				generated_ids, 
				skip_special_tokens=True
				)
			print("generated text",generated_text)
			
			
		print("ground truth",sample["ground_truth"])
		#seg_file_generated_text[seg_file].append(generated_text)
			
				#pred_verb, pred_noun = extract_verb_noun(generated_text)
		ed_final=0
		for pred_sentence in generated_text:
				#for synonym_sentence in ground_sentence:
				ground_sentence=sample["ground_truth"]
				ed=edit_distance(pred_sentence,ground_sentence)
				if ed>ed_final:
					ed_final=ed
					best_sentence=pred_sentence

		print(best_sentence)   
		exit()

			

		'''verb_cnt += (pred_verb in gt_verb_syns)
		noun_cnt += (pred_noun in gt_noun_syns)
		act_cnt += ((pred_verb in gt_verb_syns) and (pred_noun in gt_noun_syns))'''

		# with open("generated_text.pkl", "wb") as f:
		#     pickle.dump(generated_text, f, pickle.HIGHEST_PROTOCOL)

		torch.cuda.empty_cache()

		with open(save_dir + seg_file, "wb") as f:
				pickle.dump(generated_text, f, pickle.HIGHEST_PROTOCOL)

			# print(f"Generated_text: {generated_text}")
		# except:
		#     print("Problem with {}".format(seg_file))
		#     continue

	'''verb_acc = verb_cnt/num_val_samples*100
	noun_acc = noun_cnt/num_val_samples*100
	act_acc = act_cnt/num_val_samples*100
	print("Verb accuracy: {}, Noun accuracy: {}, Action acuracy: {}".format(verb_acc, noun_acc, act_acc))'''

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
	'''parser.add_argument(
		'--ram_tags_path', 
		type=str, 
		help='Directory where the RAM tags are saved.', 
		required=True
	)   '''
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
	#ram_tags_path = args.ram_tags_path

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	with open("video_blip_args_rinki.pkl", "wb") as f:
		pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

	model = VideoBlipForConditionalGeneration.from_pretrained(
		args.model,
		device_map='auto'
		).to(args.device)
	
	if args.processor is None:
		args.processor = args.model
	processor = Blip2Processor.from_pretrained(args.processor)
	#print("done")
	
	generate(
		model, 
		processor, 
		annots_path, 
		save_dir
	)
