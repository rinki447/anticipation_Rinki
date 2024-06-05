import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_gt_caption,extract_frames_from_video,generate_input_ids_and_labels,video_loader_by_frames
import glob
from transformers import (
	Blip2Processor, 
	TrainingArguments, 
	Trainer,
	HfArgumentParser,
	BitsAndBytesConfig
	)
import argparse
import json
import pickle 
from transformers import BatchFeature
from sentence_transformers import (
    SentenceTransformer, 
    util
)
from utils import DataCollatorForVideoSeq2Seq

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

class SegFileDataset(Dataset):

	def __init__(self,annots_path,ram_tags_path,num_frames,processor,num_query_tokens=32,decoder_only_lm=True,sample_for_test=False):
	

		
		with open(annots_path, "r") as f:
			self.annots = json.load(f)
		self.annots=self.annots
		self.ram_tags_path=ram_tags_path
		self.processor=processor 
		self.num_query_tokens = num_query_tokens
		self.decoder_only_lm = decoder_only_lm
		self.sample_for_test = sample_for_test
		#self.seg_files = self.load_seg_sent_narrs()###################################

		self.sent_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

		
	def extract_frames(self, clip_path, frame_ids):
		"""Given the seg_file, extract the frame-level features

		   Args:
			seg_file (str): Ego4D seg_file of the format clip_id_start_frame_XX_end_frame_YY

		   Returns:
			inputs (dict): preprocessed subset of frames from the video between the 
				start_frame=XX and end_frame=YY. 
		"""
		
		# (num_frames, H,W,C)
		# eg. (4, 1080, 1440, 3)
		pil_frames = extract_frames_from_video(clip_path,frame_ids)

		# Preprocess frames
		# dict['pixel_values': (num_frames, C, H, W): (4, 3, 224, 224)]
		inputs = self.processor(pil_frames, return_tensors='pt') ##############
				
		return inputs

	def __len__(self):
		# Return the total number of video sets
		return len(self.annots)

	def __getitem__(self, idx):


		clip_path,set_dict = self.annots[idx]  #### more steps
		clip_uid=clip_path.split("/")[-1].split(".")[0]
		observed_seg_info_list=set_dict["obs_clips"]
		forecast_seg_info_list=set_dict["forecast_clips"]


		input_frames=torch.tensor([])
		tokenized_input=[]
		ram_sent_embeds=torch.tensor([])

		with torch.no_grad():

			# info from observed segments
			for seg_info in observed_seg_info_list:

				# find ram_frame_no and ram tags caption for each of the 4 observed input videos
				start_frame=seg_info["action_clip_start_frame"]
				end_frame=seg_info["action_clip_end_frame"]
				seg_name = "{}_start_frame_{}_end_frame_{}".format(clip_uid, start_frame, end_frame)
				seg_file=seg_name+".pkl"

				with open(self.ram_tags_path + seg_file, "rb") as f:
					ram_tags_dict = pickle.load(f)

				frame_ids = []
				ram_sent_seg_embeds = torch.tensor([])

				for ind, (frame_ind, ram_tag) in enumerate(ram_tags_dict.items()):
					if ind % 2 == 0:
						frame_ids.append(frame_ind)
						narration = "The top detected objects in the image are: "  + " ".join(ram_tag.split(" | "))
						embed = self.sent_embed_model.encode(narration) # shape= (384,)
						embed=torch.tensor(embed)
						embed=embed.unsqueeze(0) # shape= [1,384]
						ram_sent_seg_embeds=torch.concat((ram_sent_seg_embeds,embed),dim=0)
				
				#ram_sent_seg_embeds shape: [8,384]=[n_frame,384]
				#ram_sent_embeds shape: [n,8,384]=[n_observed_actions,n_frame,384]
				ram_sent_seg_embeds=ram_sent_seg_embeds.unsqueeze(0)
				ram_sent_embeds=torch.concat((ram_sent_embeds,ram_sent_seg_embeds),dim=0)

				# find video_frames in F*H*W*C format for each of the 4 input observed videos
				preproc_seg_frame = self.extract_frames(clip_path, frame_ids)
				#preproc_seg_frame_tensor = torch.from_numpy(preproc_seg_frame).unsqueeze(0)
				if isinstance(preproc_seg_frame, BatchFeature):
					# Convert BatchFeature to a tensor
					preproc_seg_frame = preproc_seg_frame['pixel_values']  # Adjust key as necessary
					preproc_seg_frame = torch.tensor(preproc_seg_frame).clone().detach()
				#print(preproc_seg_frame.shape) #torch.Size([8, 3, 224, 224])
				preproc_seg_frame=preproc_seg_frame.permute(1, 0, 2, 3)
				#print(preproc_seg_frame.shape) #torch.Size([3, 8, 224, 224])
				preproc_seg_frame=preproc_seg_frame.unsqueeze(0)
				input_frames=torch.concat((input_frames,preproc_seg_frame) ,dim=0)# n*F*H*W*C
			#print(input_frames.shape)#torch.Size([4, 3, 8, 224, 224])
			
			
			# info from forecast segment 
			#seg_vid_prompt="The camera wearer will perform the following {} actions: ".format(len(forecast_seg_info_list))
			seg_vid_prompt="Answer:"
			for count, seg_info in enumerate(forecast_seg_info_list):
				# Repeat for the final ground-truth narration.
				seg_gt_noun=seg_info["noun"].split("_")[0]
				seg_gt_verb=seg_info["verb"].split("_")[0]

				#seg_vid_prompt = seg_vid_prompt+"\"{}". "{} {}\", where the verb is \"{}\" and the noun is \"{}\",".format(count+1,seg_gt_verb, 
										#seg_gt_noun, seg_gt_verb, seg_gt_noun)
				#seg_vid_prompt = seg_vid_prompt + "{}. \"{} {}\", where the verb is \"{}\" and the noun is \"{}\",".format(count+1, seg_gt_verb, seg_gt_noun, seg_gt_verb, seg_gt_noun)
				seg_vid_prompt = seg_vid_prompt + "\"{} {}\",".format(seg_gt_verb, seg_gt_noun)

			# Generate tokenized labels and ids from the prompts
			vid_prompt=seg_vid_prompt.strip(",")+"."
			##print("vid_prompt",vid_prompt)
			'''vid_prompt="Ans: "put lid", "put wheel",............., "tighten screw"." '''

			tokenized_inputs = generate_input_ids_and_labels(
					tokenizer=self.processor.tokenizer,
					prompt=PROMPTS[0],
					text=vid_prompt,
					decoder_only_lm=self.decoder_only_lm
					)
			#print(tokenized_inputs["input_ids"].shape) #torch.Size([449])
			#print(tokenized_inputs["labels"].shape) #torch.Size([449])
			
			

			
			out_dict = {}

			if self.sample_for_test:
				out_dict['seg_file'] = clip_path
				out_dict['prompt'] = PROMPTS[0]
				vid_name = clip_uid + ".mp4"
				out_dict['frames']=torch.tensor([])
				for frame in frame_ids:
					pixels= video_loader_by_frames(
						root=self.vids_dir,
						vid=vid_name,
						frame_ids=frame_id
					)
					pixels=pixels.unsqueeze(0)
					out_dict['frames']=torch.concat((out_dict['frames'],pixels),dim=0)
			else:
				# (num_tokens)
				out_dict['input_ids'] = tokenized_inputs['input_ids']
				out_dict['labels'] = tokenized_inputs['labels']
				# input_frames #(4,3, 8, 224, 224)
				out_dict['pixel_values'] = input_frames

			out_dict['ram_sent_embeds'] = ram_sent_embeds

		#print("input frame size",input_frames.shape)
		#input_frames=input_frames.unsqueeze(0)# 4*F*H*W*C

		#print(out_dict['input_ids'].shape) #torch.Size([451])
		#print(out_dict['labels'].shape) #torch.Size([451])
		#print(out_dict['pixel_values'].shape) #torch.Size([4, 3, 8, 224, 224])
		#print(out_dict['ram_sent_embeds'].shape) #torch.Size([4, 8, 384])
		
		return out_dict




'''

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_present_video',default=4)
	parser.add_argument('--n_future_video',default=20)
	parser.add_argument('--ego4d_annot_path_train',default="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json")
	parser.add_argument('--ego4d_annot_path_test',default="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_val.json")
	parser.add_argument('--ego4d_clip_path',default="/data/AmitRoyChowdhury/ego4d_data/v2/clips/")
	parser.add_argument('--forecast_annot_dir',default="/data/AmitRoyChowdhury/ego4d_data/v2/annotations")
	parser.add_argument('--no_frame',default=8)
	parser.add_argument('--ram_tags_path',default="/data/AmitRoyChowdhury/Anirudh/ram_openset_tags_ego4d/")
	processor_path="Salesforce/blip2-opt-2.7b"

	processor = Blip2Processor.from_pretrained(processor_path) ###############################
	args = parser.parse_args()

	annot_train = args.ego4d_annot_path_train
	annot_test = args.ego4d_annot_path_test
	n_future_actions = args.n_future_video
	n_present_actions = args.n_present_video
	clip_dir=args.ego4d_clip_path
	out_dir=args.forecast_annot_dir
	n_frame=args.no_frame
	ram_tags_path=args.ram_tags_path

	saved_tr_anot_file=out_dir+"/"+"anticipation_"+"train" + "_annots.json"
	saved_te_anot_file=out_dir+"/"+"anticipation_"+"test" + "_annots.json"

	dataset=SegFileDataset(saved_tr_anot_file,ram_tags_path,n_frame,processor)
	#print(len(dataset))

	data_collate=DataCollatorForVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of=8)

	# Create a DataLoader
	dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=data_collate)


	for i,out_dict in enumerate(dataloader):
		print("*****************{}th batch*********************".format(i))
		print("input_id: ",out_dict["input_ids"].shape)
		print("labels:",out_dict["labels"].shape)
		print("pixel_values:",out_dict["pixel_values"].shape)
		print("ram_sent_embeds:",out_dict["ram_sent_embeds"].shape)

	
	#*****************0th batch*********************
	#input_id:  torch.Size([2, 456])
	#labels: torch.Size([2, 456])
	#pixel_values: torch.Size([2, 4, 3, 8, 224, 224])
	#ram_sent_embeds: torch.Size([2, 4, 8, 384])
	#*****************1th batch*********************
	##input_id:  torch.Size([2, 448])
	labels: torch.Size([2, 448])
	#pixel_values: torch.Size([2, 4, 3, 8, 224, 224])
	#ram_sent_embeds: torch.Size([2, 4, 8, 384])
	#*****************2th batch*********************
	#input_id:  torch.Size([2, 448])
	#labels: torch.Size([2, 448])
	#pixel_values: torch.Size([2, 4, 3, 8, 224, 224])
	#ram_sent_embeds: torch.Size([2, 4, 8, 384])




if __name__ == "__main__":
	main()'''




