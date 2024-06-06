import os
from random import choice
import json
import pickle
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms._transforms_video as transforms_video
from utils import get_gt_caption, get_frame_ids, get_seg_start_end_frame, \
    extract_frames_from_video,video_loader_by_frames, Permute
from utils import get_gt_caption,extract_frames_from_video,generate_input_ids_and_labels,video_loader_by_frames
from utils import generate_input_ids_and_labels

from torch.utils.data import Dataset
# from eilev.data.utils import (
#     generate_input_ids_and_labels_from_interleaved, 
#     clean_narration_text
# )
from utils import (
    generate_input_ids_and_labels_from_interleaved,
    clean_narration_text
)
from transformers import BatchFeature
from sentence_transformers import (
    SentenceTransformer, 
    util
)
# Based on prompts from InstructBLIP
# Modified for image
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

# VID_PROMPTS= [
#     "Given this temporal frame-level information, what action is being performed in the video?"
# ]

#VID_PROMPTS=["You are an assistant which models human behaviour very well. You'll be provided with a sequence of images retrieved from a first-person view video and the top detected objects. Your task is to understand the overall action and describe it in one sentence."]

class SegFileDataset(Dataset):
    def __init__(self,annots_path,processor,num_query_tokens=32,decoder_only_lm=True,sample_for_test=False):

        
        with open(annots_path, "r") as f:
            self.annots = json.load(f)
        #self.annots=self.annots[:34]
        
        self.processor=processor 
        self.num_query_tokens = num_query_tokens
        self.decoder_only_lm = decoder_only_lm
        self.sample_for_test = sample_for_test
        

    

    def __len__(self):
        return len(self.annots)   

   
            
    def extract_frames(self, clip_path,num_segments,start_frame,end_frame):
        """Given the seg_file, extract the frame-level features

           Args:
            seg_file (str): Ego4D seg_file of the format clip_id_start_frame_XX_end_frame_YY

           Returns:
            inputs (dict): preprocessed subset of frames from the video between the 
                start_frame=XX and end_frame=YY. 
        """
        #vid_path = seg_file.split("_st")[0] + ".mp4"

        #start_frame, end_frame = get_seg_start_end_frame(seg_file)

        frame_ids = get_frame_ids(start_frame, end_frame, num_segments=num_segments, jitter=False)

        # (num_frames, H, C, W)
        # eg. (4, 1080, 1440, 3)
        pil_frames = extract_frames_from_video(clip_path,frame_ids)

        # 

        # Preprocess frames
        # dict['pixel_values': (num_frames, C, H, W): (4, 3, 224, 224)]
        inputs = self.processor(pil_frames, return_tensors='pt')
                
        return inputs    

    def __getitem__(self, idx):


        clip_path,set_dict = self.annots[idx]  #### more steps
        clip_uid=clip_path.split("/")[-1].split(".")[0]
        observed_seg_info_list=set_dict["obs_clips"]
        forecast_seg_info_list=set_dict["forecast_clips"]


        input_frames=torch.tensor([])
        tokenized_input=[]
        #ram_sent_embeds=torch.tensor([])

        with torch.no_grad():

            # info from observed segments
            for seg_info in observed_seg_info_list:

                # find ram_frame_no and ram tags caption for each of the 4 observed input videos
                start_frame=seg_info["action_clip_start_frame"]
                end_frame=seg_info["action_clip_end_frame"]
                seg_name = "{}_start_frame_{}_end_frame_{}".format(clip_uid, start_frame, end_frame)

                n_seg=8
                
                preproc_seg_frame = self.extract_frames(clip_path,n_seg,start_frame,end_frame)

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

            out_dict['input_ids'] = tokenized_inputs['input_ids']
            out_dict['labels'] = tokenized_inputs['labels']

            # dict['pixel_values': (num_frames, C, H, W) -> (C, num_frames, H, W): (3, 8, 224, 224)
            out_dict['pixel_values'] = input_frames

                

        #print("input frame size",input_frames.shape)
        #input_frames=input_frames.unsqueeze(0)# 4*F*H*W*C

        #print(out_dict['input_ids'].shape) #torch.Size([451])
        #print(out_dict['labels'].shape) #torch.Size([451])
        #print(out_dict['pixel_values'].shape) #torch.Size([4, 3, 8, 224, 224])
        #print(out_dict['ram_sent_embeds'].shape) #torch.Size([4, 8, 384])
        
        return out_dict