import argparse
import json
import copy
import collections
import os
import numpy as np 
def create_forecast_annotation(clip_dir,json_path,num_obs_actions,num_future_actions,out_dir,mode):
        
            with open(json_path, "r") as f:
                entries = json.load(f)['clips']
            
            
            # group annotations by source_clip_uid
            annotations = collections.defaultdict(list) #{"clip_uid1":["entry_seg1_info","entry_seg2_info"],  "clip_uid2":["entry_seg3_info","entry_seg4_info"]}
            for entry in entries:
                annotations[entry['clip_uid']].append(entry)

            # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
            annotations = {
                clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
                for clip_uid in annotations
            }

            untrimmed_clip_annotations = [] #  list of tuple;tuple[0]=video_path,tuple[1]=dict
            
            # each_dict:{"input_clips": ,"future_clips": } for each anticipation set: 4 current clips and 20 future
            # "input_clips":[entry["clips"][i],...,entry["clips"][i+n_i/p]],
            # "future_clips":[entry["clips"][i+n_i/p+1],...,entry["clips"][i+n_i/p+n_future]]}

            #for whole dataset
            for clip_uid, video_clips in annotations.items(): #video_clips=["entry_seg1_info","entry_seg2_info"]
                video_path = os.path.join(clip_dir, f'{clip_uid}.mp4')
                if len(video_clips) <= 0:
                    continue

                # Extract future annotations from video clips info.
                for i in range(
                    len(video_clips) - num_future_actions - num_obs_actions + 1
                ):
                    obs_clips = copy.deepcopy(video_clips[i : i + num_obs_actions])
                    forecast_clips = copy.deepcopy(
                        video_clips[
                            i
                            + num_obs_actions : i
                            + num_obs_actions
                            + num_future_actions
                        ]
                    )

                    untrimmed_clip_annotations.append((video_path,{"obs_clips": obs_clips, "forecast_clips": forecast_clips}))
       

            
            with open(out_dir+"/"+"anticipation2_"+mode + "_annots.json", "w") as f:
                json.dump(untrimmed_clip_annotations, f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_present_video',default=2)
    parser.add_argument('--n_future_video',default=20)
    parser.add_argument('--ego4d_annot_path_train',default="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json")
    parser.add_argument('--ego4d_annot_path_test',default="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_val.json")
    parser.add_argument('--ego4d_clip_path',default="/data/AmitRoyChowdhury/ego4d_data/v2/clips/")
    parser.add_argument('--forecast_annot_dir',default="/data/AmitRoyChowdhury/ego4d_data/v2/annotations")
    args = parser.parse_args()

    annot_train = args.ego4d_annot_path_train
    annot_test = args.ego4d_annot_path_test
    n_future_actions = args.n_future_video
    n_present_actions = args.n_present_video
    clip_dir=args.ego4d_clip_path
    out_dir=args.forecast_annot_dir

    '''uncomment below 2 lines for the first time generation of annotation files'''
    #create_forecast_annotation(clip_dir,annot_train,n_present_actions,n_future_actions,out_dir,"train")
    #create_forecast_annotation(clip_dir,annot_test,n_present_actions,n_future_actions,out_dir,"test")
    

    # check saved result #############################################################################
    train_annot=out_dir+"/"+"anticipation2_"+"train" + "_annots.json"
    with open(train_annot, "r") as f:
        saved_file=json.load(f)

    c=[]
    sum=0
    for j,k in saved_file:
        #j=j.split("/")[-1].split(".")[0] #video_path
        clip_info_list=k["obs_clips"]  #observed clips info
        #clip_info_list=k["forecast_clips"]  #forecast clips info
        l=len(clip_info_list)
        print(l)
        sum=sum+l
        c.append(l)
      
    c=np.array(c)
    #print(len(saved_file))
    print(sum)
    #############################################################################################3


if __name__ == "__main__":
    main()