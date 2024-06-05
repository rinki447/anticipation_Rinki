import os, re, pickle
import string
import tqdm
import decord
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer

C_REGEX = re.compile(r"^\#C\s+C", re.IGNORECASE)
EOS_REGEX = re.compile(r"\<\|eos\|\>$", re.IGNORECASE)
UNSURE_END_REGEX = re.compile(r"#unsure\.?$", re.IGNORECASE)
UNSURE_MIDDLE_REGEX = re.compile(r"#unsure", re.IGNORECASE)

class DataCollatorForVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if all("pixel_values" in feature for feature in features):
            pixel_values = torch.stack(
                [feature.pop("pixel_values") for feature in features]
            )
        else:
            # in some cases, we don't have pixel values, e.g.,
            # in-context learning evaluation
            pixel_values = None
        collated = super().__call__(features, return_tensors=return_tensors)
        if pixel_values is not None:
            collated["pixel_values"] = pixel_values
        return collated


class DataCollatorForInterleavedVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.cat(
            [feature.pop("pixel_values") for feature in features]
            if "pixel_values" in features[0].keys()
            else None,
        )
        video_input_masks = (
            [feature.pop("video_input_mask") for feature in features]
            if "video_input_mask" in features[0].keys()
            else None
        )
        collated = super().__call__(features, return_tensors=return_tensors)
        if video_input_masks is not None:
            max_input_id_len = collated["input_ids"].size(1)
            padded_video_input_masks = []
            for video_input_mask in video_input_masks:
                remainder = torch.tensor(
                    [0] * (max_input_id_len - len(video_input_mask))
                )
                if self.tokenizer.padding_side == "right":
                    padded_video_input_masks.append(
                        torch.cat([video_input_mask, remainder])
                    )
                else:
                    padded_video_input_masks.append(
                        torch.cat([remainder, video_input_mask])
                    )
            collated["video_input_mask"] = torch.stack(padded_video_input_masks)
        if pixel_values is not None:
            collated["pixel_values"] = pixel_values
        return collated

def generate_input_ids_and_labels(
    tokenizer: PreTrainedTokenizer, prompt: str, text: str, decoder_only_lm: bool
) -> BatchEncoding:
    """Generate input ids and labels from the given prompt and text. If
    decoder_only_lm is True, the input and label texts are the same, but label
    tokens that correspond to the prompt are masked with -100. If
    decoder_only_lm is False, the input corresponds to the prompt and the label
    to the text.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompt: prompt for the LLM
    :param text: text for the LLM to generate based on the prompt
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results
    """
    if decoder_only_lm:
        # tokenize prompt first
        prompt_tokens = tokenizer(prompt, return_attention_mask=False).input_ids

        # tokenize the narration and append eos
        preprocessed = tokenizer(
            " " + text,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        preprocessed["input_ids"].append(tokenizer.eos_token_id)

        # join tokenized prompt and narration text
        preprocessed["input_ids"] = prompt_tokens + preprocessed["input_ids"]
        preprocessed["input_ids"] = torch.tensor(preprocessed.input_ids)

        # for decoder only LMs, labels are same as input_ids, but we mask
        # tokens for the prompt
        preprocessed["labels"] = preprocessed["input_ids"].clone()
        preprocessed["labels"][: len(prompt_tokens)] = -100
    else:
        # eos is automatically appended by the tokenizer
        # we don't use return_tensors='pt' here b/c it automatically batchifies things
        # which we don't want
        preprocessed = tokenizer(prompt, return_attention_mask=False)
        preprocessed["input_ids"] = torch.tensor(preprocessed["input_ids"])
        preprocessed["labels"] = torch.tensor(
            tokenizer(text, return_attention_mask=False).input_ids
        )

    return preprocessed


def generate_input_ids_and_labels_from_interleaved(
    tokenizer: PreTrainedTokenizer,
    prompts: list[tuple[str, int]],
    text: str | None,
    num_query_tokens: int,
    decoder_only_lm: bool,
) -> dict[str, torch.Tensor]:
    """Generate input ids and labels from the given interleaved video/text data
    point. `text_video_map` specifies which videos are the last preceding
    videos for a given text, and is used to generate `video_input_mask`.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompts: list of prompts, each with the number of videos
    :param text: optional text to be completed by LLM
    :param num_query_tokens: number of qformer query tokens
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results including `input_ids`, `labels` and
        `video_input_mask`.
        `input_ids` is a tensor of shape (num_tokens),
        `labels` is a tensor of shape (num_tokens),
        `video_input_mask` is a tensor of shape (num_tokens)
    """
    # with open("eilev_prompts_4_incontext.pkl", "wb") as f:
    #     pickle.dump(prompts, f, pickle.HIGHEST_PROTOCOL)

    # with open("eilev_text_4_incontext.pkl", "wb") as f:
    #     pickle.dump(text, f, pickle.HIGHEST_PROTOCOL)

    # sys.exit()

    input_ids: list[int] = []
    labels: list[int] = []
    video_input_mask: list[int] = []
    # NOTE: FLAN tokenizer treats all whitespaces the same
    newline_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    if decoder_only_lm:
        for i, (prompt, num_videos) in enumerate(prompts):
            # first take care of the video tokens
            for _ in range(num_videos):
                input_ids.extend(
                    [tokenizer.pad_token_id] * num_query_tokens + [newline_token_id]
                ) # 33 [= 32 + 1]
                labels.extend([-100] * (num_query_tokens + 1)) # 33 [= 32 + 1]
                video_input_mask.extend([1] * num_query_tokens + [0]) # 33 [= 32 + 1]
            if i == 0:
                # if first text, start with a bos token
                input_ids = [tokenizer.bos_token_id] + input_ids # 34
                labels = [-100] + labels # 34
                video_input_mask = [0] + video_input_mask # 34
            if i != len(prompts) - 1:
                # if not last prompt, add newline
                prompt += "\n"                

            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids # 36
            input_ids.extend(prompt_tokens) # 36
            video_input_mask.extend([0] * len(prompt_tokens))
            labels.extend([-100] * len(prompt_tokens))
        if text is not None:
            # prepend a space to separate the text from the prompt
            text_tokens = tokenizer(
                " " + text + "\n", add_special_tokens=False
            ).input_ids + [tokenizer.eos_token_id]
            input_ids.extend(text_tokens)
            video_input_mask.extend([0] * len(text_tokens))
            labels.extend(text_tokens)
    else:
        for i, (prompt, num_videos) in enumerate(prompts):
            # first take care of the video tokens
            for _ in range(num_videos):
                input_ids.extend(
                    [tokenizer.pad_token_id] * num_query_tokens + [newline_token_id]
                )
                video_input_mask.extend([1] * num_query_tokens + [0])
            if i != len(prompts) - 1:
                # if not last prompt, add newline
                prompt += "\n"
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            if i == len(prompts) - 1:
                # if last prompt, add eos token
                prompt_tokens.append(tokenizer.eos_token_id)
            input_ids.extend(prompt_tokens)
            video_input_mask.extend([0] * len(prompt_tokens))
        if text is not None:
            labels.extend(tokenizer(text).input_ids)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "video_input_mask": torch.tensor(video_input_mask),
    }

def clean_narration_text(narration_text: str) -> str:
    # strip it first
    cleaned = narration_text.strip()

    # replace "#C C" with "The camera wearer"
    cleaned = re.sub(C_REGEX, "The camera wearer", cleaned).strip()

    # remove <|eos|>
    cleaned = re.sub(EOS_REGEX, "", cleaned).strip()

    # remove #unsure from the end
    cleaned = re.sub(UNSURE_END_REGEX, "", cleaned).strip()

    # replace #unsure in the middle with "something"
    cleaned = re.sub(UNSURE_MIDDLE_REGEX, "something", cleaned)

    if len(cleaned) == 0:
        return cleaned

    # if cleaned doesn't end with a punctuation, append a period
    if not cleaned[-1] in string.punctuation:
        cleaned += "."

    return cleaned



class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)


def get_seg_start_end_frame(seg_name):
    """Given a segment in the format "clip_name_start_frame_XX_end_frame_YY", get the
        start_frame= XX and end_frame=YY respectively.
    """
    start_frame = re.search(r'start_frame_(\d+)', seg_name).group(1)
    end_frame = re.search(r'end_frame_(\d+)', seg_name).group(1)

    return int(start_frame), int(end_frame)

def extract_frames_from_video(clip_path,sampled_frame_id):
    """Extract frames for the respective frame_ids from the given vid_path .

       Args:
            frame_ids (list): frame numbers.
            vids_dir (list): Directory containing the videos.
            vid_path (str): name of the video
       
       Returns:
            pil_imgs: Corresponding images in PIL format
    """
    '''sampled_frame_id=[]
    for fr in range(start_frame,end_frame):
        sampled_frame_id.append(fr)
        fr=fr+n_frame'''

    #print("samled frames",sampled_frame_id)
    #print("clip_path",clip_path)
    vr = decord.VideoReader(clip_path)
    frames = vr.get_batch(sampled_frame_id).asnumpy() # (N x H x W x C)
    #print("frames ",frames.shape) #(8, 1080, 1440, 3)


    
    pil_imgs = []
    for ind in range(len(frames)):
        frame = frames[ind]
        pil_img = Image.fromarray(frame)
        pil_imgs.append(pil_img)
        #print("pil image size",pil_img.size) #(1440, 1080)

    #print("pil image list size",len(pil_imgs)) #8
    
    return pil_imgs


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq

def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        # frames = vr.get_batch(frame_ids).asnumpy()
        frames = vr.get_batch(frame_ids)
        if not isinstance(frames, np.ndarray):
            frames = frames.asnumpy()

        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


def convert_tensors_to_images(tensor_frames):
    """Given a torch tensor of images, convert to PIL format.    

       Args:
            tensor_frames: (N, H, W, C) - N frames in torch tensor format having (H, W, C) shape

    """
    pil_transform = T.ToPILImage()

    if tensor_frames.shape[1] > 4:
        tensor_frames = tensor_frames.permute(0, 3, 1, 2) # (N, C, H, W)

    pil_frames = []
    for frame in tensor_frames:
        pil_img = pil_transform(frame)
        pil_frames.append(pil_img)
    
    return pil_frames

def extract_verb_noun_label(prompt_name):
    """Given a prompt file "llama2_prompt_response_verb_label_XX_noun_label_YY.json", 
        extract the verb_label and noun_label using regex.
    """
    verb_label = re.search(r'verb_label_(\d+)', prompt_name).group(1)
    noun_label = re.search(r'noun_label_(\d+)', prompt_name).group(1)

    return int(verb_label), int(noun_label)


def get_gt_caption(annots, seg_file):
    """Given a segment in the format "clip_name_start_frame_XX_end_frame_YY", get the
        corresponding verb_label and noun_label respectively.
    """
    clip_id = seg_file.split("_st")[0]
    start_frame, end_frame = get_seg_start_end_frame(seg_file)

    for seg_annot in annots:
        seg_start_frame = seg_annot["action_clip_start_frame"]
        seg_end_frame = seg_annot["action_clip_end_frame"]
        seg_clip_id = seg_annot['clip_uid']

        if seg_clip_id == clip_id and seg_start_frame == start_frame and\
              seg_end_frame == end_frame:
            return seg_annot['verb'], seg_annot['noun']
        
def extract_synonyms(act_name):
    """Given an act_name in the format "grocery_("nylon,_sack,_suitcase)', extract 
        ["nylon", "sack", "suitcase"].
    """
    words = re.findall(r'\b\w+\b', re.findall(r'\((.*?)\)', act_name)[0].replace('_', ' '))

    return words

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    
def extract_validation_samples(val_dataset, save_dir):

    num_samples = len(val_dataset)

    for ind in tqdm.tqdm(range(num_samples)):
        
        sample_dict = val_dataset.__getitem__(ind)
        seg_file = sample_dict['seg_file']
        save_path = save_dir + seg_file

        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                pickle.dump(sample_dict, f, pickle.HIGHEST_PROTOCOL)