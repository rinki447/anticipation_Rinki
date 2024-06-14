import argparse
import logging
from pathlib import Path
import pickle
import sys

import torch
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import Blip2Processor

from eilev.data.utils import generate_input_ids_and_labels_from_interleaved
from eilev.model.utils import process
from models import FramesBlipForConditionalGeneration


def generate(
    model: FramesBlipForConditionalGeneration,
    processor: Blip2Processor,
    sample_path: str
) -> None:
    
    # Load processed sample data.
    with open(sample_path, "rb") as f:
        inputs = pickle.load(f)

    # (num_frames, C, H, W) : (4, 3, 224, 224)
    pixel_values = inputs['pixel_values'].to(model.device)

    # sys.exit()
    # process the inputs
    generate_kwargs = {
        # (num_frames, C, H, W): ( 4, 3, 224, 224)
        "pixel_values": pixel_values, 
        "input_ids": inputs["input_ids"].unsqueeze(0).to(model.device), # (1, num_tokens)
        # (1, num_tokens): (1, 148)
        "video_input_mask": inputs["video_input_mask"].unsqueeze(0).to(model.device), 
        "max_new_tokens": 32,
        "num_beams": 5,
        "do_sample": False,
        "length_penalty": -1,
    }
    if model.config.text_config.architectures[0] == "OPTForCausalLM":
        # if the LLM is OPT, set eos_token_id to the newline character as this is the
        # setting used by BLIP-2.
        # https://github.com/salesforce/LAVIS/blob/7f00a0891b2890843f61c002a8e9532a40343648/lavis/models/blip2_models/blip2_opt.py#L91-L93
        generate_kwargs["eos_token_id"] = 50118

    generated_ids = model.generate(**generate_kwargs)  # type: ignore
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    print(f"Generated_text: {generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate action narrations using an Blipformer-trained model."
    )
    parser.add_argument("--sample_path", required=True, 
                        help="Path to file containing validation sample.")
    parser.add_argument("--model", default="kpyu/eilev-blip2-opt-2.7b")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open("blipformer_args.pkl", "wb") as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    model = FramesBlipForConditionalGeneration.from_pretrained(args.model, 
                torch_dtype=torch.bfloat16).to(args.device)
    
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    #    videos_and_texts: ['demo/examples/dough-mixer.mp4',
    #  'Question: What is the camera wearer doing?',
    #  'Answer: The camera wearer hits the scraper in his right hand on the dough mixer guard.',
    #  'demo/examples/paint.mp4',
    #  'Question: What is the camera wearer doing?',
    #  'Answer: The camera wearer paints the wall in the room with the paint brush.',
    #  'demo/examples/trowel.mp4',
    #  'Question: What is the camera wearer doing?',
    #  'Answer:']
        
    # model = kpyu/eilev-blip2-opt-2.7b
    # processor = None

    generate(model, processor, args.sample_path)
