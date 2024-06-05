import pickle
import sys
from eilev.model.v2 import VideoBlipForConditionalGeneration, Blip2ForConditionalGeneration
import transformers

import torch

def main():
    with open("rand_sample.pkl", "rb") as f:
        blipformer_rand_sample = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"    

    blipformer_rand_sample['input_ids'] = blipformer_rand_sample['input_ids'].to(device)

    blipformer_rand_sample['labels'] = blipformer_rand_sample['labels'].to(device)

    blipformer_rand_sample['video_input_mask'] = blipformer_rand_sample['video_input_mask'].\
        to(device)
    
    blipformer_rand_sample['pixel_values'] = blipformer_rand_sample['pixel_values'].\
        squeeze().to(device)
    
    # blipformer_rand_sample['pixel_values'] = blipformer_rand_sample['pixel_values'].\
    #     squeeze().to(device)
    # image_pixel_values = blipformer_rand_sample['pixel_values'].squeeze().to(device)
    # image_pixel_values = torch.zeros(4, 3, 224, 224)

    # blipformer_rand_sample['pixel_values'] = torch.randn(1, 3, 8, 224, 224).to(device)

    pixel_values = blipformer_rand_sample['pixel_values']
    input_ids = blipformer_rand_sample['input_ids']
    labels = blipformer_rand_sample['labels']
    video_input_mask = blipformer_rand_sample['video_input_mask']

    print("Shape of pixel_values:{}".format(pixel_values.shape))
    print("Shape of input_ids:{}".format(input_ids.shape))
    print("Shape of labels:{}".format(labels.shape))
    print("Shape of video_input_mask:{}".format(video_input_mask.shape))

    output_attentions = output_hidden_states = None
    return_dict = True

    model_name_or_path = "Salesforce/blip2-opt-2.7b"
    
    video_blip_model = VideoBlipForConditionalGeneration.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.bfloat16).to(device)
    
    vid_processor = transformers.Blip2Processor.from_pretrained(
        model_name_or_path
    )
    image_blip_model = Blip2ForConditionalGeneration.from_pretrained(
        model_name_or_path, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)
    video_blip_model.config.text_config.eos_token_id = vid_processor.tokenizer.eos_token_id


    # freeze everything except for qformer
    for param in video_blip_model.vision_model.parameters():
        param.requires_grad = False
    for param in video_blip_model.language_model.parameters():
        param.requires_grad = False

    # with torch.no_grad():
    #     outs = video_blip_model.forward(**blipformer_rand_sample)

    if pixel_values is not None:
        vision_outputs = image_blip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        
        assert vision_outputs is not None
        # (num_videos, time * vision_seq_len, vision_hidden_size)
        # eg. For EILEV: (17, 2056, 1408), For BLIPFormer: (1, 1028, 1408)
        image_embeds = vision_outputs[0].to(torch.bfloat16)
        print("Shape of image_embeds:{}".format(image_embeds.shape))

        # step 2: forward the query tokens through the QFormer,
        # using the image embeddings for cross-attention
        # (num_videos, time * vision_seq_len)
        # eg. For EILEV: (17, 2056), For BLIPFormer: (1, 1028)
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        # (num_videos, num_query_tokens, qformer_hidden_size)
        # eg. For EILEV: (17, 32, 768), For BLIPFormer: (4, 32, 768)
        query_tokens = video_blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1).\
            to(torch.bfloat16)
        query_outputs = video_blip_model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # (num_videos, num_query_tokens, qformer_hidden_size)
        # eg. For EILEV: (17, 32, 768), For BLIPFormer: (4, 32, 768)
        query_output = query_outputs[0]
        # query_output = query_output.view(num_frames * image_blip_model.config.num_query_tokens, -1)
        print("Shape of query_output:{}".format(query_output.shape))

        # step 3: project the qformer tokens to the language model space
        # For EILEV: 17, For BLIPFormer: 1
        num_frames = 4
        query_output = query_output.view(num_frames * video_blip_model.config.num_query_tokens,
                                          -1).to(torch.bfloat16)
        # print("query_output dtype:{}".format(query_output.dtype))
        # print("language_projection dtype:{}".format(
        #     video_blip_model.language_projection.weight.dtype))

        # query_output = torch.randn(544, 768).to(device)
        
        # (num_videos * num_query_tokens, text_hidden_size)
        # For EILEV: (544[17*32], 2560), For BLIPFormer: (128, 2560)
        video_features = video_blip_model.language_projection(
            query_output
        )
        print("Shape of video_features:{}".format(video_features.shape))

    # (batch_size, seq_len, text_hidden_size)
    # eg. # For EILEV: (1, 1064, 2560), For BLIPFormer: (1, 286, 2560)
    inputs_embeds = video_blip_model.language_model.get_input_embeddings()(input_ids)
    print("Shape of inputs_embeds:{}".format(inputs_embeds.shape))

    # torch.save(inputs_embeds, "blipformer_inputs_embeds.pt")
    # torch.save(video_features, "blipformer_video_features.pt")
    # torch.save(video_input_mask, "blipformer_video_input_mask.pt")

    video_input_mask = video_input_mask > 0

    if video_features is not None:
        # we need to clone inputs_embeds first since it may require gradients
        # and index assignment is an inplace operation
        tmp_inputs_embeds = inputs_embeds.clone()
        tmp_inputs_embeds[video_input_mask] = video_features.to(
            # for mixed-precision training
            dtype=tmp_inputs_embeds.dtype
        )
        inputs_embeds = tmp_inputs_embeds
    
    attention_mask = None

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)  # type: ignore

    if video_blip_model.config.use_decoder_only_language_model:
        outputs = video_blip_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
    else:
        outputs = video_blip_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )


    # image_embeds -> (4, 257, 1408)3

    # image_attention_mask -> (4, 257)

    # query_tokens -> (4, 32, 768)

    # video_features -> (128, 768)

    # input_ids -> (1, 378, 2560)

    # video_input_mask -> (1, 378)

    print(outputs)
 
    # sys.exit()


if __name__=="__main__":
   main()