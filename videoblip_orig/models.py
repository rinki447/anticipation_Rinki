from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import sys
import math

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2Config,
    Blip2QFormerModel,
    Blip2VisionModel,
    Blip2PreTrainedModel
)

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGenerationModelOutput,
)


class VideoBlipVisionModel(Blip2VisionModel):
    """A simple, augmented version of Blip2VisionModel to handle videos."""

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        """Flatten `pixel_values` along the batch and time dimension, pass it
        through the original vision model, then unflatten it back.

        :param pixel_values: a tensor of shape (batch, channel, time, height, width)

        :returns:
            last_hidden_state: a tensor of shape (batch, time * seq_len, hidden_size)
            pooler_output: a tensor of shape (batch, time, hidden_size)
            hidden_states:
                a tuple of tensors of shape (batch, time * seq_len, hidden_size),
                one for the output of the embeddings + one for each layer
            attentions:
                a tuple of tensors of shape (batch, time, num_heads, seq_len, seq_len),
                one for each layer
        """
        print("output_attentions:",output_attentions)
        print("output_hidden_states:",output_hidden_states)
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch,obs_actions,_, time, _, _ = pixel_values.size() #changed by Rinki

        # flatten along the batch and time dimension to create a tensor of shape
        # (batch * time, channel, height, width)
        flat_pixel_values = pixel_values.permute(0, 1, 3, 2 ,4,5).flatten(end_dim=2) #(end_dim=1) #changed by Rinki

        vision_outputs: BaseModelOutputWithPooling = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        
        print("vision_outputs.last_hidden_state shape:",vision_outputs.last_hidden_state.shape) #([64, 257, 1408])
        print("vision_outputs.pooler_output shape:",vision_outputs.pooler_output.shape) #([64,1408])
        #print("vision_outputs.attentions shape:",vision_outputs.attentions.shape) #none
        #print("vision_outputs.hidden_states shape:",vision_outputs.hidden_states.shape) #none

        # now restore the original dimensions
        # vision_outputs.last_hidden_state is of shape
        # (batch * obs_action*time, seq_len, hidden_size)
        seq_len = vision_outputs.last_hidden_state.size(1)
        last_hidden_state = vision_outputs.last_hidden_state.view(
            batch, obs_actions*time * seq_len, -1
        ) #changed by Rinki
        print("last_hidden_state modified shape:", last_hidden_state.shape)#([2, 8224, 1408])
        # vision_outputs.pooler_output is of shape
        # (batch * obs_action*time, hidden_size)
        pooler_output = vision_outputs.pooler_output.view(batch, obs_actions*time, -1) #changed by Rinki
        print("pooler_output modified shape:", pooler_output.shape)#([2, 32, 1408])


        ############################################################### not returned ###########################
        # hidden_states is a tuple of tensors of shape
        # (batch * obs_action*time, seq_len, hidden_size)
        hidden_states = (
            tuple(
                hidden.view(batch, obs_actions*time * seq_len, -1)
                for hidden in vision_outputs.hidden_states
            )
            if vision_outputs.hidden_states is not None
            else None
        )#changed by Rinki
        #print("hidden_states modified shape:", hidden_states.shape) #None
        
        # attentions is a tuple of tensors of shape
        # (batch * obs_action* time, num_heads, seq_len, seq_len)
        attentions = (
            tuple(
                hidden.view(batch, obs_action, time, -1, seq_len, seq_len)
                for hidden in vision_outputs.attentions
            )
            if vision_outputs.attentions is not None
            else None
        )
        #print("attentions modified shape:", attentions.shape) #None
        
        ##############################################################################################

        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        return (last_hidden_state, pooler_output, hidden_states, attentions)

class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map


        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility


    def forward(
        self,
        pixel_values: torch.FloatTensor, # (B, C, F, H, W) eg. (32, 3, 8, 224, 224)
        input_ids: torch.FloatTensor, # (B, num_tokens) eg. (32, 40)
        attention_mask: Optional[torch.LongTensor] = None, # (B, num_tokens) eg. (32, 40) 
        decoder_input_ids: Optional[torch.LongTensor] = None, # None
        decoder_attention_mask: Optional[torch.LongTensor] = None, # None
        output_attentions: Optional[bool] = None, # None
        output_hidden_states: Optional[bool] = None, # None
        labels: Optional[torch.LongTensor] = None, # (B, num_tokens) eg. (32, 40)
        return_dict: Optional[bool] = None, # None
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        Prepare processor, model and image input

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        ... )  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ```

        Image captioning (without providing a text prompt):

        ```python
        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two cats laying on a couch
        ```

        Visual question answering (prompt = question):

        ```python
        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```

        Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
        This greatly reduces the amount of memory used by the model while maintaining the same performance.

        ```python
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16
        ... )  # doctest: +IGNORE_RESULT

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```"""

        #print("input id: ",input_ids)
        #print("input id shape: ",input_ids.shape)

        # True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)

        # dict of keys:
        #   last_hidden_state: (B, obs_actions*num_patches*F, dim) eg. (2, [4*8*257], 1408)
        #   pooler_output: (B, obs_actions,F, dim) eg. (2,4, 8, 1408)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

         # (B, obs_actions*num_patches*F, dim) eg. (2,[4*257*8], 1408)
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        # (B, obs_action*num_patches*F) eg. (2, [4*257*8])
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], 
            dtype=torch.long, 
            device=image_embeds.device
        )
        
        # (B, num_query_tokens, dim) eg. (2, 32, 768)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # dict of keys:
        #   last_hidden_state: (B, num_query_tokens, dim) eg. (2, 32, 768)
        #   pooler_output: (B, dim) eg. (2, 768)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # (B, num_query_tokens, dim) eg. (2, 32, 768)
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        # (B, num_query_tokens, llm_dim) eg. (2, 32, 2560)
        language_model_inputs = self.language_projection(query_output)

        # (B, num_query_tokens) eg. (2, 32)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], 
            dtype=torch.long, 
            device=language_model_inputs.device
        )

        # (B, num_tokens, llm_dim) eg. (2, 40, 2560)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # (B, [num_query_tokens + num_tokens], llm_dim) eg. (2, 72[32 + 40], 2560)
        inputs_embeds = torch.cat([
            language_model_inputs, 
            inputs_embeds.to(language_model_inputs.device)
        ], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        expected_device = language_model_attention_mask.device

        # (B, [num_query_tokens + num_tokens]) eg. (32, 72[32 + 40])
        attention_mask = torch.cat([
            language_model_attention_mask, 
            attention_mask.to(expected_device)
        ], 
        dim=1)

        if self.config.use_decoder_only_language_model:

            # dict containing keys:
            #   logits: (B, num_tokens, llm_vocab_size) eg. (B, 72, 50272)
            #   past_key_values: tuple containing B past values.
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # (B, num_tokens, llm_vocab_size) eg. (B, 72, 50272)
            logits = outputs.logits if return_dict else outputs[0]
            loss = None

            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)

                # Not using the query_tokens for loss computation.
                # (B, num_tokens, llm_vocab_size) eg. (B, 48, 50272)
                logits = logits[:, -labels.size(1) :, :]

                # Shift so that tokens < n predict n
                # (B, num_tokens - 1, llm_vocab_size) eg. (32, 47, 50272)
                shift_logits = logits[..., :-1, :].contiguous()

                # (B, num_tokens - 1) eg. (32, 47)
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size), 
                    shift_labels.view(-1)
                )

        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output
        
        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor, # (B, C, F, H, W) eg. (1, 3, 8, 224, 224)
        input_ids: Optional[torch.LongTensor] = None, # (B, num_tokens) eg. (1, 9) 
        attention_mask: Optional[torch.LongTensor] = None, # (B, num_tokens) eg. (1, 9)
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]

        # (B, num_patches*F, dim) eg. (1, 2056[257*8], 1408)
        image_embeds = self.vision_model(pixel_values, return_dict=True)[0]

        # (B, num_patches*F) eg. (1, 2056[257*8])
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], 
            dtype=torch.long, 
            device=image_embeds.device
        )
        
        # (B, num_query_tokens, query_dim)
        # eg. (1, 32, 768)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )

        # (B, num_query_tokens, dim) 
        # eg. (1, 32, 768)
        query_output = query_outputs[0]
        #print("query output shape: ",query_output.shape)

        # (B, num_query_tokens, llm_dim) 
        # eg. (1, 32, 2560)
        language_model_inputs = self.language_projection(query_output)
        #print("language model inputs shape: ",language_model_inputs.shape)
        # (B, num_query_tokens) 
        # eg. (1, 32)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], 
            dtype=torch.long, 
            device=language_model_inputs.device
        )
        #print("language input attention mask shape: ",language_attention_mask.shape)

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        #print("input id attention mask shape",attention_mask.shape)

        # (B, num_input_tokens + num_query_tokens)
        # eg. (1, 41[32 + 9])
        attention_mask = torch.cat([
            language_attention_mask, 
            attention_mask.to(language_attention_mask.device)
        ], dim=1)

        # concatenate query embeddings with prompt embeddings
        # (B, num_tokens, llm_dim) eg. (1, 9, 2560)
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # (B, [num_query_tokens + num_tokens], llm_dim)
        # eg. (1, 41[32 + 9], 2560)
        inputs_embeds = torch.cat([
            language_model_inputs, 
            inputs_embeds.to(language_model_inputs.device)
        ], dim=1)

        # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
        # -1 is to account for the prepended BOS after `generate.`
        # TODO (joao, raushan): refactor `generate` to avoid these operations with VLMs
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]
        
        # (B, num_gen_tokens)
        # eg. (1, 46)
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_return_sequences=5,
            **generate_kwargs,
        )

        no_seq=outputs.shape[0]

        print("outputs shape",outputs.shape)

        # this is a temporary workaround to be consistent with other generation models and
        # have BOS as the first token, even though under the hood we are calling LM with embeds
        if not self.language_model.config.is_encoder_decoder:
            bos_tokens = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(no_seq, 1)
                .to(image_embeds.device)
            )

            print("bos token shape:",bos_tokens.shape)
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:

                outputs = torch.cat([bos_tokens, outputs], dim=-1)
        return outputs


class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config: Blip2Config) -> None:
        # HACK: we call the grandparent super().__init__() to bypass
        # Blip2ForConditionalGeneration.__init__() so we can replace
        # self.vision_model
        super(Blip2ForConditionalGeneration, self).__init__(config)

        self.vision_model = VideoBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()
