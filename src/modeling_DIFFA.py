import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig, WhisperConfig, BertConfig

try:
    from .configuration_DIFFA import DIFFAConfig
    from .modeling_whisper_encoder import WhisperEncoder
except:
    from configuration_DIFFA import DIFFAConfig
    from modeling_whisper_encoder import WhisperEncoder


from src.subsample import WhisperProjector
from transformers.models.bert.modeling_bert import BertEncoder

import librosa
import os
from typing import List, Dict, Any
import numpy as np

from src.llm_generate import generate as llm_generate
from src.dataloader import build_chat_prompt


def tile_vector(vector, target_length):
    repeat_times = (target_length + len(vector) - 1) // len(vector)  
    expanded = vector.repeat(repeat_times)
    return expanded[:target_length]

class QformerConnector(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        

        if self.cfg.whisper_model_id == "openai/whisper-medium":
            self.target_layer_ids = [5, 11, 17, 23]
        elif self.cfg.whisper_model_id == "openai/whisper-small":
            self.target_layer_ids = [2, 5, 8, 11]
        elif self.cfg.whisper_model_id == "openai/whisper-tiny":
            self.target_layer_ids = [0,1,2,3]
        elif self.cfg.whisper_model_id == "openai/whisper-large-v3":
            self.target_layer_ids = [3, 7, 11, 15, 19, 23, 27, 31]
        else:
            raise NotImplementedError(f"model_id {self.cfg.whisper_model_id} not implemented")


        self.layer_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.cfg.prompt_size, self.cfg.whisper_config["d_model"])) for _ in range(len(self.target_layer_ids))]
        )
        
        # (prompt_size, target_layers)
        self.layer_weights = nn.Parameter(torch.zeros(self.cfg.prompt_size, len(self.target_layer_ids), dtype=torch.float))

        qformer_config = BertConfig()
        qformer_config.num_hidden_layers = 2
        qformer_config.num_attention_heads = self.cfg.whisper_config["encoder_attention_heads"]
        qformer_config.hidden_size = self.cfg.whisper_config["d_model"]
        qformer_config.add_cross_attention = True
        qformer_config.is_decoder = True

        self.qformer = BertEncoder(qformer_config)
        self.proj = nn.Sequential(
                nn.LayerNorm(self.cfg.whisper_config["d_model"]),
                nn.Linear(self.cfg.whisper_config["d_model"], self.cfg.llm_config["d_model"]) # project to llama hidden size
            )
    
    def forward(self, encoder_hidden_states):
        layer_prompt_outputs = []
        for idx, encoder_hidden_state in enumerate(encoder_hidden_states):
            if idx in self.target_layer_ids:
                layer_prompt = self.layer_prompts[self.target_layer_ids.index(idx)].expand(encoder_hidden_state.size(0), -1, -1)
                qformer_output = self.qformer(
                    hidden_states=layer_prompt,
                    encoder_hidden_states=encoder_hidden_state,
                )
                layer_prompt_output = qformer_output.last_hidden_state
                layer_prompt_outputs.append(layer_prompt_output)
        
        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3)
        
        self.norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1).unsqueeze(-1)
        
        output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_model)
        
        output = self.proj(output)
        
        return output

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class DIFFAModel(PreTrainedModel):
    config_class = DIFFAConfig
    base_model_prefix = "DIFFA"

    def __init__(self, config: DIFFAConfig, tokenizer=None, stage=2):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.llm_config = LlamaConfig(**config.llm_config)

        self.whisper_model = WhisperEncoder(self.whisper_config)
        self.llm_model = None

        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("/path/to/models/LLaDA-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        in_d = self.whisper_config.d_model
        out_d = self.llm_config.d_model

        self.trainable_prompts = torch.nn.Embedding(20, out_d) # 20 trainable embeddings as prompt
        self.semantic_connector = WhisperProjector(downsample_rate=4, idim=in_d, odim=out_d)

        self.acoustic_connector = QformerConnector(config)
        
        self.input_text_instruction="What can you hear from the audio?"
        self.stage = stage

        self.processed_labels = None
        self.processed_t = None

    def get_processed_infos(self):
        # discard
        return self.processed_labels, self.processed_t

    def forward(
        self,
        input_audio_features: torch.Tensor,  # [B, T, n_mels]
        input_audio_features_mask: torch.Tensor, 
        input_ids: torch.Tensor,
        audio_info_lengths: torch.Tensor,
        prompt_lengths: torch.Tensor,
        num_prompt_tokens: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
        # optional
        audio_ids=None,
        messages=None,
        target_text=None,
        input_text=None,
    ):
        
        ### 1. forward speech
        speech_embeds, speech_attention_mask = self.get_speech_features(input_audio_features, input_audio_features_mask, self.stage)

        inputs = self.insert_audio_embedding(input_ids, labels, t, audio_info_lengths,prompt_lengths, speech_embeds, speech_attention_mask)
        
        inputs_embeds = inputs["inputs_embeds"]
        attention_mask= inputs["attention_mask"]
        self.processed_labels = inputs["labels"]
        self.processed_t = inputs["t"]
        
        return self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=self.processed_labels,
        )

    def insert_audio_embedding(
        self,
        input_ids,
        labels,
        t,
        audio_info_lengths,
        prompt_lengths,
        speech_embeds: torch.Tensor,
        speech_attention_mask: torch.Tensor,
    ):
        batch_size = speech_embeds.size(0)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_labels = []
        batch_t = []
        for i in range(batch_size):        
            speech_embed = speech_embeds[i].unsqueeze(0)  # [1, speech_len, hidden]
            
            
            text_embeds = self.llm_model.get_input_embeddings()(input_ids[i].unsqueeze(0))
            prefix_embeds = text_embeds[:, :audio_info_lengths[i]]
            suffix_embeds = text_embeds[:, audio_info_lengths[i]:]
            combined_embeds = torch.cat([prefix_embeds, speech_embed, suffix_embeds], dim=1) # 1 * L
            

            
            padding = torch.full((speech_embed.size(1),), -100, dtype=torch.long, device=labels.device)
            label = labels[i] 
            label =  torch.cat([label[:audio_info_lengths[i]], padding, label[audio_info_lengths[i]:]], dim=0)
            temp_t = tile_vector(t[i], label.size(0))

            
            batch_inputs_embeds.append(combined_embeds)
            batch_labels.append(label.unsqueeze(0))  
            batch_t.append(temp_t.unsqueeze(0))
        
        inputs_embeds = torch.cat(batch_inputs_embeds, dim=0)
        # attention_mask = torch.cat(batch_attention_mask, dim=0)
        labels = torch.cat(batch_labels, dim=0).to(self.llm_model.device)
        t = torch.cat( batch_t, dim=0).to(self.llm_model.device)
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "labels": labels,
            "t": t,
        }

    
    def get_speech_features(self, speech_values, speech_attention_mask, stage=1):

        w2v_args = {
            "input_features": speech_values, #1*80*3000
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args) 
        speech_embeds = output.last_hidden_state # B x T x C
        all_hidden_states = output.hidden_states
        speech_lengths = output.output_lengths # real length of audios

        if stage == 1:
            # compute length and mask
            speech_semantic_embeds = self.semantic_connector(speech_embeds) # B x L x odim, ([8, 103, 4096])
            speech_semantic_embeds = self.prompt_wrap(speech_semantic_embeds) # B x ( L + learnable_prompt_length) x odim, torch.Size([8, 123, 4096])
            
            # speech_embeds = speech_embeds.transpose(0,1) # T x B x C -> B x T x C
            speech_embeds = speech_semantic_embeds
            speech_padding_mask = lengths_to_padding_mask(speech_lengths // 4 + 20)
            speech_atts = ~speech_padding_mask
            

        elif stage == 2:
            speech_acoustic_embeds = self.acoustic_connector(all_hidden_states) # B x L x odim , ([8, 64, 4096])

            # compute length and mask
            speech_semantic_embeds = self.semantic_connector(speech_embeds) # B x L x odim, ([8, 103, 4096])
            
            speech_padding_mask = lengths_to_padding_mask(speech_lengths // 4 + 64) # 64 is the prompt size of acoustic connector
            speech_atts = ~speech_padding_mask
            
            # concat two embeddings
            speech_embeds = torch.concatenate([speech_acoustic_embeds, speech_semantic_embeds] , dim=1) #B x L_all x odim
    
        return speech_embeds, speech_atts

    
    def prompt_wrap(self, speech_embeds):
        batch_size = speech_embeds.size(0)
        prompt_ids = torch.tensor(list(range(20)), dtype=torch.int64, device=speech_embeds.device)
        prompt_ids = prompt_ids.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        prompt_embeds = self.trainable_prompts(prompt_ids)

        wrapped_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        return wrapped_embeds
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None,**kwargs):
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config, **kwargs)

        if os.path.isdir(pretrained_model_name_or_path):
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")),strict=False
            )
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

        else:
            raise NotImplementedError

        return model

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        input_audio_features: torch.Tensor,
        input_audio_mask: torch.Tensor,
        system_prompt : str,
        input_texts: List[str], #prompt
        # Parameters from the reference script
        steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        mask_id: int = 126336,
        # Original parameters
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        # Use max_new_tokens as gen_length
        gen_length = max_new_tokens
        
        # 0. Setup
        device = self.llm_model.device

        # 1. Prepare prompt embeddings (audio + text)
        speech_embeds, _ = self.get_speech_features(
            input_audio_features,
            input_audio_mask,
            stage=self.stage
        )

        # 2.prepare text embed
        audio_info_lengths = []
        prompt_lengths = []
        input_ids = []

        batch = speech_embeds.size(0)
        for i in range(batch):
            audio_info = [ 
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "This is the audio: "},
                ]
            prompt_info = [ 
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "This is the audio: <audio>. " + input_texts[i]},
                ]

            audio_info_text = build_chat_prompt(audio_info,  add_generation_prompt=False)
            audio_info_text = audio_info_text[:-10] # remove <|eot_id|>
            prompt_text = build_chat_prompt(prompt_info, add_generation_prompt=True)

            # Tokenize
            tokenized_audio_info = tokenizer(audio_info_text, return_tensors="pt", max_length=4096, truncation=True)
            tokenized_prompt = tokenizer(prompt_text, return_tensors="pt", max_length=4096, truncation=True)

            # get length
            audio_info_lengths.append(tokenized_audio_info.attention_mask.sum(-1) - 1)
            prompt_lengths.append(tokenized_prompt.attention_mask.sum(-1))
            input_ids.append(tokenized_prompt.input_ids)
        audio_info_lengths = torch.cat(audio_info_lengths,dim=0)
        prompt_lengths = torch.cat(prompt_lengths,dim=0)
        input_ids = torch.cat(input_ids,dim=0) 

        out = llm_generate(self, input_ids, audio_info_lengths=audio_info_lengths, speech_embeds=speech_embeds, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id)
        res = self.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        return res