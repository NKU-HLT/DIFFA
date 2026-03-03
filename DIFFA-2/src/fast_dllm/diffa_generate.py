# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from src.fast_dllm.model.modeling_llada import LLaDAModelLM
from loguru import logger


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def insert_audio_embedding(text_embeds, speech_embeds, audio_info_lengths):
        batch_size = speech_embeds.size(0)
        batch_inputs_embeds = []
        
        # 3. 处理音频嵌入位置
        for i in range(batch_size):        
            # 假设audio_marker在system之后，所以插入位置为system_end
            speech_embed = speech_embeds[i].unsqueeze(0)  # [1, speech_len, hidden]
            # 分割文本嵌入并插入语音嵌入
            prefix_embeds = text_embeds[:, :audio_info_lengths[i]]
            suffix_embeds = text_embeds[:, audio_info_lengths[i]:]
            combined_embeds = torch.cat([prefix_embeds, speech_embed, suffix_embeds], dim=1) # 1 * L
            batch_inputs_embeds.append(combined_embeds)

        inputs_embeds = torch.cat(batch_inputs_embeds, dim=0)
        return inputs_embeds

def remove_audio_infos(logits, speech_embeds, audio_info_lengths):

    prefix_logits = logits[:,:audio_info_lengths]
    suffix_logits = logits[:, (audio_info_lengths+speech_embeds.size(1)):]
    logits = torch.cat([prefix_logits,suffix_logits],dim=1)

    return logits

@ torch.no_grad()
def generate(model, prompt, audio_info_lengths, speech_embeds, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # insert audio embedding, and get embed_x.
    x = torch.full((prompt.shape[0], prompt.shape[1]+gen_length), mask_id, dtype=torch.long).to(model.device)    
    x[:, :prompt.shape[1]] = prompt.clone()
    embed_x = model.llm_model.get_input_embeddings()(x)
    embed_x = insert_audio_embedding(embed_x, speech_embeds, audio_info_lengths).to(torch.bfloat16)

    # align length, get discrete x.
    x = torch.full((prompt.shape[0], prompt.shape[1]+speech_embeds.shape[1]+gen_length), mask_id, dtype=torch.long).to(model.device)    
    x[:, :prompt.shape[1]] = prompt.clone()
    # for debug, set pseudo audio tokens to 1.
    x[:,prompt.shape[1]: prompt.shape[1]+speech_embeds.shape[1]] = torch.full((x.shape[0], speech_embeds.shape[1]), 1, dtype=torch.long).to(model.device)
    
    # audio as part of prompt
    prompt_len = prompt.shape[1] + speech_embeds.shape[1]
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    eos_id=126081

    # pdb.set_trace()
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt_len + num_block * block_length: prompt_len + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0

        stop_flag=False

        while True:
            nfe += 1
            mask_index = (x == mask_id)
            # insert audio embeddings.
            logits = model.llm_model(inputs_embeds = embed_x.to(torch.bfloat16)).logits

            mask_index[:, prompt_len + (num_block + 1) * block_length:] = 0

            # factor
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            
            x[transfer_index] = x0[transfer_index]
            embed_x[transfer_index] = model.llm_model.get_input_embeddings()(x)[transfer_index]

        
            i += 1
            block_start = prompt_len + num_block * block_length
            block_end = prompt_len + (num_block + 1) * block_length
            curr_block=x[:, block_start: block_end]

            if (curr_block == eos_id).any():
                first_eos_idx = (curr_block == eos_id).nonzero(as_tuple=True)[1][0]
                   
                if (curr_block[:, :first_eos_idx] == mask_id).sum() == 0:
                    x[:, block_start + first_eos_idx + 1 : block_end] = eos_id
                    stop_flag = True
                    break

            if (curr_block == mask_id).sum() == 0:
                break
            
        if stop_flag:
            break
        
            
    # remove audio tokens
    x = torch.cat([x[:,:prompt.shape[1]], x[:,prompt_len:]],dim=1)
    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained("/path/to/models/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("/path/to/models/LLaDA-8B-Instruct", trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()