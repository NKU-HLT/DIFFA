import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


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
             cfg_scale=0., remasking='low_confidence', mask_id=126336, ):
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
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id #un_x 是全mask的
                x_ = torch.cat([x, un_x], dim=0)
                # insert audio embeddings.
                # for x
                embed_x = model.llm_model.get_input_embeddings()(x)
                embed_x = insert_audio_embedding(embed_x, speech_embeds, audio_info_lengths)
                # for un_x
                embed_un_x = model.llm_model.get_input_embeddings()(un_x)
                embed_un_x = insert_audio_embedding(embed_un_x, speech_embeds, audio_info_lengths)
                embed_x_ = torch.cat([embed_x,embed_un_x], dim=0)

                logits = model.llm_model(inputs_embeds = embed_x_.to(torch.bfloat16)).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # insert audio embeddings.
                embed_x = model.llm_model.get_input_embeddings()(x)
                embed_x = insert_audio_embedding(embed_x, speech_embeds, audio_info_lengths)
                logits = model.llm_model(inputs_embeds = embed_x.to(torch.bfloat16)).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # recover x0, remove audio embeddings
            logits = remove_audio_infos(logits, speech_embeds, audio_info_lengths)
            logits_with_noise = remove_audio_infos(logits_with_noise, speech_embeds, audio_info_lengths)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x