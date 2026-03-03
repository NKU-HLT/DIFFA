from .base import VoiceAssistant
from transformers import AutoModel
import torch


class DiVAAssistant(VoiceAssistant):
    def __init__(self):
        self.model = AutoModel.from_pretrained("WillHeld/DiVA-llama-3-v0-8b", cache_dir='./cache', trust_remote_code=True)

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        assert audio['sampling_rate'] == 16000
        audio = audio['array']
        return self.model.generate([audio], max_new_tokens=max_new_tokens)[0]

    def generate_text(
        self,
        text,
    ):
        return generate_text(self.model, [text], max_new_tokens=2048)[0]


@torch.no_grad()
def generate_text(
    llm,
    text_prompt,
    do_sample=False,
    logits_processor=None,
    max_new_tokens=1024,
):
    bsz = 1
    user_prompt_text = torch.tensor(
        llm.tokenizer(
            text_prompt,
            add_special_tokens=False,
            padding=True,
            padding_side="right",
        )["input_ids"],
        device=llm.pre_system.device,
    )
    prefix = torch.cat(
        [
            llm.pre_system.expand(
                bsz,
                -1,
            ),
            user_prompt_text,
            llm.post_system.expand(
                bsz,
                -1,
            ),
        ],
        axis=1,
    )

    prefix_embed = llm.llm_decoder.model.embed_tokens(prefix).expand(bsz, -1, -1)
    suffix = llm.final_header
    suffix_embed = llm.llm_decoder.model.embed_tokens(suffix).expand(bsz, -1, -1)
    inputs_embeds = torch.cat([prefix_embed, suffix_embed], axis=1)
    outs = [[] for i in range(bsz)]
    complete = [False] * bsz
    outputs = None
    greedy = 1
    i = 0
    while not all(complete) and len(outs[0]) < max_new_tokens:
        past_key_values = outputs.past_key_values if outputs else None
        outputs = llm.llm_decoder(
            inputs_embeds=inputs_embeds.to(
                llm.llm_decoder.model.embed_tokens.weight.device
            ).half(),
            return_dict=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        next_token_logits = outputs.logits[:, -1, :]

        if logits_processor:
            local_outs = torch.tensor(outs) if outs != [] else suffix
            local_outs = local_outs.reshape(1, -1)
            next_token_logits = logits_processor(
                local_outs,
                next_token_logits.reshape(1, -1),
            )
            next_token_logits = next_token_logits.flatten()
        if do_sample:
            logits = next_token_logits / temperature
            probs = F.softmax(logits, dim=-1)
            greedy = torch.multinomial(probs, num_samples=1)[0]
        else:
            greedy = next_token_logits.argmax(dim=-1)
        for token_index, out in enumerate(greedy.flatten().tolist()):
            if not complete[token_index]:
                outs[token_index].append(out)
            if out == 128009:
                complete[token_index] = True

        next_embed = llm.llm_decoder.model.embed_tokens(greedy.reshape(-1, 1))
        inputs_embeds = next_embed
    return llm.tokenizer.batch_decode(outs, skip_special_tokens=True)

