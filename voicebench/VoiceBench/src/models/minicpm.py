from .base import VoiceAssistant
from transformers import AutoModel, AutoTokenizer
import transformers
import torch
import soundfile as sf


class MiniCPMAssistant(VoiceAssistant):
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=False
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

        self.sys_prompt = model.get_sys_prompt(mode='audio_assistant', language='en')

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ): 
        user_question = {'role': 'user', 'content': [audio['array']]}
        msgs = [sys_prompt, user_question]
        
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=2048,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
        )

        return res
