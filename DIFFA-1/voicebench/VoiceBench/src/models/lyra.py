from .base import VoiceAssistant
import torch

from .src_lyra.constants import DEFAULT_SPEECH_TOKEN
from .src_lyra.conversation import conv_templates, SeparatorStyle
from .src_lyra.model.builder import load_pretrained_model
from .src_lyra.utils import disable_torch_init
from .src_lyra.mm_utils import tokenizer_speech_token, get_model_name_from_path
import os
from huggingface_hub import snapshot_download


class LyraAssistant(VoiceAssistant):
    def load_model(self, model_path):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _, speech_processor = load_pretrained_model(model_path, None,
                                                                                       model_name, False,
                                                                                       False,
                                                                                       model_lora_path=None,
                                                                                       use_flash_attn=False,
                                                                                       device='cuda')

        model_lora_path = f'{model_path}/speech_lora'
        model.load_adapter(model_lora_path, adapter_name="speech")
        print(f"Loading Speech LoRA weights from {model_lora_path}")

        model.to(torch.float16)

        self.speech_processor = speech_processor
        self.tokenizer = tokenizer
        self.model = model
        model.set_adapter(["speech"])

        model_lora_path = f'{model_path}/speech_lora'
        mm_projector_weights = torch.load(os.path.join(model_lora_path, 'non_lora_trainables.bin'), map_location='cpu')
        mm_projector_weights = {k[6:]: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        print("load new projectors....", mm_projector_weights.keys())
        status = model.load_state_dict(mm_projector_weights, strict=False)
        self.model = self.model.cuda()

    def download_model(self):
        raise NotImplementedError

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        image_tensor = None
        wav = torch.Tensor(audio['array'])
        speech_tensor = []
        whipser_len = 16000 * 30
        speech_num = wav.shape[0] // whipser_len + 1
        for i in range(speech_num):
            temp_wav = wav[i * whipser_len:(i + 1) * whipser_len]
            _speech_tensor = self.speech_processor(raw_speech=temp_wav,
                                                   sampling_rate=16000,
                                                   return_tensors="pt",
                                                   return_attention_mask=True)["input_features"].squeeze()  # (128, 3000)
            speech_tensor.append(_speech_tensor)

        speech_tensor = torch.stack(speech_tensor, dim=0).squeeze()

        speech_tensor = [speech_tensor.to(dtype=self.model.dtype, device=self.model.device, non_blocking=True)]

        inp = DEFAULT_SPEECH_TOKEN
        conv_mode = 'qwen2vl'
        conv = conv_templates[conv_mode].copy()

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                speeches=speech_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=max_new_tokens,
                bos_token_id=151643,  # Begin of sequence token
                eos_token_id=[151645, 151643],  # End of sequence token
                pad_token_id=151643,  # Pad token
                use_cache=True)
        output_ids, _ = outputs
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


class LyraMiniAssistant(LyraAssistant):
    def __init__(self):
        self.download_model()
        self.load_model('model_zoo/Lyra_Mini_3B')

    def download_model(self):
        if not os.path.exists("model_zoo/vision/Qwen2VL_2B_ViT"):
            snapshot_download(
                repo_id="zszhong/Lyra_Qwen2VL_2B_ViT",
                local_dir="model_zoo/vision/Qwen2VL_2B_ViT",
            )
        if not os.path.exists('model_zoo/audio/whisper-large-v3-turbo'):
            snapshot_download(
                repo_id="openai/whisper-large-v3-turbo",
                local_dir='model_zoo/audio/whisper-large-v3-turbo',
            )
        if not os.path.exists('model_zoo/Lyra_Mini_3B'):
            snapshot_download(
                repo_id="zszhong/Lyra_Mini_3B",
                local_dir='model_zoo/Lyra_Mini_3B',
            )


class LyraBaseAssistant(LyraAssistant):
    def __init__(self):
        self.download_model()
        self.load_model('model_zoo/Lyra_Base_9B')

    def download_model(self):
        if not os.path.exists("model_zoo/vision/Qwen2VL_7B_ViT"):
            snapshot_download(
                repo_id="zszhong/Lyra_Qwen2VL_7B_ViT",
                local_dir="model_zoo/vision/Qwen2VL_7B_ViT",
            )
        if not os.path.exists('model_zoo/audio/whisper-large-v3'):
            snapshot_download(
                repo_id="openai/whisper-large-v3",
                local_dir='model_zoo/audio/whisper-large-v3',
            )
        if not os.path.exists('model_zoo/Lyra_Base_9B'):
            snapshot_download(
                repo_id="zszhong/Lyra_Base_9B",
                local_dir='model_zoo/Lyra_Base_9B',
            )