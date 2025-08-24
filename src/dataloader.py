import json
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from tqdm import tqdm

from transformers import WhisperFeatureExtractor 


def build_chat_prompt(messages, add_generation_prompt=True):
    """
    Build a chat-style prompt from structured messages.

    Args:
        messages (list): List of dicts with keys {"role", "content"}.
        add_generation_prompt (bool): Whether to append assistant role prompt.

    Returns:
        str: Formatted prompt string.
    """
    prompt = "<|startoftext|>"
    for msg in messages:
        if msg["role"] == "system":
            prompt += "<|start_header_id|>system<|end_header_id|>\n\n" + msg["content"] + "<|eot_id|>"
        elif msg["role"] == "user":
            prompt += "<|start_header_id|>user<|end_header_id|>\n\n" + msg["content"] + "<|eot_id|>"
        elif msg["role"] == "assistant":
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + msg["content"] + "<|eot_id|>"
        elif msg["role"] == "audio":
            prompt += "<|start_header_id|>audio<|end_header_id|>\n\n" + msg["content"] + "<|eot_id|>"
        else:
            raise ValueError(f"Unsupported role: {msg['role']}")

    if add_generation_prompt:
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt



class DIFFADataset(Dataset):
    def __init__(self, json_path, model_config, max_audio_length=30, processor=None, num_workers=16, eval=False, stage=1):
        """
        Dataset class for DIFFA model training and evaluation.

        Args:
            json_path (str): Path to the JSON file containing audio and text info.
            model_config: DIFFAConfig instance with model configurations.
            max_audio_length (int): Maximum audio duration in seconds.
            processor: Audio processor, if None it will be loaded from config.
            num_workers (int): Number of workers for parallel audio loading.
            eval (bool): Whether in evaluation mode (default False).
            stage (int): Dataset stage indicator.
        """
        self.data = []
        self.max_audio_length = max_audio_length
        self.sample_rate = 16000  # Default Whisper sample rate
        self.num_workers = num_workers

        # Load dataset from JSON
        self._load_data(json_path)

        # Evaluation mode (for diffusion LLM)
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))
        self.stage = stage


    def _load_data(self, json_path):
        """Load dataset entries from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Fetch a single sample by index."""
        item = self.data[idx]
        audio_path = item['audio_filepath']

        try:
            audio, cur_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if cur_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=cur_sample_rate, new_freq=self.sample_rate)
                audio = resampler(audio)
            audio = audio.squeeze(0)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            exit()

        # Truncate audio to max length
        max_length = int(self.max_audio_length * self.sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]

        # Set target for stage 1
        if self.stage == 1:
            item["target"] = item["transcription"]

        res = {
            "audio_id": item["audio_id"],
            "audio_waveform": audio,
            "input": item["input"],
            "messages": {
                "audio_filepath": item["audio_filepath"],
                "transcription": item["transcription"],
                "dataset": item["dataset"],
                "duration": item["duration"],
                "seed_transcript": item["seed_transcript"],
            },
            "target_text": item["target"],
        }

        # Extra field for diffusion LLM
        if self.eval:
            res["t"] = self.t[idx]
        return res


def forward_process(batch, mask_token_id, eps=1e-3):
        """
        Apply random masking to input tokens.

        Args:
            batch (dict): Batch containing input_ids.
            mask_token_id (int): Token ID used for masking.
            eps (float): Lower bound for noise ratio.

        Returns:
            tuple: (noisy_batch, t, mask_indices)
        """
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

def collate_fn(batch, tokenizer, processor, max_length=2048,
                system_prompt: str = "You are a helpful voice assistant.",
                ):
    """
    Custom collate function for audio-text dataset.

    It handles audio preprocessing, text tokenization, 
    and noisy masking for diffusion-based training.

    Args:
        batch (list): List of dataset items.
        tokenizer: Tokenizer for text input.
        processor: Audio processor.
        max_length (int): Maximum text length.
        system_prompt (str): Default system prompt.

    Returns:
        dict: Collated batch with audio features, tokenized inputs, and labels.
    """
    # 1. Extract audio and text from batch
    audio_waveforms = [item['audio_waveform'].numpy() for item in batch]
    input_texts = [item['input'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    # 2. Process audio features
    processor_outputs = processor(
        audio_waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True
    )
    input_audio_features = processor_outputs.input_features
    input_audio_features_mask = processor_outputs.attention_mask

    # 3. Process text and build prompts
    audio_info_length = []
    prompt_lengths = []
    full_inputs = []

    for i in range(len(batch)):
        audio_info = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "This is the audio: "},
        ]
        prompt_info = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "This is the audio: <audio>. " + input_texts[i]},
        ]
        chat_info = prompt_info + [{"role": "assistant", "content": target_texts[i]}]

        audio_info_text = build_chat_prompt(audio_info, add_generation_prompt=False)
        audio_info_text = audio_info_text[:-10]
        prompt_text = build_chat_prompt(prompt_info, add_generation_prompt=False)
        full_text = build_chat_prompt(chat_info, add_generation_prompt=False)

        # Tokenize prompts
        tokenized_audio_info = tokenizer(audio_info_text, return_tensors="pt", max_length=max_length, truncation=True)
        tokenized_prompt = tokenizer(prompt_text, return_tensors="pt", max_length=max_length, truncation=True)

        # Collect lengths
        audio_info_length.append(tokenized_audio_info.attention_mask.sum(-1) - 1)
        prompt_lengths.append(tokenized_prompt.attention_mask.sum(-1))
        full_inputs.append(full_text)

    # 4. Construct new batch dict
    new_batch = {}
    full_inputs_ids = tokenizer(full_inputs, return_tensors="pt", max_length=max_length, truncation=True, padding="longest")["input_ids"] 
    new_batch["input_ids"] = full_inputs_ids
    new_batch["audio_info_lengths"] = torch.cat(audio_info_length, dim=0).unsqueeze(1)
    new_batch["prompt_lengths"] = torch.cat(prompt_lengths, dim=0).unsqueeze(1)
    new_batch["labels"] = new_batch["input_ids"].clone()

    # Save original input_ids before masking
    original_input_ids = new_batch["input_ids"].clone()

    # 5. Apply noisy masking
    mask_token_id = 126336
    noisy_ids, t, mask_indices = forward_process(new_batch, mask_token_id=mask_token_id)
    new_batch["input_ids"] = noisy_ids.long()
    new_batch["t"] = t

    # 6. Update labels
    # a. Set non-masked tokens to -100 (ignored in loss)
    new_batch["labels"][~mask_indices] = -100

    # b. Restore prompt tokens and mask their labels
    prompt_lengths_tensor = new_batch["prompt_lengths"] 
    range_tensor = torch.arange(new_batch["input_ids"].shape[1], device=new_batch["input_ids"].device).unsqueeze(0)
    prompt_mask = range_tensor < prompt_lengths_tensor

    new_batch["input_ids"][prompt_mask] = original_input_ids[prompt_mask]
    new_batch["labels"][prompt_mask] = -100
    new_batch["num_prompt_tokens"] = prompt_mask.sum()
    
    # 7. Return final collated batch
    return {
        "audio_ids": [item["audio_id"] for item in batch],
        "input_audio_features": input_audio_features,
        "input_audio_features_mask": input_audio_features_mask,
        "messages": [item["messages"] for item in batch],
        "input_ids": new_batch["input_ids"],
        "audio_info_lengths": new_batch["audio_info_lengths"],
        "prompt_lengths": new_batch["prompt_lengths"],
        "labels": new_batch["labels"],
        "t": new_batch["t"],
        "num_prompt_tokens": new_batch["num_prompt_tokens"],
    }



def prepare_train_dataset_and_collator(
    json_path,
    model_config,
    tokenizer,
    processor=None,
    max_audio_length=30,
    max_length=2048,
    num_workers=4,
    stage=1,
):
    """
    Prepare training dataset and collator function.

    Args:
        json_path (str): Path to training dataset JSON file.
        model_config: Model configuration object.
        tokenizer: Tokenizer instance.
        processor: Audio processor (default: None).
        max_audio_length (int): Maximum audio duration in seconds.
        max_length (int): Maximum text length.
        num_workers (int): Number of data loading workers.
        stage (int): Dataset stage.

    Returns:
        tuple: (train_dataset, data_collator)
    """
    # Create training dataset
    train_dataset = DIFFADataset(
        json_path=json_path,
        model_config=model_config,
        max_audio_length=max_audio_length,
        num_workers=num_workers,
        stage=stage,
    )
    
    # Create data collator
    data_collator = lambda batch: collate_fn(
        batch,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
    )
    
    return train_dataset, data_collator
