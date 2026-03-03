# src/vrpo_dataloader.py

import torch
from torch.utils.data import Dataset

from src.dataloader import DIFFADataset, build_chat_prompt  # 复用现有的
# 的 collate_fn 已经用 AutoProcessor / WhisperFeatureExtractor，所以这里依旧用 processor


class DIFFAPreferenceDataset(Dataset):
    """
    在原有 DIFFADataset 基础上，额外暴露 chosen / rejected 文本。

    约定 json 中每条数据类似：
        {
          "audio_id": ...,
          "audio_filepath": ...,
          "input": "...",                 # 用户 query
          "target": "...",                # 选中的回复
          "rejected_target_text": "...",              # 被拒绝的回复
          "transcription": ...,
          ...
        }

    可以通过 chosen_field / rejected_field 改字段名。
    """

    def __init__(
        self,
        json_path,
        model_config,
        max_audio_length=30,
        num_workers=16,
        stage=1,
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
    ):
        # 直接复用原来的 DIFFADataset（包括音频加载、target_text 等）
        self.base_dataset = DIFFADataset(
            json_path=json_path,
            model_config=model_config,
            max_audio_length=max_audio_length,
            num_workers=num_workers,
            eval=False,
            stage=stage,
        )
        self.data = self.base_dataset.data  # 原始 json 列表
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        base_item 是原来 __getitem__ 的返回：
            {
              "audio_id": ...,
              "audio_waveform": ...,
              "input": ...,
              "messages": {...},
              "target_text": ...,
              ["t": ... (eval 模式)]
            }
        raw_item 是原始 json 字典，可以拿到 chosen / rejected。
        """
        base_item = self.base_dataset[idx]
        raw_item = self.data[idx]

        # 从 json 中读出偏好文本
        rejected_text = raw_item[self.rejected_field]

        base_item["rejected_target_text"] = rejected_text
        return base_item


def vrpo_collate_fn(
    batch,
    tokenizer,
    processor,
    max_length=2048,
    system_prompt: str = "You are a helpful voice assistant. Imagine you can hear the audio clips. Focus on the audios and respond directly to the prompts.",
):
    """
    VRPO 用的 collate_fn：
      - 复用原来的 audio + prompt 构造逻辑
      - 不做 forward_process mask
      - 输出 chosen/rejected 的 full 序列，交给 VRPO 的 ELBO 估计使用
    """
    # 1. audio waveform（完全照的写法）
    audio_waveforms = [item['audio_waveform'].numpy() for item in batch]
    input_texts = [item['input'] for item in batch]
    chosen_texts = [item['target_text'] for item in batch]
    rejected_texts = [item['rejected_target_text'] for item in batch]

    # 2. Whisper processor：让它自己 padding
    processor_outputs = processor(
        audio_waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_audio_features = processor_outputs.input_features
    input_audio_features_mask = processor_outputs.attention_mask

    # 3. 构造 audio_info_text / prompt_text / full_text（chosen & rejected）
    audio_info_length = []
    prompt_lengths = []
    full_chosen_texts = []
    full_rejected_texts = []

    for i in range(len(batch)):
        # 和原来一模一样的 system / user 信息
        audio_info = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "This is the audio: "},
        ]
        prompt_info = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "This is the audio: <audio>. " + input_texts[i]},
        ]

        # 两个不同的 assistant 回复
        chat_info_chosen = prompt_info + [{"role": "assistant", "content": chosen_texts[i]}]
        chat_info_rejected = prompt_info + [{"role": "assistant", "content": rejected_texts[i]}]

        audio_info_text = build_chat_prompt(audio_info, add_generation_prompt=False)
        audio_info_text = audio_info_text[:-10]  # 和原来一样，去掉最后一个 <|eot_id|>
        prompt_text = build_chat_prompt(prompt_info, add_generation_prompt=False)
        full_text_chosen = build_chat_prompt(chat_info_chosen, add_generation_prompt=False)
        full_text_rejected = build_chat_prompt(chat_info_rejected, add_generation_prompt=False)

        # 记录 audio_info_length / prompt_lengths —— 完全复用的逻辑
        tokenized_audio_info = tokenizer(
            audio_info_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        tokenized_prompt = tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        audio_info_length.append(tokenized_audio_info.attention_mask.sum(-1) - 1)
        prompt_lengths.append(tokenized_prompt.attention_mask.sum(-1))

        full_chosen_texts.append(full_text_chosen)
        full_rejected_texts.append(full_text_rejected)

    # 4. 批量 tokenize chosen / rejected 的 full 序列（Whisper 会自己 pad）
    chosen_input_ids = tokenizer(
        full_chosen_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="longest",
    )["input_ids"]

    rejected_input_ids = tokenizer(
        full_rejected_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="longest",
    )["input_ids"]

    # 5. 把长度信息堆起来（和原来一样）
    audio_info_lengths = torch.cat(audio_info_length, dim=0).unsqueeze(1)
    prompt_lengths = torch.cat(prompt_lengths, dim=0).unsqueeze(1)

    return {
        "audio_ids": [item["audio_id"] for item in batch],
        "input_audio_features": input_audio_features,
        "input_audio_features_mask": input_audio_features_mask,
        "messages": [item["messages"] for item in batch],  # 可选，保持一致
        "audio_info_lengths": audio_info_lengths,
        "prompt_lengths": prompt_lengths,
        
        # VRPO 关键字段：
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
        # 如果想调试，也可以把原始文本一起带上
        "input_texts": input_texts,
        "chosen_texts": chosen_texts,
        "rejected_texts": rejected_texts,
    }


def prepare_vrpo_dataset_and_collator(
    json_path,
    model_config,
    tokenizer,
    processor=None,
    max_audio_length=30,
    max_length=2048,
    num_workers=4,
    stage=1,
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
):
    """
    VRPO 用的数据集 + collator，最大程度复用原来的代码。

    - Dataset：继续用 DIFFADataset 做底层音频 & 基本字段读取
    - 额外从 json 中拿 chosen / rejected 文本
    - collator：和原来的 collate_fn 一样的 audio 处理，只是输出 chosen/rejected 的 full 序列
    """
    train_dataset = DIFFAPreferenceDataset(
        json_path=json_path,
        model_config=model_config,
        max_audio_length=max_audio_length,
        num_workers=num_workers,
        stage=stage,
        chosen_field=chosen_field,
        rejected_field=rejected_field,
    )

    data_collator = lambda batch: vrpo_collate_fn(
        batch,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,
    )

    return train_dataset, data_collator
