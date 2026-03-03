from datasets import load_dataset, Audio
from argparse import ArgumentParser
import json
from tqdm import tqdm
from loguru import logger


import torch
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor
from loguru import logger

from src.modeling_whisper_encoder import WhisperEncoder
from src.modeling_DIFFA import DIFFAModel

import os
from collections import OrderedDict

from peft import PeftModel, LoraConfig, get_peft_model
import re


def load_my_model(model_path, whisper_path, llm_path, accelerate=None, stage=2):
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    model = DIFFAModel.from_pretrained(model_path).cuda()
    model.stage = stage
    model.whisper_model = WhisperEncoder.from_pretrained(whisper_path).cuda()

    if accelerate == "fast_dllm" or "fast_dllm_with_dual_cache":
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
    else:
        from transformers import AutoModel as LLaDAModelLM
    


    model.llm_model = LLaDAModelLM.from_pretrained(llm_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer, feature_extractor

def extract_and_load_lora_weights(checkpoint_path, base_model):
    """
    从pytorch.bin中提取并加载LoRA权重到基础模型
    
    Args:
        checkpoint_path: checkpoint路径(包含pytorch.bin或pytorch_model.bin)
        base_model: 基础LLM模型
    
    Returns:
        加载了LoRA权重的模型
    """
    logger.info(f"Extracting LoRA weights from checkpoint: {checkpoint_path}")
    
    # 查找checkpoint文件
    if os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
    elif os.path.isdir(checkpoint_path):
        # 尝试多个可能的文件名
        possible_files = ['pytorch.bin', 'pytorch_model.bin', 'model.bin']
        checkpoint_file = None
        for fname in possible_files:
            fpath = os.path.join(checkpoint_path, fname)
            if os.path.exists(fpath):
                checkpoint_file = fpath
                break
        if checkpoint_file is None:
            raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading from: {checkpoint_file}")
    
    # 加载完整的state dict
    full_state_dict = torch.load(checkpoint_file, map_location='cpu')
    
    # 提取LoRA权重
    lora_state_dict = OrderedDict()
    llm_prefix = 'llm_model.'
    lora_keywords = ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']
    
    # 分析LoRA配置
    target_modules = ["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"]

    lora_r = 8 #8
    lora_alpha = 16 # notice! it is very important!
    
    for key, value in full_state_dict.items():
        is_lora = any(keyword in key for keyword in lora_keywords)
        is_llm = key.startswith(llm_prefix)
        
        if is_lora and is_llm:
            # 移除llm_model前缀
            new_key = key.replace(llm_prefix, '')
            
            # 转换为PEFT格式: base_model.model.xxx
            if not new_key.startswith('base_model.model.'):
                new_key = 'base_model.model.' + new_key
            
            lora_state_dict[new_key] = value

    
    logger.info(f"Found {len(lora_state_dict)} LoRA parameters")
    logger.info(f"LoRA rank: {lora_r}")
    logger.info(f"Target modules: {target_modules}")
    
    if len(lora_state_dict) == 0:
        logger.warning("No LoRA weights found! Returning base model.")
        return base_model
    
    # 创建LoRA配置
    lora_config = LoraConfig(
        r = lora_r if lora_r else 8,
        lora_alpha= lora_alpha,
        target_modules=list(target_modules) if target_modules else ["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    logger.info(f"LoRA config: {lora_config}")
    
    # 应用LoRA配置到基础模型
    model = get_peft_model(base_model, lora_config)
    
    # 加载LoRA权重
    logger.info("Loading LoRA weights into model...")
    incompatible_keys = model.load_state_dict(lora_state_dict, strict=False)
    
    if incompatible_keys.missing_keys:
        logger.warning(f"Missing keys: {incompatible_keys.missing_keys[:5]}...")  # 只显示前5个
    if incompatible_keys.unexpected_keys:
        logger.warning(f"Unexpected keys: {incompatible_keys.unexpected_keys[:5]}...")
    
    logger.info("LoRA weights loaded successfully!")
    
    return model



def load_my_model_merged_lora(model_path, whisper_path, llm_path, accelerate=None, stage=2):
    """
    加载并合并LoRA权重
    """
    logger.info("Loading model with merged LoRA from checkpoint...")
    
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    
    model = DIFFAModel.from_pretrained(model_path).cuda()
    model.stage = stage
    model.whisper_model = WhisperEncoder.from_pretrained(whisper_path).cuda()
    
    if accelerate == "fast_dllm" or "fast_dllm_with_dual_cache":
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
    else:
        from transformers import AutoModel as LLaDAModelLM
    # 加载基础模型
    base_llm = LLaDAModelLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 提取并加载LoRA
    lora_model = extract_and_load_lora_weights(model_path, base_llm)
    
    # 合并权重
    logger.info("Merging LoRA weights...")
    model.llm_model = lora_model.merge_and_unload().cuda()
    logger.info("LoRA weights merged successfully")
    
    model.eval()
    return model, tokenizer, feature_extractor


def generate_response(model, tokenizer, feature_extractor, audio_array, instruction, args):
    """生成回复"""
    audio_tensor = torch.tensor(audio_array)
    inputs = feature_extractor(
        audio_tensor.numpy(), 
        sampling_rate=16000, 
        return_tensors="pt", 
        return_attention_mask=True
    ).to("cuda")
    
    with torch.no_grad():
        response = model.generate(
            tokenizer,
            input_audio_features=inputs.input_features,
            input_audio_mask=inputs.attention_mask,
            system_prompt="You are a helpful voice assistant.",
            input_texts=[instruction],
            steps=args.steps,
            block_length=args.block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            mask_id=126336,
            max_new_tokens=args.max_new_tokens,
            accelerate=args.accelerate,
        )
    return response.strip()


def read_audio(wav_path):
    """读取音频文件"""
    audio, sr = torchaudio.load(wav_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    audio = audio.squeeze(0).numpy()
    return audio
