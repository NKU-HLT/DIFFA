#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple

import datasets
import torch

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import AutoTokenizer, AutoModel, AutoConfig, WhisperConfig
# from transformers.deepspeed import is_deepspeed_zero3_enabled

from src.dataloader import prepare_train_dataset_and_collator
from src.modeling_DIFFA import DIFFAModel
from src.modeling_whisper_encoder import WhisperEncoder
from src.configuration_DIFFA import DIFFAConfig
from transformers import WhisperFeatureExtractor 
from src.sft_trainer import dLLMTrainer

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    llm_model: str = field(
        default="/path/to/models/LLaDA-8B-Instruct", metadata={"help": "the path of base model"}
    )
    whisper_model: str = field(
        default="/path/to/models/whisper-small", metadata={"help": "the path of whisper model"}
    )
    stage1_ckpt: str = field(
        default="None", metadata={"help": "the path of whisper model"}
    )
    training_stage: int = field(default=1)



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dataset_path: str = field(
        default="",
        metadata={
            "help": "train_dataset_path"
        },
    )



def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model)

    # wait to fix DIFFA Config.
    whisper_config = WhisperConfig.from_pretrained(model_args.whisper_model)
    llm_config = AutoConfig.from_pretrained(model_args.llm_model,trust_remote_code=True)
    DIFFA_config = DIFFAConfig(
        whisper_config.to_dict(),
        llm_config.to_dict(),
    )

    ### 5. Load dataset
    processor = WhisperFeatureExtractor()

    train_dataset, data_collator = prepare_train_dataset_and_collator(
            json_path=data_args.train_dataset_path,
            model_config=DIFFA_config,
            tokenizer=tokenizer,
            processor= processor,
            max_audio_length=30,
            max_length=2048,
            stage=model_args.training_stage,
        )
    
    
    # 6. Load pretrained model
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    model = DIFFAModel(DIFFA_config, tokenizer, stage=model_args.training_stage)
    if model_args.training_stage == 2:
        model = DIFFAModel.from_pretrained(model_args.stage1_ckpt)
        logger.info(f"Load ckpt from stage1: {model_args.stage1_ckpt}")
    model.whisper_model = WhisperEncoder.from_pretrained(model_args.whisper_model)
    model.llm_model = AutoModel.from_pretrained(model_args.llm_model,trust_remote_code=True)

    for name, param in model.whisper_model.named_parameters():
        param.requires_grad = False
    for name, param in model.llm_model.named_parameters():
        param.requires_grad = False
    

    # 7. Initialize Trainer
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 8. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    results = {}
    # 9. Save tokenizer for inference load
    tokenizer.save_pretrained(training_args.output_dir)

    return results


if __name__ == "__main__":
    main()