import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple

import datasets
import torch
from torch.utils.tensorboard import SummaryWriter

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


from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training



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
    previours_ckpt: str = field(
        default="None", metadata={"help": "the path of whisper model"}
    )
    training_stage: int = field(default=1)


@dataclass
class LoraArguments:
    """
    Arguments for LoRA configuration.
    """
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter for scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules for LoRA. If None, will use default modules."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )


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


def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量和比例
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def get_default_lora_target_modules(model_type: str) -> List[str]:
    """
    根据模型类型返回默认的LoRA目标模块
    """
    # model_type: llada
    return ["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"] # for llada


def apply_lora_to_llm(model, lora_args: LoraArguments, llm_config):
    if not lora_args.use_lora:
        logger.info("LoRA is disabled, skipping LoRA configuration")
        return model
    
    if lora_args.lora_target_modules:
        target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    else:
        model_type = "llada"
        target_modules = get_default_lora_target_modules(model_type)
    
    logger.info(f"Applying LoRA to modules: {target_modules}")
    
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    

    model.llm_model = get_peft_model(model.llm_model, lora_config)
    
    logger.info("LoRA configuration applied successfully")
    logger.info(f"LoRA config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, "
                f"dropout={lora_args.lora_dropout}, target_modules={target_modules}")
    
    return model


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    training_args.report_to = ["tensorboard"]
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    training_args.logging_steps = 10  # 每10步记录一次,可根据需要调整
    
    os.makedirs(training_args.logging_dir, exist_ok=True)

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
    logger.info(f"LoRA parameters {lora_args}")
    logger.info(f"Dataset parameters {data_args}")
    logger.info(f"TensorBoard logs will be saved to: {training_args.logging_dir}")

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

    # 4. Load nnizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model)

    # wait to fix DIFFA Config.
    whisper_config = WhisperConfig.from_pretrained(model_args.whisper_model)
    llm_config = AutoConfig.from_pretrained(model_args.llm_model, trust_remote_code=True)
    DIFFA_config = DIFFAConfig(
        whisper_config.to_dict(),
        llm_config.to_dict(),
    )

    ### 5. Load dataset
    processor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)

    train_dataset, data_collator = prepare_train_dataset_and_collator(
            json_path=data_args.train_dataset_path,
            model_config=DIFFA_config,
            tokenizer=tokenizer,
            processor=processor,
            max_audio_length=30,
            max_length=2048,
            stage=model_args.training_stage,
        )
    
    
    # 6. Load pretrained model
    model = DIFFAModel(DIFFA_config, tokenizer, stage=model_args.training_stage)
    if model_args.training_stage == 2:
        model = DIFFAModel.from_pretrained(model_args.previours_ckpt)
        logger.info(f"Load ckpt from stage1: {model_args.previours_ckpt}")
    
    model.whisper_model = WhisperEncoder.from_pretrained(model_args.whisper_model)
    model.llm_model = AutoModel.from_pretrained(model_args.llm_model, trust_remote_code=True)

    # 冻结whisper模型
    for name, param in model.whisper_model.named_parameters():
        param.requires_grad = False
    
    # 冻结LLM模型(LoRA会自动处理需要训练的参数)
    for name, param in model.llm_model.named_parameters():
        param.requires_grad = False
    
    # 应用LoRA到LLM backbone
    model = apply_lora_to_llm(model, lora_args, llm_config)
    
    # 打印可训练参数信息
    logger.info("\n" + "="*50)
    logger.info("Trainable Parameters Summary:")
    logger.info("="*50)
    print_trainable_parameters(model)
    logger.info("="*50 + "\n")

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
        
        logger.info("Starting training...")
        logger.info(f"TensorBoard command: tensorboard --logdir={training_args.logging_dir}")
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        # 如果使用LoRA,额外保存LoRA权重
        if lora_args.use_lora:
            lora_output_dir = os.path.join(training_args.output_dir, "lora_weights")
            os.makedirs(lora_output_dir, exist_ok=True)
            model.llm_model.save_pretrained(lora_output_dir)
            logger.info(f"LoRA weights saved to: {lora_output_dir}")
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    results = {}
    # 9. Save tokenizer for inference load
    tokenizer.save_pretrained(training_args.output_dir)

    return results


if __name__ == "__main__":
    main()