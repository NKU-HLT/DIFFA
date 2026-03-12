import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import torch
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    AutoConfig,
    WhisperConfig,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import WhisperFeatureExtractor, AutoModel

from src.modeling_DIFFA import DIFFAModel
from src.modeling_whisper_encoder import WhisperEncoder
from src.configuration_DIFFA import DIFFAConfig
from src.vrpo.vrpo_dataloader import prepare_vrpo_dataset_and_collator
from src.vrpo.vrpo_trainer import DIFFA_VRPOTrainer

from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


# ========= 1. 参数定义 =========

@dataclass
class ModelArguments:
    """基础模型路径 + 上一阶段 ckpt 路径"""
    llm_model: str = field(
        default="/path/to/models/LLaDA-8B-Instruct",
        metadata={"help": "LLaDA base 模型路径（用于 tokenizer/config + base 权重）"},
    )
    whisper_model: str = field(
        default="/path/to/models/whisper-small",
        metadata={"help": "Whisper base 模型路径（用于 encoder + feature extractor）"},
    )
    previours_ckpt: str = field(
        default="",
        metadata={"help": "上一阶段 LoRA SFT 的 checkpoint 目录（Trainer.save_model 输出目录）"},
    )
    training_stage: int = field(
        default=3,
        metadata={"help": "当前训练 stage 标记（VRPO 可以设为 3，仅用于日志）"},
    )


@dataclass
class DataTrainingArguments:
    """VRPO preference 数据"""
    train_dataset_path: str = field(
        default="",
        metadata={"help": "VRPO preference json/jsonl 文件路径，需包含 chosen / rejected 字段"},
    )
    max_audio_length: int = field(
        default=30,
        metadata={"help": "最大音频长度（秒），传给 DIFFADataset 使用"},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "最大文本长度（token），用于 tokenizer 截断"},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "DataLoader 的 num_workers（如你后续想用）"},
    )
    chosen_field: str = field(
        default="target",
        metadata={"help": "json 中优选回复字段名"},
    )
    rejected_field: str = field(
        default="rejected",
        metadata={"help": "json 中劣选回复字段名"},
    )


@dataclass
class LoraArguments:
    """LoRA 配置（要和上一阶段保持一致）"""
    use_lora: bool = field(
        default=True,
        metadata={"help": "VRPO 阶段是否继续使用 LoRA（一般 True）"},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "逗号分隔的 target modules，如果为空走默认列表"},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "bias 类型: none / all / lora_only"},
    )


@dataclass
class VRPOArguments:
    """VRPO 超参"""
    beta: float = field(
        default=0.2,
        metadata={"help": "DPO/VRPO 中的 beta 系数"},
    )
    mc_steps: int = field(
        default=4,
        metadata={"help": "ELBO Monte Carlo 采样次数"},
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "[MASK] token id，需要和 SFT 阶段一致"},
    )


# ========= 2. 若干工具函数 =========

def get_default_lora_target_modules(model_type: str) -> List[str]:
    """根据模型类型返回默认的 LoRA target modules"""
    # 你的 LLaDA 上一阶段用的是这些
    return ["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"]


def apply_lora_to_llm(model: DIFFAModel, lora_args: LoraArguments, llm_config):
    """对 model.llm_model 挂载 LoRA 模块（不加载权重，只创建结构）"""
    if not lora_args.use_lora:
        logger.info("LoRA is disabled, skip PEFT wrapping.")
        return model

    # 确定 target modules
    if lora_args.lora_target_modules:
        target_modules = [model.strip() for m in lora_args.lora_target_modules.split(",")]
    else:
        target_modules = get_default_lora_target_modules("llada")

    logger.info(f"Applying LoRA to LLaDA modules: {target_modules}")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model.llm_model = get_peft_model(model.llm_model, lora_config)
    logger.info("LoRA modules created (PEFT wrapped LLaDA).")
    return model


def find_stage1_ckpt_file(stage1_ckpt_dir: str) -> str:
    """在 previours_ckpt 目录里找到实际的权重文件"""
    if os.path.isfile(stage1_ckpt_dir):
        return stage1_ckpt_dir

    possible_files = ["pytorch_model.bin", "pytorch.bin", "model.bin"]
    for fname in possible_files:
        fpath = os.path.join(stage1_ckpt_dir, fname)
        if os.path.exists(fpath):
            return fpath

    raise FileNotFoundError(f"No model bin file found in {stage1_ckpt_dir}")


def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    logger.info(
        f"Trainable params: {trainable:,} || All params: {total:,} || "
        f"Trainable%: {100 * trainable / total:.4f}%"
    )


def freeze_speech_and_llada_base_keep_lora(model: DIFFAModel):
    """
    冻结 Whisper encoder 和 LLaDA base 权重，只保留 LoRA (以及 DIFFA 其他模块) 可训练。

    约定：
    - LLaDA 的 LoRA 参数名里包含: lora_A / lora_B / lora_embedding_A / lora_embedding_B
    - 只动 model.whisper_model / model.llm_model，DIFFA 其他模块（如 audio projector）默认保持可训练
    """

    # 1) 冻结 Whisper encoder
    if hasattr(model, "whisper_model") and model.whisper_model is not None:
        for name, p in model.whisper_model.named_parameters():
            p.requires_grad = False
        logger.info("Frozen Whisper encoder parameters.")

    # 2) 冻结 LLaDA base，只保留 LoRA
    if hasattr(model, "llm_model") and model.llm_model is not None:
        lora_keywords = ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]

        for name, p in model.llm_model.named_parameters():
            if any(kw in name for kw in lora_keywords):
                p.requires_grad = True
            else:
                p.requires_grad = False

        logger.info("Frozen LLaDA base params, kept LoRA params trainable.")


# ========= 3. 主函数 =========

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoraArguments, VRPOArguments, TrainingArguments)
    )
    model_args, data_args, lora_args, vrpo_args, training_args = parser.parse_args_into_dataclasses()

    # TensorBoard
    training_args.report_to = ["tensorboard"]
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    training_args.logging_steps = 10
    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Logging
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

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        f"distributed: {bool(training_args.local_rank != -1)}, fp16: {training_args.fp16}"
    )
    logger.info(f"Training args: {training_args}")
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"LoRA args: {lora_args}")
    logger.info(f"VRPO args: {vrpo_args}")

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    # 检查是否从 VRPO ckpt 继续
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output dir ({training_args.output_dir}) exists and not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, will resume VRPO from {last_checkpoint}. "
                "Use --overwrite_output_dir to avoid this."
            )

    # 固定随机种子
    set_seed(training_args.seed)

    # ===== 3.1 tokenizer / config / processor =====
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model)
    whisper_config = WhisperConfig.from_pretrained(model_args.whisper_model)
    llm_config = AutoConfig.from_pretrained(model_args.llm_model, trust_remote_code=True)
    diffa_config = DIFFAConfig(
        whisper_config.to_dict(),
        llm_config.to_dict(),
    )

    processor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)

    # ===== 3.2 VRPO dataset & collator =====
    if not data_args.train_dataset_path:
        raise ValueError("`--train_dataset_path` (VRPO preference 数据) 不能为空。")

    train_dataset, data_collator = prepare_vrpo_dataset_and_collator(
        json_path=data_args.train_dataset_path,
        model_config=diffa_config,
        tokenizer=tokenizer,
        processor=processor,
        max_audio_length=data_args.max_audio_length,
        max_length=data_args.max_length,
        num_workers=data_args.num_workers,
        stage=model_args.training_stage,
        chosen_field=data_args.chosen_field,
        rejected_field=data_args.rejected_field,
    )

    # ===== 3.3 构造 policy / ref 模型：初始化 → 挂 LoRA → load ckpt =====
    if not model_args.previours_ckpt:
        raise ValueError("`--previours_ckpt` 必须指定为上一阶段 LoRA SFT 的 checkpoint 目录。")

    ckpt_file = find_stage1_ckpt_file(model_args.previours_ckpt)
    logger.info(f"Stage1 checkpoint file: {ckpt_file}")

    # --- 构造一个函数，把“初始化 + 挂 LoRA + load ckpt”封装一下 ---
    def build_model_with_lora_and_ckpt() -> DIFFAModel:
        # 1) 初始化 DIFFA 结构
        model = DIFFAModel(diffa_config, tokenizer, stage=model_args.training_stage)

        # 2) 加载 Whisper encoder & LLaDA base
        model.whisper_model = WhisperEncoder.from_pretrained(model_args.whisper_model)
        model.llm_model = AutoModel.from_pretrained(
            model_args.llm_model,
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16,  # 如果训练显存吃不消，这里可以改回 float32
        )

        # 3) 对 LLaDA 挂载 LoRA 模块（结构层面）
        model = apply_lora_to_llm(model, lora_args, llm_config)

        # 4) 加载上一阶段训练好的权重（包含 DIFFA + LoRA）
        state_dict = torch.load(ckpt_file, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"[Stage1 ckpt] Missing keys (first 10): {missing[:10]}")
        if unexpected:
            logger.warning(f"[Stage1 ckpt] Unexpected keys (first 10): {unexpected[:10]}")

        return model

    # policy model：继续训练
    model = build_model_with_lora_and_ckpt()

    # reference model：完全冻结
    ref_model = build_model_with_lora_and_ckpt()
    ref_model.to(torch.bfloat16)
    ref_model.eval()

    for _, p in ref_model.named_parameters():
        p.requires_grad = False

    # 只冻结 Whisper 和 LLaDA base，让 LoRA + DIFFA 其他模块可训练
    freeze_speech_and_llada_base_keep_lora(model)

    logger.info("\n" + "=" * 50)
    logger.info("Trainable parameter summary (Policy, VRPO stage):")
    logger.info("=" * 50)
    print_trainable_parameters(model)
    logger.info("=" * 50 + "\n")

    # ===== 3.4 VRPO Trainer =====
    trainer = DIFFA_VRPOTrainer(
        ref_model=ref_model,
        beta=vrpo_args.beta,
        mc_steps=vrpo_args.mc_steps,
        mask_id=vrpo_args.mask_id,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ===== 3.5 训练 =====
    if training_args.do_train:
        resume_ckpt = None
        if training_args.resume_from_checkpoint is not None:
            resume_ckpt = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            resume_ckpt = last_checkpoint

        logger.info("Starting VRPO training...")
        logger.info(f"TensorBoard: tensorboard --logdir={training_args.logging_dir}")

        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 保存 tokenizer，方便推理
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
