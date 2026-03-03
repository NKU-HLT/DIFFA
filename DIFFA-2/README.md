
# <img src="assets/diffa_logo.png" alt="logo" width="80" style="vertical-align: middle;"/> DIFFA-2: A Practical Diffusion Large Language Model for General Audio Understanding

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2601.23161v1) 
[![🤗 Hugging Face](https://img.shields.io/badge/🤗Hugging%20Face-DIFFA-FFEB3B)](https://huggingface.co/zhoujiaming777/DIFFA-2) 
[![GitHub](https://img.shields.io/badge/Github-DIFFA-blue)](https://github.com/NKU-HLT/DIFFA) 


**DIFFA-2** is a next-generation diffusion-based large language model (dLLM) tailored for general audio understanding. Building upon the original DIFFA framework, DIFFA-2 enhances model capabilities by leveraging a novel architecture, advanced training strategies, and practical inference mechanisms. Its core innovations include:

- 🚀 **Dual-Adapter Architecture**: A hybrid design combining **semantic** and **acoustic** adapters to capture both linguistic and sound characteristics, enabling comprehensive understanding across speech, sound, and music domains.
- 📈 **Four-Stage Curriculum**: A meticulously designed training pipeline that includes: 
  1. **Semantic Alignment** — Aligning audio data with semantic understanding.
  2. **Joint Alignment** — Joint optimization of both acoustic and semantic representations.
  3. **Supervised Fine-Tuning (SFT)** — Further refining the model with LoRA to ensure high performance.
  4. **Variance-Reduced Preference Optimization (VRPO)** — A cutting-edge technique for aligning model outputs with human preferences.
- ⚡ **Practical Inference**: The model integrates **Factor-based Parallel Decoding (FPD)**, allowing faster and more efficient inference by reducing latency in comparison to traditional autoregressive models.
- 🏆 **Strong Performance**: DIFFA-2 achieves comparable performance on widely recognized audio understanding datasets such as MMSU, MMAU, and MMAR.

---

## 🛠️ Installation

To begin using DIFFA-2, it is recommended to set up a clean conda environment for managing dependencies.

```bash
# Create and activate a new environment
conda create -n diffa2 python=3.10
conda activate diffa2

# Install the required dependencies
pip install -r requirements.txt
````

---

## 📂 Data Preparation

DIFFA-2 follows a four-stage data preparation process that ensures high-quality training and model alignment with human preferences.

### Stages 1-3: Supervised Fine-Tuning (SFT) Data

The initial three stages of data preparation align with the DIFFA protocol ([See Example](https://huggingface.co/zhoujiaming777/DIFFA/blob/main/data/stage2_train.json)):

1. **Stage 1**: **Semantic Alignment** using ASR data (freezing dLLM).
2. **Stage 2**: **Joint Alignment** between semantic and acoustic features.
3. **Stage 3**: **Supervised Fine-Tuning (SFT)** with **LoRA** to fine-tune the model on specific audio datasets.

### Stage 4: Preference Optimization (VRPO)

Stage 4 introduces **Variance-Reduced Preference Optimization (VRPO)**, which is designed to align the model's output with human preferences. This stage requires datasets that include both chosen and rejected pairs. For example:

```json
[
  {
    "audio_id": "VocalSound/VocalSound/audio_16k/f1899_0_sigh.wav",
    "audio_filepath": "/path/to/datasets/VocalSound/VocalSound/audio_16k/f1899_0_sigh.wav",
    "transcription": "",
    "input": "After listening to the sound, can you summarize what stands out most in just one sentence?",
    "target": "The sound that stands out most is a sigh at the beginning.",
    "dataset": "VocalSound",
    "duration": 1.0,
    "seed_transcript": "[00:00-00:02] (Sigh)",
    "chosen": "The sound that stands out most is a sigh at the beginning.",
    "rejected": "The sound that stands out most is a soft whisper at the beginning.",
    "dpo_explanation": "The incorrect answer described the sound as a 'soft whisper' instead of the correct 'sigh'.",
    "dpo_source": "openqa_qwen3_32b_v1"
  }
]
```

---

## 🚅 Training

DIFFA-2 adopts a multi-stage training approach to ensure robust learning and fine-tuning.

**Note**: Before running any scripts, make sure to update the `MODEL_PATH`, `DATA_PATH`, and `OUTPUT_DIR` in the respective shell scripts.

### Example: Run Training for Stage X (Replace X with 1, 2, 3, or 4)

```bash
bash train_stageX_multinode.sh
```

---

## 📊 Evaluation & Inference

To evaluate the performance of DIFFA-2, we provide scripts that allow you to benchmark the model on standard audio understanding datasets such as MMSU.

### Standard Inference

Ensure that the model path is correctly specified in the script:

```bash
bash run_mmsu_inference.sh
```

### Accelerated Inference (FPD)

To enable **Factor-based Parallel Decoding (FPD)**, which accelerates inference, simply add the `--accelerate` flag for interence_xxx.py.

---

## 📝 Citation

If you find **DIFFA-2** useful in your research, please cite our paper:

```bibtex
@article{zhou2026diffa,
  title={DIFFA-2: A Practical Diffusion Large Language Model for General Audio Understanding},
  author={Zhou, Jiaming and Cheng, Xuxin and Zhao, Shiwan and Jia, Yuhang and Liu, Cao and Zeng, Ke and Cai, Xunliang and Qin, Yong},
  journal={arXiv preprint arXiv:2601.23161},
  year={2026}
}

@article{zhou2025diffa,
  title={DIFFA: Large Language Diffusion Models Can Listen and Understand},
  author={Zhou, Jiaming and Chen, Hongjie and Zhao, Shiwan and Kang, Jian and Li, Jie and Wang, Enzhi and Guo, Yujie and Sun, Haoqin and Wang, Hui and Kong, Aobo and others},
  journal={arXiv preprint arXiv:2507.18452},
  year={2025}
}
```

