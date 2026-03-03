# <img src="assets/diffa_logo.png" alt="logo" width="80" style="vertical-align: middle;"/> DIFFA: Large Language Diffusion Models Can Listen and Understand

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2507.18452)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗Hugging%20Face-DIFFA-FFEB3B)](https://huggingface.co/zhoujiaming777/DIFFA)
[![GitHub](https://img.shields.io/badge/Github-DIFFA-blue)](https://github.com/NKU-HLT/DIFFA)

---

**DIFFA** is the **first diffusion-based large audio-language model (LALM)** for spoken language understanding.  
It leverages a frozen diffusion LLM with **dual adapters** (semantic + acoustic) to enhance **audio perception and reasoning**.  
As the first exploration of diffusion-based large language models (dLLMs) in speech and audio understanding, DIFFA opens new directions for non-autoregressive multimodal learning.
This repository provides the training data, checkpoints, inference scripts, and reproducible training pipelines to facilitate further research on diffusion LLMs in the audio domain.


---

## 🔥 News
- **2025.11.11**: DIFFA is accepted by AAAI 2026 !
- **2025.08.25**: Released the DIFFA [checkpoint](https://huggingface.co/zhoujiaming777/DIFFA) and code!
- **2025.07.25**: Our paper is released on [arXiv](https://arxiv.org/abs/2507.18452). 🎉  

---

## 🚀 Overview
Despite using only **960h ASR** and **127h synthetic instruction data**, DIFFA achieves competitive results compared to models trained on **hundreds of thousands of hours**.  

<p align="center">
  <img src="assets/radar.png" alt="Radar comparison between DIFFA and Qwen2-Audio" width="500">
</p>

*Figure: Radar chart comparing **DIFFA** and **Qwen2-Audio-Instruct** across multiple audio-language benchmarks.*

---

## ⚙️ Setup

### Python Environment
```bash
git clone https://github.com/NKU-HLT/DIFFA.git
cd DIFFA
conda create -n diffa python=3.10
conda activate diffa
pip install -r requirements.txt
````

### Checkpoints

Please download and set up the following models:

* [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
* [Whisper-Small](https://huggingface.co/openai/whisper-small)
* [DIFFA checkpoint](https://huggingface.co/zhoujiaming777/DIFFA)

Update `llm_path`, `whisper_path`, and `model_path` in the inference scripts before running.

---

## 🔍 Inference

We provide inference code for the following benchmarks:

* [MMSU](https://github.com/dingdongwang/mmsu_bench)
* [MMAU](https://github.com/Sakshi113/MMAU)
* [VoiceBench](https://github.com/MatthewCYM/VoiceBench)

Example (MMSU):

```bash
bash run_mmsu_inference.sh
```

After inference, run `evaluate.py` for each benchmark to compute final metrics.

---

## ⚠️ Note on Inference Speed

Currently, DIFFA’s inference is slower than autoregressive audio-language models. This is mainly because its backbone **LLaDA** has not yet been optimized for efficiency. In particular, diffusion-based LLMs lack KV-cache support and parallel decoding, which makes decoding slower compared to autoregressive models. Since this work is the *first exploration* of diffusion LLMs in the audio domain, our focus is on **evaluating performance** rather than optimizing speed. If you are interested in acceleration, we recommend looking into recent training-free methods such as [Fast-dLLM](https://arxiv.org/abs/2505.22618), which report **27.6× faster inference** and represent a promising direction for future integration.

---

## 📖 Training

We provide training scripts for reimplementation.

### Data Preparation

* **Stage 1**: [LibriSpeech](https://www.openslr.org/12)
* **Stage 2**: VoxCeleb1, AccentDB, IEMOCAP, DailyTalk, VCTK-Corpus

Data format and indices are available on [Hugging Face](https://huggingface.co/zhoujiaming777/DIFFA).

### Training Script

```bash
# Stage 1
bash train_stage1.sh

# Stage 2
bash train_stage2.sh
```

---

## 🙏 Acknowledgements

We sincerely thank the following open-source projects and authors for their contributions, which greatly inspired and facilitated this work:

* [BLSP](https://github.com/cwang621/blsp)
* [DESTA-2](https://github.com/kehanlu/DeSTA2/tree/main)
* [LLaDA](https://github.com/ML-GSAI/LLaDA)
* [d1](https://github.com/dllm-reasoning/d1)

These open-source efforts have greatly inspired and supported the development of DIFFA.
---

## 📖 Citation

If you find DIFFA useful, please cite:

```bibtex
@article{zhou2025diffa,
  title={DIFFA: Large Language Diffusion Models Can Listen and Understand},
  author={Zhou, Jiaming and Chen, Hongjie and Zhao, Shiwan and Kang, Jian and Li, Jie and Wang, Enzhi and Guo, Yujie and Sun, Haoqin and Wang, Hui and Kong, Aobo and others},
  journal={arXiv preprint arXiv:2507.18452},
  year={2025}
}
```

