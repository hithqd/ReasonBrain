# Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning


<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2502.11079-b31b1b.svg)](https://arxiv.org/abs/2507.01908)&nbsp;
</div>

> [**Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning**](https://arxiv.org/abs/2507.01908)<br>
> [Qingdong He](https://scholar.google.com/citations?user=gUJWww0AAAAJ&hl=zh-CN)<sup> 1* </sup>, [Xueqin Chen](https://scholar.google.com/citations?user=6F-iHFsAAAAJ&hl=zh-CN)<sup> 2* </sup>, [Chaoyi Wang](https://orcid.org/0000-0003-0164-1953)<sup> 3 </sup>, [Yanjie Pan](https://github.com/chfyfr)<sup> 4 </sup>, [Xiaobin Hu](https://scholar.google.com/citations?user=3lMuodUAAAAJ&hl=th)<sup> 1 </sup>, [Zhenye Gan](https://scholar.google.com/citations?user=fa4NkScAAAAJ)<sup> 1 </sup>, [Yabiao Wang](https://scholar.google.com/citations?user=xiK4nFUAAAAJ&hl=zh-CN)<sup> 1 </sup>, [Chengjie Wang](https://scholar.google.com/citations?user=fqte5H4AAAAJ&hl=zh-CN)<sup> 1 </sup>, [Xiangtai Li](https://scholar.google.com/citations?user=FL3ReD0AAAAJ&hl=zh-CN)<sup> 5 </sup>, [Jiangning Zhang](https://scholar.google.com.hk/citations?user=2hA4X9wAAAAJ&hl=zh-CN)<sup> 1 </sup>
> <br><sup> * </sup>Equal contribution

<p align="center">
  <b>🎉🎉🎉Accepted by ICML 2026</b>
</p>

## 🌠 Key Features

<img src='assets/cover.png' width='100%' />
<br>

This repository provides a clean, modular PyTorch implementation of **ReasonBrain**, a framework that performs **hypothetical instruction-based image editing** by combining:

- **LLaVA-v1.1-7B** as the Multimodal LLM (LoRA fine-tuned).
- **FLUX.1-dev** as the diffusion backbone.
- **FRCE** (Fine-grained Reasoning Cue Extraction): dual-branch (Patch + SAM segmentation) visual cues plus ID-Controller for textual cues.
- **CME** (Cross-Modal Enhancer): vision-/text-oriented mixed cross-attention to recover details lost in the MLLM bottleneck.
- **QFormer**: 6-layer, 77 query tokens, aligns MLLM hidden states to the diffusion conditioning space.

It also provides utilities to (re)build **Reason50K**, the 51K-sample dataset covering Physical / Temporal / Causal / Story reasoning.

## 🚩 **Updates**

- ✅ **[2026-05-01]** ReasonBrain was accepted by **ICML 2026**.

## 🛠️ Installation

We recommend Python 3.10 + CUDA 12.1 + a recent PyTorch.

```bash
# 1. clone
git clone <this-repo>
cd ReasonBrain

# 2. create env
conda create -n reasonbrain python=3.10 -y
conda activate reasonbrain

# 3. PyTorch (adapt CUDA version to your driver)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# 4. project deps
pip install -r requirements.txt

# 5. optional: register the package
pip install -e .
```

### Pretrained Weights

| Component            | Source / HuggingFace ID                                  |
| -------------------- | -------------------------------------------------------- |
| LLaVA-v1.1-7B        | `liuhaotian/LLaVA-7b-v1` (or compatible LLaVA-1.5 / 1.6) |
| FLUX.1-dev           | `black-forest-labs/FLUX.1-dev`                           |
| SAM (ViT-H)          | `facebook/sam-vit-huge`                                  |
| CLIP ViT-L/14        | `openai/clip-vit-large-patch14`                          |

By default these are auto-downloaded by Hugging Face on first use. Cache location can be set via `HF_HOME`.

---

## 🎬 Dataset: Reason50K

The `(source_image, hypothetical_instruction, target_image)` triples spanning four reasoning categories:

| Category   | Example instruction                                       |
| ---------- | --------------------------------------------------------- |
| Physical   | "What happens to this ice cube left at room temperature?" |
| Temporal   | "Show this scene 50 years from now."                      |
| Causal     | "What if the dam in the picture broke?"                   |
| Story      | "After the dragon attacks, what does the village look like?" |

### 1. Expected on-disk format

```
data/reason50k/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── images/
    ├── src/000000.png
    └── tgt/000000.png
```

Each JSONL line:

```json
{
  "id": "000000",
  "category": "physical",
  "src_image": "images/src/000000.png",
  "tgt_image": "images/tgt/000000.png",
  "instruction": "What if this glass was dropped onto the floor?",
  "objects": ["glass", "floor"]
}
```

### 2. Re-building Reason50K

The construction pipeline follows §3 of the paper (reverse-generation: target → source). Given seed prompts:

```bash
python scripts/build_dataset.py \
    --config configs/data.yaml \
    --seeds data/seeds.jsonl \
    --out data/reason50k
```

Steps performed automatically:

1. GPT generates hypothetical instructions + target descriptions.
2. spaCy extracts referenced entities (`objects`).
3. A T2I diffusion model + IP-Adapter generates several candidate source images.
4. GPT scoring + LPIPS/CLIP perceptual filters pick the best source.

> Building the full 50K dataset requires OpenAI API credit & GPU time. The shipped sample (`data/sample/`) lets you smoke-test the full training/inference loop end-to-end.

---

## 🔥 Training

Single-node multi-GPU (using 🤗 Accelerate):

```bash
accelerate config            # one-time setup
accelerate launch scripts/train.py --config configs/default.yaml
```

Multi-node (paper setup: 16× H20) — fill in your hostfile / launcher and:

```bash
accelerate launch \
    --num_processes 16 --num_machines 2 \
    scripts/train.py --config configs/default.yaml
```

Key hyper-parameters (from the paper, exposed in `configs/default.yaml`):

| Item                          | Value         |
| ----------------------------- | ------------- |
| Optimizer                     | AdamW         |
| Learning rate                 | 1e-3          |
| Weight decay                  | 1e-2          |
| Batch size (global)           | 16            |
| LoRA rank / alpha             | 8 / 16        |
| Extra special tokens in MLLM  | 32 (`[IMG_*]`)|
| QFormer layers / queries      | 6 / 77        |
| MLLM                          | LLaVA-v1.1-7B |
| Diffusion model               | FLUX.1-dev    |

Checkpoints are written to `outputs/<run_name>/`.

---

## 🎮 Inference

```bash
python scripts/infer.py \
    --config configs/default.yaml \
    --ckpt outputs/reasonbrain/last \
    --src_image examples/glass.png \
    --instruction "What if this glass was dropped onto the floor?" \
    --out edited.png
```

For Python usage:

```python
from reasonbrain.inference.pipeline import ReasonBrainPipeline

pipe = ReasonBrainPipeline.from_pretrained("outputs/reasonbrain/last").to("cuda")
edited = pipe("examples/ice_cube.png",
              "What happens to this ice cube left at room temperature?")
edited.save("edited.png")
```

---

## 📊 Evaluation

```bash
python scripts/evaluate.py \
    --ckpt outputs/reasonbrain/last \
    --test data/reason50k/test.jsonl \
    --metrics clip_t clip_i dino lpips
```

Implemented metrics (`reasonbrain/evaluation/metrics.py`):

- **CLIP-T**: text-image alignment between `target_caption` and the edited image.
- **CLIP-I / DINO**: visual similarity between the edited image and the GT target.
- **LPIPS**: perceptual distance.


## ⭐ Citation

If ReasonBrain is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our [paper](https://arxiv.org/abs/2507.01908).

### BibTeX
```bibtex
@inproceedings{he2026reasonbrain,
  title     = {Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning},
  author    = {He, Qingdong and Chen, Xueqin and Wang, Chaoyi and Pan, Yanjie and
               Hu, Xiaobin and Gan, Zhenye and Wang, Yabiao and Wang, Chengjie and
               Li, Xiangtai and Zhang, Jiangning},
  booktitle = {ICML},
  year      = {2026}
}
```

## 📧 Contact
If you have any comments or questions regarding this open-source project, please open a new issue or contact [Qingdong He](heqingdong@alu.uestc.edu.cn).


## ⚖️ License

Code in this repository is released under the Apache-2.0 license. Note that the
underlying models (LLaVA, FLUX, SAM, CLIP) come with their own licenses; please respect
them.
