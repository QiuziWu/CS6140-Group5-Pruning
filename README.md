# Structured Pruning for CRDNN-based VAD
## CS6140 — Machine Learning | Northeastern University | Group 5

本子项目实现对 SpeechBrain `vad-crdnn-libriparty` 模型的 **structured / unstructured L1-norm pruning**，在 LibriParty eval 集上评估 F1 / Miss Rate / False Alarm 等指标，研究稀疏度对 VAD 准确率与推理延迟的影响。

---

## 📁 文件结构
```
CS6140-Group5-Pruning/
├── README.md                 # 本文件
├── REPORT.md                 # 实验报告（方法 + 结果 + 分析）
├── 04_pruning.py             # Pruning + F1 评估主脚本
├── pretrained_models/
│   └── vad-crdnn-libriparty/ # 从 HuggingFace 自动下载
└── results/
    ├── pruning_results.csv   # 各稀疏度下的量化结果
    └── pruning_plots.png     # 四格可视化图
```

---

## ⚙️ 环境

- Python 3.12（3.13/3.14 与 SpeechBrain 有兼容性问题）
- 依赖：`torch==2.5.1`, `torchaudio==2.5.1`, `speechbrain==1.0.3`, `huggingface_hub<0.26`, `matplotlib`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchaudio==2.5.1
pip install speechbrain==1.0.3 "huggingface_hub<0.26" matplotlib
```

LibriParty 数据集 (`data/LibriParty/dataset/eval` + `metadata/eval.json`) 可从主 KD 仓库获取，或运行其中的 `setup_data.sh`。

---

## 🚀 运行

```bash
# 完整模式：含 per-file F1 推理 (≈ 10–15 分钟)
python 04_pruning.py \
    --audio_dir data/LibriParty/dataset/eval \
    --label_dir data/LibriParty/dataset/metadata

# 快速模式：只测模型大小 + latency (≈ 30 秒)
python 04_pruning.py --skip_inference
```

输出：
- `results/pruning_results.csv` — 逐行记录每个 sparsity 下的指标
- `results/pruning_plots.png` — F1 / MissR·FA / Latency / Non-zero Params 四张子图

---

## 🔬 方法摘要

Pruning 按层类型分别处理：

| 层类型 | Pruning 方式 | 维度 | 原因 |
|---|---|---|---|
| `Linear` | `ln_structured` (L1, dim=0) | 输出神经元 | 行级稀疏可真正减少计算量 |
| `GRU`    | `l1_unstructured`          | 权重元素 | PyTorch 不支持 recurrent 结构化剪枝 |
| `Conv2d` | `l1_unstructured`          | 权重元素 | 通道剪枝需重写下游层形状 |

Sparsity 扫描：**0% → 10% → 30% → 50%**，每次在 **原始 FP32 baseline 的副本** 上进行（避免累积剪枝）。

评估：对每个 wav 文件获得 `get_speech_segments()` 输出，和 `eval.json` 中 ground-truth 区间比对，frame-level（10 ms）计算 precision / recall / F1 / miss rate / false alarm。

---

## 📊 结果概览

| Model          | Sparsity | Non-zero | Latency | F1     | MissR  | FA     |
|----------------|----------|----------|---------|--------|--------|--------|
| FP32 Baseline  | 0%       | 109,744  | 19.3 ms | 0.9633 | 0.0400 | 0.0290 |
| Pruned 10%     | 9.46%    | 99,357   | 20.0 ms | 0.9610 | 0.0516 | 0.0219 |
| Pruned 30%     | 28.33%   | 78,657   | 21.0 ms | 0.9228 | 0.1294 | 0.0073 |
| Pruned 50%     | 47.19%   | 57,960   | 20.8 ms | 0.0013 | 0.9993 | 0.0000 |

**核心发现**：在 10% 稀疏度下几乎无精度损失 (F1 0.9633 → 0.9610)，30% 仍可用但 miss rate 翻了 3 倍，**50% 模型完全坍塌**——几乎不再触发语音激活。详细分析见 [REPORT.md](REPORT.md)。

---

## 🎯 相关子项目

本仓库为组 Project 的 Pruning 分支，另有：
- **Knowledge Distillation**：`CS6140-Group5-VAD-KD`
- **Baseline / Energy VAD**：同 KD 仓库 `01_`、`02_` 脚本
