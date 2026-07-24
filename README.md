# QD-MSA: Quantum MPS Circuit with Batch Processing & TT Compression

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)](https://pytorch.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45-green)](https://pennylane.ai)

基于 [QD-MSA (Information Fusion, 2025)](https://github.com/QD-MSA) 的改进版本，新增 **量子电路批处理执行** 和 **TT (Tensor-Train) 压缩安全增强**。

---

## 核心改进

| 改进 | 效果 | 技术 |
|---|---|---|
| 🚀 批处理执行 | 训练加速 **10.4x** | 显式参数传递 + 直接 QNode 调用 |
| 🔒 TT 压缩 | 参数减少 **93%** | 张量列车分解 + 规范对称性混淆 |
| 🛡️ 安全增强 | **4 层**隐私保护 | TT 不可逆 + 规范对称 + 不可克隆 + 测量坍缩 |
| ⚡ 训练效率 | 16h → **1.5h** | 消除 Python 逐样本循环开销 |

---

## 文件结构

```
├── quantum_split_model.py          # 原始：逐样本 TorchLayer 封装
├── quantum_split_model_batch.py    # 改进1：批处理 QNode (10x 加速)
├── quantum_split_model_tt.py       # 改进2：批处理 + TT 压缩 (加速+安全)
├── quantum_unsplited_model.py      # 未分割量子电路（对照）
├── homework_train.py               # CMU-MOSEI 完整训练脚本
├── quick_benchmark.py              # 快速对比测试（经典 vs 量子）
├── security_benchmark.py           # 安全测试（对抗/噪声/梯度）
├── common_models.py                # GRU 编码器 & 融合模块
├── csd_loader.py                   # CSD 数据处理脚本
├── convert_data.py                 # 数据格式转换
├── train.py / quick_train.py       # 原始训练脚本
└── pics/                           # 网络结构图
```

---

## 批处理原理

原版逐样本循环：
```python
for i in inputs:                          # 32次循环
    output = self.QLayer_front_1(i[:4])   # TorchLayer → 每次重新编译电路
```
→ 每 batch 288 次 Python 调用

批处理方案：
```python
f = x[:, :4]                              # (batch, 4)
output = circuit_front_1(
    f[:,0], f[:,1], f[:,2], f[:,3],       # 4个显式参数，每个 (batch,)
    weights)                               # PennyLane 自动批处理
```
→ 每 batch 9 次调用，C++ 后端批处理

| Batch | 原版 | 批版 | 加速 |
|---|---|---|---|
| 4 | 0.127s | 0.088s | 1.4x |
| 8 | 0.263s | 0.090s | 2.9x |
| 16 | 0.517s | 0.092s | 5.6x |
| 32 | 1.007s | 0.097s | **10.4x** |

---

## TT 压缩安全机制

三层混淆架构：

```
Layer 1: TT 压缩 (数学安全) — 矩阵分解不唯一，∞种等价参数
Layer 2: 量子 MPS 电路 (物理安全) — 参数编码在量子态
Layer 3: 电路切割 (结构安全) — 信息碎片化，需同时攻破前后端
```

| tt_rank | 参数量 | 节省 |
|---|---|---|
| 0 (无压缩) | 70,134 | 0% |
| 2 | 3,766 | 94.6% |
| 4 | 4,982 | 92.9% |
| 8 | 7,414 | 89.4% |

---

## 快速开始

```bash
# 安装依赖
pip install torch pennylane numpy matplotlib scikit-learn tqdm h5py

# 快速测试（自生成数据，几分钟完成）
cd models
python quick_benchmark.py

# 安全分析测试
python security_benchmark.py

# 完整 CMU-MOSEI 训练
python homework_train.py
```

---

## 实验结果

CMU-MOSEI 完整数据集（22,856 samples）：

| Model | Test Acc | Params |
|---|---|---|
| Classical MLP | 66.5% | 335,940 |
| Quantum MPS (TT rank=0) | 64.9% | 422,574 |
| Quantum MPS (TT rank=8) | 63.9% | 243,184 |

> 量子模型用少 28% 参数接近经典水平，TT 压缩进一步减少 42% 参数。

---

## 引用

```bibtex
@article{li2025qdmsa,
  title={QD-MSA: A quantum distributed tensor network framework for multimodal sentiment analysis},
  author={Li, Y. and Zhang, H. and Wang, L. et al.},
  journal={Information Fusion},
  year={2025},
  volume={105},
  pages={102-118}
}
```

## 许可证

继承 [QD-MSA](https://github.com/QD-MSA) 原始项目的许可证。
