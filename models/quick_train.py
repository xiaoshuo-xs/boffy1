"""
QD-MSA 快速训练版本 — 适合完成作业/演示
- 截断序列长度加速
- 只用部分数据
- 输出清晰的结果报告
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import sys
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from common_models import GRU, GRUWithLinear, MMDL, Concat
from quantum_split_model import QNNSplited
from quantum_unsplited_model import QNNUnsplitted


# ============================================================
# 1. 数据加载（截断序列，加速训练）
# ============================================================

class MMSADataset(Dataset):
    def __init__(self, data, max_seq_len=50):
        self.samples = []
        self.num_classes = 7
        for item in data:
            # 截断长序列
            t_len = min(item['text'].shape[0], max_seq_len)
            a_len = min(item['audio'].shape[0], max_seq_len)
            v_len = min(item['vision'].shape[0], max_seq_len)
            m_len = min(t_len, a_len, v_len)

            if m_len < 3:
                continue

            self.samples.append({
                'text': torch.FloatTensor(item['text'][:m_len]),
                'audio': torch.FloatTensor(item['audio'][:m_len]),
                'vision': torch.FloatTensor(item['vision'][:m_len]),
                'label': int(item['label']),
                'len': m_len
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'text': s['text'],
            'audio': s['audio'],
            'vision': s['vision'],
            'label': s['label'],
            'len': s['len']
        }


def collate_fn(batch):
    text_b, audio_b, vision_b, lens_b, labels_b = [], [], [], [], []

    for item in batch:
        text_b.append(item['text'])
        audio_b.append(item['audio'])
        vision_b.append(item['vision'])
        lens_b.append(item['len'])
        labels_b.append(item['label'])

    # Pad
    def pad(seqs):
        m = max(s.shape[0] for s in seqs)
        d = seqs[0].shape[1]
        p = torch.zeros(len(seqs), m, d)
        for i, s in enumerate(seqs):
            p[i, :s.shape[0], :] = s
        return p

    data = (
        [pad(text_b), pad(audio_b), pad(vision_b)],
        [torch.LongTensor(lens_b), torch.LongTensor(lens_b), torch.LongTensor(lens_b)]
    )
    return data, torch.LongTensor(labels_b)


# ============================================================
# 2. 训练函数
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for data, labels in tqdm(loader, desc="  Train", leave=False):
        x_data, x_lens = data
        x_data = [d.to(device) for d in x_data]
        x_lens = [l.to(device) for l in x_lens]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model([x_data, x_lens])
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    for data, labels in tqdm(loader, desc="  Eval", leave=False):
        x_data, x_lens = data
        x_data = [d.to(device) for d in x_data]
        x_lens = [l.to(device) for l in x_lens]
        labels = labels.to(device)

        outputs = model([x_data, x_lens])
        total_loss += criterion(outputs, labels).item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1, all_preds, all_labels


# ============================================================
# 3. 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  QD-MSA 快速训练 — Multimodal Sentiment Analysis")
    print("  量子-经典混合模型 (Quantum Split MPS Circuit)")
    print("=" * 70)

    # ---- 配置 ----
    data_path = 'D:/qdmsa_data/mosei/processed.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = 50       # 截断到最多 50 个时间步
    n_train_samples = 800  # 训练集样本数
    n_val_samples = 150    # 验证集样本数
    n_test_samples = 200   # 测试集样本数
    batch_size = 8
    epochs = 15
    lr = 1e-4
    model_type = 'split'   # 'split' 或 'unsplit'

    print(f"\n配置:")
    print(f"  设备:     {device}")
    print(f"  模型:     {model_type}")
    print(f"  序列截断: {max_seq_len}")
    print(f"  训练样本: {n_train_samples}")
    print(f"  Batch:    {batch_size}")
    print(f"  Epochs:   {epochs}")
    print(f"  LR:       {lr}")

    # ---- 加载数据 ----
    print(f"\n[1/4] 加载数据...")
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"  原始数据: {len(raw_data)} segments")

    # 随机采样
    np.random.seed(42)
    idx = np.random.permutation(len(raw_data))
    total_needed = n_train_samples + n_val_samples + n_test_samples
    idx = idx[:total_needed]

    train_idx = idx[:n_train_samples]
    val_idx = idx[n_train_samples:n_train_samples + n_val_samples]
    test_idx = idx[n_train_samples + n_val_samples:]

    train_data = [raw_data[i] for i in train_idx]
    val_data = [raw_data[i] for i in val_idx]
    test_data = [raw_data[i] for i in test_idx]

    train_ds = MMSADataset(train_data, max_seq_len)
    val_ds = MMSADataset(val_data, max_seq_len)
    test_ds = MMSADataset(test_data, max_seq_len)

    print(f"  训练集: {len(train_ds)}  验证集: {len(val_ds)}  测试集: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- 构建模型 ----
    print(f"\n[2/4] 构建模型...")

    # 自动检测维度
    sample = train_ds[0]
    text_dim = sample['text'].shape[1]
    audio_dim = sample['audio'].shape[1]
    vision_dim = sample['vision'].shape[1]
    print(f"  特征维度: Text={text_dim}, Audio={audio_dim}, Vision={vision_dim}")

    encoder_hidden = {
        0: min(256, text_dim * 2),
        1: min(128, audio_dim * 2 + 2),
        2: min(128, vision_dim * 2 + 2)
    }

    encoders = [
        GRUWithLinear(text_dim, max(64, text_dim // 2), encoder_hidden[0],
                      dropout=1e-1, has_padding=True, batch_first=True),
        GRUWithLinear(audio_dim, max(64, audio_dim // 2), encoder_hidden[1],
                      dropout=1e-1, has_padding=True, batch_first=True),
        GRUWithLinear(vision_dim, max(64, vision_dim // 2), encoder_hidden[2],
                      dropout=1e-1, has_padding=True, batch_first=True)
    ]

    fusion = Concat(masks=[0, 1, 2])
    input_shape = sum(encoder_hidden[m] for m in [0, 1, 2])

    if model_type == 'split':
        head = QNNSplited(input_shape=input_shape, output_shape=7,
                          hidden_dim=256, with_shortcut=True)
    else:
        head = QNNUnsplitted(input_shape=input_shape, output_shape=7,
                             hidden_dim=256, with_shortcut=True)

    model = MMDL(encoders, fusion, head, has_padding=True).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_p:,}  可训练: {train_p:,}")
    print(f"  融合输入: {input_shape} 维")

    # ---- 训练 ----
    print(f"\n[3/4] 开始训练...")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        marker = " *** BEST" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{model_type}.pt')

    # ---- 测试 ----
    print(f"\n[4/4] 测试集评估...")
    print("-" * 70)

    model.load_state_dict(torch.load(f'best_model_{model_type}.pt', weights_only=True))
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test F1:       {test_f1:.4f}")

    print(f"\n  分类报告:")
    print(classification_report(test_labels, test_preds,
                                target_names=['-3','-2','-1','0','+1','+2','+3'],
                                zero_division=0))

    # ---- 训练过程总结 ----
    print("=" * 70)
    print("  训练过程")
    print("=" * 70)
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(len(history['train_loss'])):
        marker = " *" if history['val_acc'][i] == best_val_acc else ""
        print(f"  {i+1:<8} {history['train_loss'][i]:<12.4f} "
              f"{history['train_acc'][i]:<12.4f} "
              f"{history['val_loss'][i]:<12.4f} "
              f"{history['val_acc'][i]:<12.4f}{marker}")

    print(f"\n  最佳验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  对应测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  论文报告准确率: 81.62% (需更多数据和 epoch)")
    print(f"\n  ✓ 训练完成! 可以截图给老师看。")
    print("=" * 70)


if __name__ == '__main__':
    main()
