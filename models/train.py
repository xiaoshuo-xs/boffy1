"""
QD-MSA 完整训练脚本
支持 CMU-MOSI / CMU-MOSEI 真实数据 + 模拟数据快速验证

用法:
    # 用模拟数据快速验证（推荐先跑这个）
    python train.py --synthetic

    # 用真实数据训练（需先手动下载）
    python train.py --data_dir ../data/mosei

    # 完整参数
    python train.py --synthetic --dataset mosei --model_type split --epochs 30 --batch_size 16
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import pickle
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import argparse

from common_models import GRU, GRUWithLinear, MMDL, Concat
from quantum_split_model import QNNSplited
from quantum_unsplited_model import QNNUnsplitted


# ============================================================
# 1. 模拟数据生成（用于快速验证流程）
# ============================================================

def generate_synthetic_data(num_samples=2280, dataset_type='mosei'):
    """
    生成模拟多模态数据，格式跟 CMU-MOSEI/MOSI 对齐

    CMU-MOSEI: ~22800 segments, 7-class sentiment [-3, 3]
    CMU-MOSI:  ~2200 segments, 7-class sentiment [-3, 3]

    各模态 feature dim (与 example.py 一致):
        mosei: text=713, audio=300, vision=74
        mosi:  text=35,  audio=74,  vision=300
    """
    print(f"Generating synthetic data ({dataset_type}, {num_samples} samples)...")

    if dataset_type == 'mosei':
        text_dim, audio_dim, vision_dim = 713, 300, 74
    else:
        text_dim, audio_dim, vision_dim = 35, 74, 300

    np.random.seed(42)
    data = []
    for i in range(num_samples):
        # 随机序列长度 (3-50 个时间步，模拟词级别对齐)
        seq_len = np.random.randint(3, 50)

        # 生成有意义的随机特征（用正弦波模拟时间序列结构）
        t = np.linspace(0, 4 * np.pi, seq_len).reshape(-1, 1)

        text_feat = np.sin(t * np.random.randn(text_dim) * 0.5 +
                           np.random.randn(text_dim)).astype(np.float32)
        audio_feat = np.cos(t * np.random.randn(audio_dim) * 0.3 +
                            np.random.randn(audio_dim)).astype(np.float32)
        vision_feat = np.sin(t * np.random.randn(vision_dim) * 0.4 +
                             np.random.randn(vision_dim)).astype(np.float32)

        # 7-class label [0, 6]
        label = np.random.randint(0, 7)

        data.append({
            'text': text_feat,
            'audio': audio_feat,
            'vision': vision_feat,
            'label': int(label),
            'seg_id': f'syn_{i:05d}'
        })

    print(f"Generated {len(data)} synthetic segments")
    print(f"  Text dim: {text_dim}, Audio dim: {audio_dim}, Vision dim: {vision_dim}")
    return data


# ============================================================
# 2. 真实数据加载（CMU Multimodal SDK）
# ============================================================

def download_and_process_mosei(data_dir):
    """加载并处理 CMU-MOSEI 数据集 (支持预下载的 CSD 文件)"""
    from mmsdk import mmdatasdk

    os.makedirs(data_dir, exist_ok=True)

    # 检查常见的 CSD 文件位置
    csd_paths = [
        # 从 HuggingFace 下载的文件名
        (os.path.join(data_dir, 'CMU_MOSEI_TimestampedWordVectors.csd'), 'text'),
        (os.path.join(data_dir, 'CMU_MOSEI_COVAREP.csd'), 'audio'),
        (os.path.join(data_dir, 'CMU_MOSEI_VisualFacet42.csd'), 'vision'),
        # SDK 默认命名
        (os.path.join(data_dir, 'cmu_mosei_text.csd'), 'text'),
        (os.path.join(data_dir, 'cmu_mosei_audio.csd'), 'audio'),
        (os.path.join(data_dir, 'cmu_mosei_vision.csd'), 'vision'),
    ]

    # 找到存在的文件
    text_path = audio_path = vision_path = None
    for path, kind in csd_paths:
        if os.path.exists(path):
            if kind == 'text' and text_path is None:
                text_path = path
            elif kind == 'audio' and audio_path is None:
                audio_path = path
            elif kind == 'vision' and vision_path is None:
                vision_path = path

    if text_path and audio_path and vision_path:
        print(f"Loading pre-downloaded CSD files...")
        print(f"  Text:   {os.path.basename(text_path)}")
        print(f"  Audio:  {os.path.basename(audio_path)}")
        print(f"  Vision: {os.path.basename(vision_path)}")
        text_ds = mmdatasdk.mmdataset({'text': text_path})
        audio_ds = mmdatasdk.mmdataset({'audio': audio_path})
        vision_ds = mmdatasdk.mmdataset({'vision': vision_path})
    else:
        print("No pre-downloaded CSD files found.")
        print("Please download from: https://huggingface.co/datasets/reeha-parkar/cmu-mosei-comp-seq")
        raise FileNotFoundError(f"CSD files not found in {data_dir}")

    print("Aligning modalities...")
    aligned = mmdatasdk.mmdataset.align(text_ds, audio_ds, vision_ds)
    return aligned


def download_and_process_mosi(data_dir):
    """下载并处理 CMU-MOSI 数据集"""
    from mmsdk import mmdatasdk

    os.makedirs(data_dir, exist_ok=True)

    csd_text = os.path.join(data_dir, 'cmu_mosi_text.csd')
    csd_audio = os.path.join(data_dir, 'cmu_mosi_audio.csd')
    csd_vision = os.path.join(data_dir, 'cmu_mosi_vision.csd')

    if os.path.exists(csd_text) and os.path.exists(csd_audio) and os.path.exists(csd_vision):
        print("Loading pre-downloaded CSD files...")
        text_ds = mmdatasdk.mmdataset(csd_text)
        audio_ds = mmdatasdk.mmdataset(csd_audio)
        vision_ds = mmdatasdk.mmdataset(csd_vision)
    else:
        print("Downloading CMU-MOSI from remote...")
        hl = mmdatasdk.cmu_mosi.highlevel
        text_ds = mmdatasdk.mmdataset(hl['glove_vectors'], data_dir)
        audio_ds = mmdatasdk.mmdataset(hl['COVAREP'], data_dir)
        vision_ds = mmdatasdk.mmdataset(hl['FACET 4.1'], data_dir)

    print("Aligning modalities...")
    aligned = mmdatasdk.mmdataset.align(text_ds, audio_ds, vision_ds)
    return aligned


def process_aligned_data(aligned_dataset, dataset_type='mosei'):
    """将对齐后的数据转为训练格式，自动检测 feature dim"""
    print("Processing aligned segments...")

    segments = list(aligned_dataset.keys())
    processed = []

    for seg_id in tqdm(segments, desc="Processing"):
        seg = aligned_dataset[seg_id]

        # 检测实际 key 名称
        keys = list(seg.keys())
        text_key = next((k for k in keys if 'glove' in k.lower() or 'word' in k.lower()), keys[0])
        audio_key = next((k for k in keys if 'covarep' in k.lower() or 'opensmile' in k.lower()), None)
        vision_key = next((k for k in keys if 'facet' in k.lower() or 'openface' in k.lower()), None)
        label_key = next((k for k in keys if 'label' in k.lower()), None)

        text_feat = seg.get(text_key)
        if text_feat is None:
            continue

        # 标签处理
        labels = None
        if label_key and label_key in seg:
            labels = seg[label_key]

        if labels is None:
            continue

        # 缺失模态补零
        if audio_key and audio_key in seg and seg[audio_key] is not None:
            audio_feat = seg[audio_key]
        else:
            audio_feat = np.zeros((text_feat.shape[0], 74))

        if vision_key and vision_key in seg and seg[vision_key] is not None:
            vision_feat = seg[vision_key]
        else:
            vision_feat = np.zeros((text_feat.shape[0], 35))

        # 对齐时间维度
        min_len = min(text_feat.shape[0], audio_feat.shape[0], vision_feat.shape[0])
        text_feat = text_feat[:min_len]
        audio_feat = audio_feat[:min_len]
        vision_feat = vision_feat[:min_len]

        # Segment 级别标签
        if labels.ndim > 1 and labels.shape[1] > 1:
            seg_label = np.argmax(np.bincount(np.argmax(labels[:min_len], axis=1)))
        else:
            seg_label = int(np.round(np.mean(labels[:min_len])))

        processed.append({
            'text': text_feat.astype(np.float32),
            'audio': audio_feat.astype(np.float32),
            'vision': vision_feat.astype(np.float32),
            'label': seg_label,
            'seg_id': seg_id
        })

    print(f"Processed {len(processed)} segments")
    if processed:
        print(f"  Text dim: {processed[0]['text'].shape[1]}")
        print(f"  Audio dim: {processed[0]['audio'].shape[1]}")
        print(f"  Vision dim: {processed[0]['vision'].shape[1]}")
    return processed


# ============================================================
# 3. PyTorch Dataset
# ============================================================

class MMSADataset(Dataset):
    """多模态情感分析数据集"""

    def __init__(self, data, num_classes=7):
        self.data = data
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text': torch.FloatTensor(item['text']),
            'audio': torch.FloatTensor(item['audio']),
            'vision': torch.FloatTensor(item['vision']),
            'label': item['label'],
            'text_len': item['text'].shape[0],
            'audio_len': item['audio'].shape[0],
            'vision_len': item['vision'].shape[0]
        }


def collate_fn(batch):
    """填充变长序列到同一 batch 内最大长度"""
    text_list, audio_list, vision_list = [], [], []
    text_lens, audio_lens, vision_lens = [], [], []
    labels = []

    for item in batch:
        text_list.append(item['text'])
        audio_list.append(item['audio'])
        vision_list.append(item['vision'])
        text_lens.append(item['text_len'])
        audio_lens.append(item['audio_len'])
        vision_lens.append(item['vision_len'])
        labels.append(item['label'])

    def pad_sequences(seqs):
        max_len = max(s.shape[0] for s in seqs)
        feat_dim = seqs[0].shape[1]
        padded = torch.zeros(len(seqs), max_len, feat_dim)
        for i, s in enumerate(seqs):
            padded[i, :s.shape[0], :] = s
        return padded

    data = (
        [pad_sequences(text_list), pad_sequences(audio_list), pad_sequences(vision_list)],
        [torch.LongTensor(text_lens), torch.LongTensor(audio_lens), torch.LongTensor(vision_lens)]
    )
    labels = torch.LongTensor(labels)
    return data, labels


# ============================================================
# 4. 训练与评估
# ============================================================

def split_dataset(data, train_r=0.7, val_r=0.1):
    """随机划分训练/验证/测试集"""
    np.random.seed(42)
    idx = np.random.permutation(len(data))
    n_train = int(len(data) * train_r)
    n_val = int(len(data) * val_r)
    return (
        [data[i] for i in idx[:n_train]],
        [data[i] for i in idx[n_train:n_train + n_val]],
        [data[i] for i in idx[n_train + n_val:]]
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for data, labels in tqdm(loader, desc="Train", leave=False):
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

    for data, labels in tqdm(loader, desc="Eval", leave=False):
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
    return total_loss / len(loader), acc, f1


# ============================================================
# 5. 模型构建
# ============================================================

def auto_build_model(data_sample, dataset_type, model_type, hidden_dim, device):
    """根据数据自动构建匹配维度的模型"""
    text_dim = data_sample['text'].shape[1]
    audio_dim = data_sample['audio'].shape[1]
    vision_dim = data_sample['vision'].shape[1]

    print(f"\nAuto-detected feature dimensions:")
    print(f"  Text: {text_dim}, Audio: {audio_dim}, Vision: {vision_dim}")

    # 根据输入维度选择合适的编码器配置
    encoder_hidden = {
        0: min(600, text_dim * 2),
        1: min(200, audio_dim * 2 + 2),
        2: min(200, vision_dim * 2 + 2)
    }

    encoders = [
        GRUWithLinear(text_dim, max(100, text_dim // 2), encoder_hidden[0],
                       dropout=1e-1, has_padding=True, batch_first=True),
        GRUWithLinear(audio_dim, max(100, audio_dim // 2), encoder_hidden[1],
                       dropout=1e-1, has_padding=True, batch_first=True),
        GRUWithLinear(vision_dim, max(100, vision_dim // 2), encoder_hidden[2],
                       dropout=1e-1, has_padding=True, batch_first=True)
    ]

    fusion = Concat(masks=[0, 1, 2])
    input_shape = sum(encoder_hidden[m] for m in [0, 1, 2])

    print(f"  Fusion input shape: {input_shape}")

    if model_type == 'split':
        head = QNNSplited(input_shape=input_shape, output_shape=7,
                          hidden_dim=hidden_dim, with_shortcut=True)
    else:
        head = QNNUnsplitted(input_shape=input_shape, output_shape=7,
                             hidden_dim=hidden_dim, with_shortcut=True)

    model = MMDL(encoders, fusion, head, has_padding=True)
    return model.to(device)


def build_fixed_model(dataset_type, model_type, hidden_dim, device):
    """使用与 example.py 一致的固定维度构建模型"""
    if dataset_type == 'mosi':
        encoder_hidden = {0: 300, 1: 300, 2: 400}
        encoders = [
            GRUWithLinear(35, 100, encoder_hidden[0], dropout=1e-1, has_padding=True, batch_first=True),
            GRUWithLinear(74, 300, encoder_hidden[1], dropout=1e-1, has_padding=True, batch_first=True),
            GRUWithLinear(300, 900, encoder_hidden[2], dropout=1e-1, has_padding=True, batch_first=True)
        ]
    else:
        encoder_hidden = {0: 600, 1: 200, 2: 200}
        encoders = [
            GRUWithLinear(713, 500, encoder_hidden[0], dropout=1e-1, has_padding=True, batch_first=True),
            GRUWithLinear(74, 300, encoder_hidden[1], dropout=1e-1, has_padding=True, batch_first=True),
            GRUWithLinear(300, 600, encoder_hidden[2], dropout=1e-1, has_padding=True, batch_first=True)
        ]

    fusion = Concat(masks=[0, 1, 2])
    input_shape = sum(encoder_hidden[m] for m in [0, 1, 2])

    if model_type == 'split':
        head = QNNSplited(input_shape=input_shape, output_shape=7,
                          hidden_dim=hidden_dim, with_shortcut=True)
    else:
        head = QNNUnsplitted(input_shape=input_shape, output_shape=7,
                             hidden_dim=hidden_dim, with_shortcut=True)

    model = MMDL(encoders, fusion, head, has_padding=True)
    return model.to(device)


# ============================================================
# 6. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='QD-MSA Training')
    parser.add_argument('--dataset', type=str, default='mosei', choices=['mosei', 'mosi'])
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='真实数据目录 (需提前手动下载 .csd 文件放入)')
    parser.add_argument('--synthetic', action='store_true',
                        help='使用模拟数据快速验证流程')
    parser.add_argument('--synthetic_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--model_type', type=str, default='split',
                        choices=['split', 'unsplit'])
    parser.add_argument('--auto_dim', action='store_true', default=True,
                        help='自动检测数据维度（推荐）')
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"QD-MSA Training")
    print(f"  Device:     {device}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Model:      {args.model_type}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print("=" * 60)

    # ---- 加载数据 ----
    if args.synthetic:
        print("\n>>> Using SYNTHETIC data (for pipeline verification)")
        data = generate_synthetic_data(args.synthetic_samples, args.dataset)
    else:
        data_dir = os.path.join(args.data_dir, args.dataset)
        cache_path = os.path.join(data_dir, 'processed.pkl')

        if os.path.exists(cache_path):
            print(f"\n>>> Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"\n>>> Downloading/processing real {args.dataset.upper()} data...")
            os.makedirs(data_dir, exist_ok=True)
            try:
                if args.dataset == 'mosei':
                    aligned = download_and_process_mosei(data_dir)
                else:
                    aligned = download_and_process_mosi(data_dir)
                data = process_aligned_data(aligned, args.dataset)
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Data cached to {cache_path}")
            except Exception as e:
                print(f"\n[ERROR] Real data loading failed: {e}")
                print("Falling back to synthetic data. Use --synthetic to skip download.")
                data = generate_synthetic_data(args.synthetic_samples, args.dataset)

    print(f"\nTotal samples: {len(data)}")

    # ---- 划分数据集 ----
    train_data, val_data, test_data = split_dataset(data)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_ds = MMSADataset(train_data)
    val_ds = MMSADataset(val_data)
    test_ds = MMSADataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    # ---- 构建模型 ----
    print("\n>>> Building model...")
    if args.synthetic or args.auto_dim:
        model = auto_build_model(data[0], args.dataset, args.model_type,
                                 args.hidden_dim, device)
    else:
        model = build_fixed_model(args.dataset, args.model_type,
                                  args.hidden_dim, device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_p:,}")
    print(f"  Trainable params: {train_p:,}")

    # ---- 训练 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0
    best_path = f"best_{args.dataset}_{args.model_type}.pt"

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"TL: {train_loss:.4f} TA: {train_acc:.4f} TF1: {train_f1:.4f} | "
              f"VL: {val_loss:.4f} VA: {val_acc:.4f} VF1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  >>> Best model (VA: {val_acc:.4f})")

    # ---- 测试 ----
    print("\n" + "=" * 60)
    print("Testing")
    print("=" * 60)
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  F1: {test_f1:.4f}")

    if args.synthetic:
        print("\n[INFO] Synthetic data results are for pipeline verification only.")
        print("       To train on real data, download CMU-MOSEI/MOSI .csd files and re-run without --synthetic.")

    print("\nDone!")


if __name__ == '__main__':
    main()
