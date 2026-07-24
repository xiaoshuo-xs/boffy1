"""
QD-MSA 快速对比测试 — 量子 MPS vs 经典 MLP
自生成数据，不需外部数据集，几分钟跑完
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score

from quantum_split_model import QNNSplited


# ============================================================
# 1. 生成合成分类数据（模拟多模态特征）
# ============================================================
def make_data(n_samples=1000, n_features=64, n_classes=3, seed=42):
    """生成简单的分类数据集，模拟融合后的多模态特征"""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    for c in range(n_classes):
        # 每个类的数据围绕不同的中心点
        center = rng.randn(n_features) * 2 + c * 1.5
        X_c = center + rng.randn(n_samples // n_classes, n_features) * 1.2
        X.append(X_c)
        y.append(np.full(n_samples // n_classes, c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    # 打乱
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ============================================================
# 2. 经典 MLP
# ============================================================
class ClassicalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. 量子 MPS 分类头
# ============================================================
class QuantumClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        # 经典特征降维 → 量子编码
        self.pre_quantum = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
        )
        self.quantum = QNNSplited(
            input_shape=input_dim // 2,
            output_shape=num_classes,
            hidden_dim=hidden_dim,
            with_shortcut=True,
        )

    def forward(self, x):
        x = self.pre_quantum(x)
        return self.quantum(x)


# ============================================================
# 4. 训练和评估
# ============================================================
def train_eval(model, train_loader, test_loader, device, epochs, lr, name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*55}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            correct += (out.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Loss: {train_loss:.3f} | "
              f"Acc: {correct/total:.4f} | "
              f"Time: {elapsed:.0f}s")

    # 测试
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y_batch.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    total_time = time.time() - t0
    print(f"\n  测试 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  测试 F1:       {f1:.4f}")
    print(f"  总用时:         {total_time:.0f}s ({total_time/60:.1f} min)")
    return acc, f1, total_time


# ============================================================
# 5. 主函数
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_str}")

    # 超参数（少量数据快速验证）
    N_TRAIN, N_TEST = 600, 200
    N_FEATURES, N_CLASSES = 64, 3
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-3

    print(f"\n数据: {N_TRAIN} train + {N_TEST} test | "
          f"特征: {N_FEATURES} dim | 类别: {N_CLASSES} | "
          f"Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")

    # 生成数据
    X, y = make_data(N_TRAIN + N_TEST, N_FEATURES, N_CLASSES)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    results = {}

    # ---- 经典 MLP ----
    classical = ClassicalMLP(N_FEATURES, 128, N_CLASSES)
    acc, f1, t = train_eval(
        classical, train_loader, test_loader, device, EPOCHS, LR, "Classical MLP")
    results['Classical MLP'] = {'acc': acc, 'f1': f1, 'time': t}

    # ---- 量子 MPS ----
    quantum = QuantumClassifier(N_FEATURES, N_CLASSES, hidden_dim=128)
    acc, f1, t = train_eval(
        quantum, train_loader, test_loader, device, EPOCHS, LR, "Quantum MPS")
    results['Quantum MPS'] = {'acc': acc, 'f1': f1, 'time': t}

    # ---- 汇总 ----
    print("\n" + "=" * 55)
    print("  最终对比")
    print("=" * 55)
    print(f"  {'模型':<20} {'Acc':<10} {'F1':<10} {'时间':<10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<20} {r['acc']:.4f}     {r['f1']:.4f}     {r['time']:.0f}s")

    # 量子优势分析
    q = results['Quantum MPS']
    c = results['Classical MLP']
    c_params = sum(p.numel() for p in classical.parameters())
    q_params = sum(p.numel() for p in quantum.parameters())
    print(f"\n  Classical params: {c_params:,}")
    print(f"  Quantum params:   {q_params:,}")
    print(f"  参数减少:         {(1 - q_params / c_params) * 100:.1f}%")
    print(f"  Quantum Acc 优势: {(q['acc'] - c['acc']) * 100:.2f} 百分点")
    print("=" * 55)


if __name__ == '__main__':
    main()
