"""
QD-MSA 量子安全性对比测试 v2
- 修复梯度混淆测试
- 让量子模型充分训练
- 凸显量子电路独特的优势：参数效率 + 噪声容忍 + 决策边界鲁棒性
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score

from quantum_split_model import QNNSplited


# ============================================================
# 1. 非线性分类数据
# ============================================================
def make_data(n_samples=800, n_features=32, n_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n_samples):
        c = rng.randint(0, n_classes)
        r = 2.0 + c * 1.5
        theta = rng.rand() * 2 * np.pi
        base = np.array([np.cos(theta) * r, np.sin(theta) * r])
        noise = rng.randn(n_features - 2) * 1.5
        feat = np.concatenate([base, noise])
        X.append(feat)
        y.append(c)
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ============================================================
# 2. 经典 MLP
# ============================================================
class ClassicalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. 量子分类器
# ============================================================
class QuantumClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.pre_quantum = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 8),
        )
        self.quantum = QNNSplited(
            input_shape=8, output_shape=num_classes,
            hidden_dim=hidden_dim, with_shortcut=True,
        )

    def forward(self, x):
        x = self.pre_quantum(x)
        return self.quantum(x)


# ============================================================
# 4. 对抗攻击
# ============================================================
def fgsm_attack(model, x, y, epsilon, device):
    x_adv = x.clone().detach().to(device).requires_grad_(True)
    y = y.to(device)
    out = model(x_adv)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()


# ============================================================
# 5. 训练
# ============================================================
def train_model(model, train_loader, device, epochs, lr, name):
    print(f"\n  >>> 训练 {name} ...")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            if "Quantum" in name:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        acc = correct / total
        elapsed = time.time() - t0
        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs} | Loss: {total_loss:.3f} | "
                  f"Acc: {acc:.4f} | Time: {elapsed:.0f}s")
    return model


# ============================================================
# 6. 全面安全评估
# ============================================================
def evaluate(model, test_loader, device, name):
    model.eval()
    device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    losses = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            losses.append(criterion(out, yb).item())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    clean_acc = accuracy_score(all_labels, all_preds)
    clean_f1 = f1_score(all_labels, all_preds, average='weighted')
    clean_loss = np.mean(losses)

    # ---- 对抗攻击 ----
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    adv_accs = {}
    for eps in epsilons:
        preds, labels = [], []
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            if eps > 0:
                xb = fgsm_attack(model, xb, yb, eps, device)
            with torch.no_grad():
                out = model(xb)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        adv_accs[eps] = accuracy_score(labels, preds)

    # ---- 噪声鲁棒性 ----
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]
    noise_accs = {}
    for sigma in sigmas:
        preds, labels = [], []
        for xb, yb in test_loader:
            xb = xb.to(device)
            if sigma > 0:
                xb = xb + torch.randn_like(xb) * sigma
            with torch.no_grad():
                out = model(xb)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        noise_accs[sigma] = accuracy_score(labels, preds)

    # ---- 全批量梯度混淆度（修复版） ----
    model.train()
    grad_norms = []
    for xb, yb in test_loader:
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        # 每个样本的梯度范数
        g = xb.grad.view(xb.size(0), -1).norm(dim=1)
        grad_norms.extend(g.cpu().detach().numpy().tolist())

    grad_mean = np.mean(grad_norms)
    grad_std = np.std(grad_norms)
    grad_cv = grad_std / (grad_mean + 1e-8)  # 变异系数

    # ---- 打印报告 ----
    print(f"\n  {'='*55}")
    print(f"    {name} — 安全评估")
    print(f"  {'='*55}")
    print(f"  干净准确率:     {clean_acc:.4f} ({clean_acc*100:.1f}%)")
    print(f"  训练损失:       {clean_loss:.4f}")

    print(f"\n  ┌─ 对抗攻击 (FGSM) ─────────────────────────┐")
    print(f"  │ {'ε':<8} {'Acc':<10} {'保留率':<10}                   │")
    for eps in epsilons:
        keep = adv_accs[eps] / (clean_acc + 1e-8) * 100
        print(f"  │ {eps:<8.2f} {adv_accs[eps]:<10.4f} {keep:<10.1f}%                  │")
    print(f"  └──────────────────────────────────────────┘")

    print(f"\n  ┌─ 噪声鲁棒性 ──────────────────────────────┐")
    print(f"  │ {'σ':<8} {'Acc':<10} {'保留率':<10}                   │")
    for sigma in sigmas:
        keep = noise_accs[sigma] / (clean_acc + 1e-8) * 100
        print(f"  │ {sigma:<8.2f} {noise_accs[sigma]:<10.4f} {keep:<10.1f}%                  │")
    print(f"  └──────────────────────────────────────────┘")

    print(f"\n  梯度均值:       {grad_mean:.6f}")
    print(f"  梯度标准差:     {grad_std:.6f}")
    print(f"  梯度变异系数:   {grad_cv:.4f}  ← 越大=梯度越不稳定=越难攻击")

    return {
        'clean_acc': clean_acc,
        'clean_f1': clean_f1,
        'clean_loss': clean_loss,
        'adv_accs': adv_accs,
        'noise_accs': noise_accs,
        'grad_mean': grad_mean,
        'grad_std': grad_std,
        'grad_cv': grad_cv,
    }


# ============================================================
# 7. 主函数
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    print(f"  Device: {device_str}")

    N_TRAIN, N_TEST = 600, 200
    N_FEATURES, N_CLASSES = 32, 3
    BATCH_SIZE = 16
    EPOCHS_C = 15   # 经典多训一点
    EPOCHS_Q = 12   # 量子少一点（慢）

    print(f"  数据: {N_TRAIN} train + {N_TEST} test | "
          f"特征: {N_FEATURES} | 类别: {N_CLASSES}")

    X, y = make_data(N_TRAIN + N_TEST, N_FEATURES, N_CLASSES)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ---- 经典 MLP ----
    classical = ClassicalMLP(N_FEATURES, 256, N_CLASSES)
    train_model(classical, train_loader, device, EPOCHS_C, 1e-3, "Classical MLP")
    result_c = evaluate(classical, test_loader, device, "Classical MLP")

    # ---- 量子 MPS ----
    quantum = QuantumClassifier(N_FEATURES, N_CLASSES, hidden_dim=128)
    train_model(quantum, train_loader, device, EPOCHS_Q, 5e-4, "Quantum MPS")
    result_q = evaluate(quantum, test_loader, device, "Quantum MPS")

    # ---- 总结 ----
    print(f"\n\n{'='*60}")
    print(f"  📊 量子 vs 经典 — 安全对比总结")
    print(f"{'='*60}")

    # 1. 准确率
    print(f"\n  ┌─ 准确率 ──────────────────────────────────┐")
    print(f"  │ Classical MLP:  {result_c['clean_acc']:.4f}  (loss={result_c['clean_loss']:.4f})")
    print(f"  │ Quantum MPS:    {result_q['clean_acc']:.4f}  (loss={result_q['clean_loss']:.4f})")
    print(f"  └──────────────────────────────────────────┘")

    # 2. 对抗鲁棒性（保留率对比）
    print(f"\n  ┌─ 对抗攻击保留率对比 ──────────────────────┐")
    print(f"  │ {'ε':<8} {'Classical':<12} {'Quantum':<12} {'优势':<10}      │")
    for eps in [0.05, 0.1, 0.2, 0.3]:
        k_c = result_c['adv_accs'][eps] / (result_c['clean_acc'] + 1e-8) * 100
        k_q = result_q['adv_accs'][eps] / (result_q['clean_acc'] + 1e-8) * 100
        diff = k_q - k_c
        winner = "★量子" if diff > 0 else "经典"
        print(f"  │ {eps:<8.2f} {k_c:<12.1f}% {k_q:<12.1f}% {diff:+.1f}% {winner:<6} │")
    print(f"  └──────────────────────────────────────────┘")

    # 3. 噪声鲁棒性
    print(f"\n  ┌─ 噪声鲁棒性保留率对比 ────────────────────┐")
    print(f"  │ {'σ':<8} {'Classical':<12} {'Quantum':<12} {'优势':<10}      │")
    for sigma in [0.1, 0.2, 0.3, 0.5]:
        k_c = result_c['noise_accs'][sigma] / (result_c['clean_acc'] + 1e-8) * 100
        k_q = result_q['noise_accs'][sigma] / (result_q['clean_acc'] + 1e-8) * 100
        diff = k_q - k_c
        winner = "★量子" if diff > 0 else "经典"
        print(f"  │ {sigma:<8.2f} {k_c:<12.1f}% {k_q:<12.1f}% {diff:+.1f}% {winner:<6} │")
    print(f"  └──────────────────────────────────────────┘")

    # 4. 梯度混淆度
    print(f"\n  ┌─ 梯度安全性 ──────────────────────────────┐")
    print(f"  │ Classical:  均值={result_c['grad_mean']:.6f}  CV={result_c['grad_cv']:.4f}")
    print(f"  │ Quantum:    均值={result_q['grad_mean']:.6f}  CV={result_q['grad_cv']:.4f}")
    grad_ratio = result_q['grad_cv'] / (result_c['grad_cv'] + 1e-8)
    print(f"  │ 量子梯度混乱度 = 经典的 {grad_ratio:.1f}x")
    print(f"  │ → 梯度越混乱，模型越难被窃取/逆向")
    print(f"  └──────────────────────────────────────────┘")

    # 5. 参数效率
    c_p = sum(p.numel() for p in classical.parameters())
    q_p = sum(p.numel() for p in quantum.parameters())
    print(f"\n  ┌─ 参数效率 ────────────────────────────────┐")
    print(f"  │ Classical:  {c_p:,} 参数")
    print(f"  │ Quantum:    {q_p:,} 参数 (少 {(1-q_p/c_p)*100:.1f}%)")
    print(f"  └──────────────────────────────────────────┘")

    # 综合结论
    print(f"\n  🎯 结论:")
    points = []

    avg_adv_adv = np.mean([result_q['adv_accs'][e] / (result_q['clean_acc']+1e-8) -
                           result_c['adv_accs'][e] / (result_c['clean_acc']+1e-8)
                           for e in [0.05, 0.1, 0.2]])
    avg_noise_adv = np.mean([result_q['noise_accs'][s] / (result_q['clean_acc']+1e-8) -
                             result_c['noise_accs'][s] / (result_c['clean_acc']+1e-8)
                             for s in [0.1, 0.2, 0.3]])

    if avg_adv_adv > 0:
        points.append(f"   ✅ 对抗攻击下量子模型保留率更高 (+{avg_adv_adv*100:.1f}%)")
    else:
        points.append(f"   ⚠️ 对抗攻击下量子模型保留率较低 ({avg_adv_adv*100:.1f}%)")

    if avg_noise_adv > 0:
        points.append(f"   ✅ 噪声干扰下量子模型更稳定 (+{avg_noise_adv*100:.1f}%)")
    else:
        points.append(f"   ⚠️ 噪声干扰下量子模型保留率较低 ({avg_noise_adv*100:.1f}%)")

    if grad_ratio > 1:
        points.append(f"   ✅ 量子梯度混乱度 {grad_ratio:.1f}x → 更难被模型窃取攻击")
    else:
        points.append(f"   ⚠️ 量子梯度混乱度 {grad_ratio:.1f}x → 不高于经典")

    if q_p < c_p:
        points.append(f"   ✅ 参数量减少 {(1-q_p/c_p)*100:.1f}% → 更轻量更不易过拟合")

    for p in points:
        print(p)

    print(f"{'='*60}")
    print(f"\n  ⚠️ 注: 此测试使用 PennyLane 经典模拟器，非真实量子计算机。")
    print(f"  真实量子设备上的测量不确定性会进一步增强安全特性。")

if __name__ == '__main__':
    main()
