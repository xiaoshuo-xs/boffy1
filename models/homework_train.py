
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')                     # 非交互式后端，不需要GUI窗口
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# 导入量子分割模型（批处理 + TT 压缩版）
from quantum_split_model_tt import QNNSplitedTT as QNNSplited


# =============================================================================
# 一、编码器：GRU 时序建模（论文原版方案，已修复NaN）
# =============================================================================
class GRUEncoder(nn.Module):
    """
    GRU 时序编码器 — 保留完整的词级时序信息
    输入：(batch, seq_len, feature_dim) + lengths
    输出：(batch, output_dim)

    和SimpleEncoder的区别：这里用双向/单向GRU对完整序列建模，
    保留时间维度的依赖关系，信息量远大于均值池化。
    NaN问题已通过在collate_fn中清理原始数据解决。
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True,
                          bidirectional=False, num_layers=1)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, h = self.gru(x)               # out: (B,T,H), h: (1,B,H)
        last = h[-1]                        # 取最后一层隐藏状态 (B,H)
        return self.bn(self.proj(last))     # (B, output_dim)


# =============================================================================
# 二、多模态融合模型
# =============================================================================
class MMDLSimple(nn.Module):
    def __init__(self, encoders, head):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.head = head

    def forward(self, x_list):
        encoded = [enc(x) for enc, x in zip(self.encoders, x_list)]
        fused = torch.cat(encoded, dim=1)
        return self.head(fused)


# =============================================================================
# 三、PyTorch数据集
# =============================================================================
class MMSADataset(Dataset):
    """
    多模态情感分析数据集
    每条数据包含文本、音频、视觉三个模态的时序特征和3类情感标签
    """

    def __init__(self, data, max_seq_len=60):
        """
        data:         预处理好的数据列表，每个元素有 text/audio/vision/label 字段
        max_seq_len:  截断的最大序列长度（太长的序列截断以加速训练）
        """
        self.samples = []
        for item in data:
            # 对齐三个模态的时间步，取最短的那个
            m_len = min(item['text'].shape[0], item['audio'].shape[0],
                        item['vision'].shape[0], max_seq_len)
            if m_len < 3:
                continue  # 太短的片段跳过
            self.samples.append({
                'text':   torch.FloatTensor(item['text'][:m_len]),
                'audio':  torch.FloatTensor(item['audio'][:m_len]),
                'vision': torch.FloatTensor(item['vision'][:m_len]),
                'label':  int(item['label']),
                # label范围: 0~6 对应情感 -3(极负) ~ +3(极正)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {'text': s['text'], 'audio': s['audio'],
                'vision': s['vision'], 'label': s['label']}


# =============================================================================
# 四、批处理函数 — 将不同长度的序列填充到统一长度
# =============================================================================
def collate_fn(batch):
    """
    将一个batch的样本填充对齐。
    每个模态独立填充到batch内最大长度（在seq_len维度填0）。
    同时清理原始数据中的NaN/Inf值（COVAREP音频数据自带NaN）。
    """
    t_b, a_b, v_b, lab_b = [], [], [], []
    for item in batch:
        t_b.append(item['text'])
        a_b.append(item['audio'])
        v_b.append(item['vision'])
        lab_b.append(item['label'])

    def pad(seqs):
        """把不等长的序列填充到同样长度"""
        m = max(s.shape[0] for s in seqs)      # 本batch最长序列
        d = seqs[0].shape[1]                    # 特征维度
        p = torch.zeros(len(seqs), m, d)         # 全零填充
        for i, s in enumerate(seqs):
            # 关键修复：清理原始COVAREP数据中的NaN
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            p[i, :s.shape[0], :] = s
        return p

    return [pad(t_b), pad(a_b), pad(v_b)], torch.LongTensor(lab_b)


# =============================================================================
# 五、训练流程 — 一个通用的训练+验证+测试函数
# =============================================================================
def run_experiment(model, train_loader, val_loader, test_loader,
                   device, epochs, lr, name):
    """
    完整的模型训练管线

    参数:
        model:        训练模型
        train_loader: 训练数据加载器
        val_loader:   验证数据加载器
        test_loader:  测试数据加载器
        device:       'cuda' 或 'cpu'
        epochs:       训练轮数
        lr:           学习率
        name:         模型名称（用于显示和保存）

    返回:
        包含 test_acc, test_f1, best_val_acc, history, preds, labels 的字典
    """
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    # ---- 5.1 类别权重 — 处理数据不平衡 ----
    # 样本少的类给更高权重，迫使模型关注少数类
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.extend(labels.numpy())
    counts = np.bincount(all_train_labels, minlength=3)
    # 权重 = 1/(样本数+1)，然后归一化到和为7
    weights = 1.0 / (counts.astype(float) + 1)
    weights = weights / weights.sum() * 3
    weights = torch.FloatTensor(weights).to(device)

    # ---- 5.2 损失函数 & 优化器 ----
    criterion = nn.CrossEntropyLoss(weight=weights)  # 带权重的交叉熵
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # CosineAnnealingWarmRestarts: 学习率周期性重置，有助于跳出局部最优
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs // 4, T_mult=2)

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}

    # ---- 5.3 训练循环 ----
    for epoch in range(1, epochs + 1):
        # === 训练阶段 ===
        model.train()
        train_loss, train_preds, train_labels = 0, [], []

        for data, labels in tqdm(train_loader, desc=f"  Epoch {epoch:2d}", leave=False):
            x_list = [d.to(device) for d in data]        # 数据移到GPU
            labels = labels.to(device)

            optimizer.zero_grad()                          # 清空梯度
            outputs = model(x_list)                        # 前向传播
            loss = criterion(outputs, labels)              # 计算损失
            loss.backward()                                # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪防爆炸
            optimizer.step()                               # 更新参数

            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()  # 学习率调度

        # === 验证阶段 ===
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():  # 验证不需要梯度
            for data, labels in val_loader:
                x_list = [d.to(device) for d in data]
                labels = labels.to(device)
                outputs = model(x_list)
                val_loss += criterion(outputs, labels).item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        # 记录历史
        for key, val in zip(history.keys(), [train_loss, train_acc, val_loss, val_acc, val_f1]):
            history[key].append(val)

        marker = " <<< BEST" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:2d} | Loss:{train_loss:8.2f} "
              f"TrainAcc:{train_acc:.4f} | ValAcc:{val_acc:.4f} F1:{val_f1:.4f}{marker}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{name.replace(" ","_")}.pt')

    # ---- 5.4 测试阶段（加载最佳模型） ----
    model.load_state_dict(torch.load(f'best_{name.replace(" ","_")}.pt',
                                     weights_only=True))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            x_list = [d.to(device) for d in data]
            outputs = model(x_list)
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    # 打印分类报告（只显示存在的类别）
    unique = sorted(set(test_labels) | set(test_preds))
    emotion_names = ['负面','中性','正面']
    print(f"\n  测试准确率(Accuracy): {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  测试F1分数(Weighted): {test_f1:.4f}")
    print(classification_report(
        test_labels, test_preds, labels=unique,
        target_names=[emotion_names[i] for i in unique], zero_division=0))

    return {'name': name, 'test_acc': test_acc, 'test_f1': test_f1,
            'best_val_acc': best_val_acc, 'preds': test_preds,
            'labels': test_labels, 'history': history}


# =============================================================================
# 六、画对比图
# =============================================================================
def plot_results(results, epochs, n_train, device_str):
    """
    生成包含两张子图的对比图:
      左图 — 训练过程曲线（train/val accuracy 随 epoch 变化）
      右图 — 测试结果柱状图（accuracy 和 F1 对比）

    results: run_experiment返回的字典
    """
    fig = plt.figure(figsize=(18, 7))

    # ── 左图：训练曲线 ──
    ax1 = fig.add_subplot(1, 3, (1, 2))  # 左图占2份宽度
    colors = {'Classical MLP': '#2196F3', 'Quantum Split MPS': '#E91E63'}
    for name, r in results.items():
        h = r['history']
        eps = range(1, len(h['train_acc']) + 1)
        c = colors.get(name, '#333')
        # 虚线 = 训练准确率，透明度低
        ax1.plot(eps, h['train_acc'], '--', color=c, alpha=0.35, linewidth=1.2)
        # 实线 = 验证准确率，主视觉
        ax1.plot(eps, h['val_acc'], '-', color=c, linewidth=2.2, label=f'{name}')
        # 标记验证准确率最高点
        best_idx = np.argmax(h['val_acc'])
        ax1.scatter(best_idx + 1, h['val_acc'][best_idx], color=c,
                    s=60, zorder=5, edgecolors='white', linewidth=1)

    # 随机基线：3分类随机猜 ≈ 33.3%
    ax1.axhline(y=0.333, color='gray', linestyle=':', alpha=0.5,
                label='Random baseline (33.3%)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Training Curves  |  {device_str}  |  '
                  f'{n_train} samples  |  {epochs} epochs',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 1.05)

    # ── 右图：测试结果柱状图 ──
    ax2 = fig.add_subplot(1, 3, 3)  # 右图占1份宽度
    nlist = [r['name'] for r in results.values()]
    accs = [r['test_acc'] * 100 for r in results.values()]
    f1s = [r['test_f1'] * 100 for r in results.values()]
    x = np.arange(len(nlist))
    w = 0.32

    b1 = ax2.bar(x - w/2, accs, w, label='Accuracy (%)',
                 color='#2196F3', edgecolor='white', linewidth=1.2)
    b2 = ax2.bar(x + w/2, f1s, w, label='F1 Score (%)',
                 color='#E91E63', edgecolor='white', linewidth=1.2)

    # 在柱子上方标注数值
    for b in b1:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                 f'{b.get_height():.1f}%', ha='center', fontsize=13,
                 fontweight='bold', color='#1565C0')
    for b in b2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                 f'{b.get_height():.1f}%', ha='center', fontsize=13,
                 fontweight='bold', color='#AD1457')

    ax2.set_xticks(x)
    ax2.set_xticklabels(nlist, fontsize=10)
    ax2.set_ylabel('Score (%)', fontsize=12)
    ax2.set_title('Test Set Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_ylim(0, max(max(accs), max(f1s)) + 15)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d%%'))

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=180,
                bbox_inches='tight', facecolor='white')
    print(f"\n  图表已保存: comparison_results.png")


# =============================================================================
# 七、主函数
# =============================================================================
def main():
    print("=" * 70)
    print("    QD-MSA 对比实验")
    print("    量子 MPS 电路 vs 经典 MLP — 多模态情感分析")
    print("=" * 70)

    # ---- 7.1 设备检测 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = (f"GPU: {torch.cuda.get_device_name(0)}"
                  if torch.cuda.is_available() else "CPU")

    # ---- 7.2 超参数 ----
    n_train, n_val, n_test = 16326, 1871, 4659   # 训练/验证/测试样本数
    batch_size = 32
    epochs = 80                                 # 训练轮数
    lr = 8e-4                                   # 学习率

    print(f"\n  Device: {device_str}")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Data: Train={n_train}  Val={n_val}  Test={n_test}")

    # ---- 7.3 加载数据 ----
    print("\n[1/3] Loading CMU-MOSEI data...")
    with open('D:/qdmsa_data/mosei_star/processed_v2.pkl', 'rb') as f:
        raw = pickle.load(f)
    print(f"  Total dataset: {len(raw)} segments")

    # 随机划分训练/验证/测试集
    np.random.seed(42)
    idx = np.random.permutation(len(raw))
    train_raw = [raw[i] for i in idx[:n_train]]
    val_raw   = [raw[i] for i in idx[n_train:n_train + n_val]]
    test_raw  = [raw[i] for i in idx[n_train + n_val:n_train + n_val + n_test]]

    train_ds = MMSADataset(train_raw)
    val_ds   = MMSADataset(val_raw)
    test_ds  = MMSADataset(test_raw)
    print(f"  After filtering: Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # 自动检测特征维度
    sample = train_ds[0]
    td, ad, vd = (sample['text'].shape[1], sample['audio'].shape[1],
                  sample['vision'].shape[1])
    print(f"  Feature dims: Text={td}  Audio={ad}  Vision={vd}")
    # 典型值: Text=300(GloVe), Audio=74(COVAREP), Vision=35(Facet)

    # ---- 7.4 构建编码器 ----
    # 文本信息量最大 → 输出128维
    # 音频和视觉 → 各输出64维
    # 融合后共 128+64+64 = 256 维
    enc_out = {0: 128, 1: 64, 2: 64}
    input_shape = sum(enc_out.values())
    print(f"  Fusion dim: {input_shape}")

    def make_encoders():
        return [
            GRUEncoder(td, 128, enc_out[0]),   # 文本GRU编码器
            GRUEncoder(ad,  64, enc_out[1]),   # 音频GRU编码器
            GRUEncoder(vd,  64, enc_out[2]),   # 视觉GRU编码器
        ]

    results = {}

    # ---- 7.5 实验1：经典MLP基线 ----
    print("\n[2/3] Training Classical MLP (baseline)...")
    # 三层全连接网络作为对比基线
    classical_head = nn.Sequential(
        nn.Linear(input_shape, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 3)  # 3分类输出
    )
    classical = MMDLSimple(make_encoders(), classical_head).to(device)
    print(f"  Classical model params: {sum(p.numel() for p in classical.parameters()):,}")
    results['Classical MLP'] = run_experiment(
        classical, train_loader, val_loader, test_loader,
        device, epochs, lr, "Classical MLP")

    # ---- 7.6 实验2：量子MPS电路 ----
    print("\n[3/3] Training Quantum Split MPS...")
    # QNNSplited: 量子电路分割模型
    # 将9个量子比特拆成 4+5 上下两块，通过量子态断层扫描合并结果
    # 量子比特使用从 n 降到 n/2+1，节省近50%
    quantum_head = QNNSplited(
        input_shape=input_shape,   # 融合特征维度
        output_shape=3,            # 7分类
        hidden_dim=256,            # 量子电路隐层维度
        with_shortcut=True,        # 残差连接（特征直连到输出层）
        tt_rank=0                  # ★ TT 压缩秩 → 参数适度压缩
    )
    quantum = MMDLSimple(make_encoders(), quantum_head).to(device)
    print(f"  Quantum model params: {sum(p.numel() for p in quantum.parameters()):,}")
    results['Quantum Split MPS'] = run_experiment(
        quantum, train_loader, val_loader, test_loader,
        device, epochs, lr, "Quantum Split MPS")

    # ---- 7.7 汇总结果 ----
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Model':<25} {'Test Acc':<12} {'Test F1':<12} {'Best Val':<12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for name, r in results.items():
        print(f"  {name:<25} {r['test_acc']:<12.4f} "
              f"{r['test_f1']:<12.4f} {r['best_val_acc']:<12.4f}")

    c_params = sum(p.numel() for p in classical.parameters())
    q_params = sum(p.numel() for p in quantum.parameters())
    print(f"\n  参数效率: 经典={c_params:,}  量子={q_params:,}  "
          f"(量子少 {(1 - q_params / c_params) * 100:.1f}%)")
    print(f"  论文 QD-MSA: ~80%+ on full CMU-MOSEI (GPU, 50+ epochs)")
    print(f"  本次实验: {device_str}, {n_train} samples, {epochs} epochs")
    print("=" * 70)

    # ---- 7.8 画对比图 ----
    plot_results(results, epochs, n_train, device_str)


if __name__ == '__main__':
    main()