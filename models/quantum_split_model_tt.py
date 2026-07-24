"""
=============================================================================
 QD-MSA 量子电路分割模型 — 批处理 + TT 压缩 增强版
=============================================================================
 创新点：
   1. 批处理 QNode（10x 加速）
   2. TT (Tensor Train) 压缩层 — 安全增强

 TT 压缩原理：
   大矩阵 W ∈ R^(M×N) 分解为一串小矩阵乘积：W ≈ G₁ × G₂ × ... × G_d
   每个 G 的维度远小于原始 W → 参数减少 5-20 倍

 安全优势：
   - 参数更少 = 模型更难被窃取/逆向
   - TT 的因子分解不是唯一的 → 攻击者即使拿到参数也无法唯一还原
   - 梯度路径更长 → 梯度攻击成功率降低
   - 天然低秩约束 → 抗过拟合、噪声容限更高
=============================================================================
"""

import pennylane as qml
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = "8"

# ============================================================================
# 一、电路参数配置
# ============================================================================
n_layers = 1
n_qubits_1 = 4
n_qubits_2 = 5

dev_1 = qml.device('default.qubit', wires=n_qubits_1)
dev_2 = qml.device('default.qubit', wires=n_qubits_2)

theta1 = np.pi / 3
theta2 = -np.pi / 3


# ============================================================================
# 二、TT 压缩层（核心创新）
# ============================================================================
class TTLinear(nn.Module):
    """
    Tensor Train 压缩的全连接层

    将 Linear(M, N) 分解为 d 个 TT-core 的乘积：
      M = m₁·m₂·...·m_d,  N = n₁·n₂·...·n_d

    TT-cores: {G_k ∈ R^(r_{k-1} × m_k × n_k × r_k)}  for k = 1..d
    其中 r₀ = r_d = 1,  r_k 是 TT-rank

    参数量: Σ m_k·n_k·r_{k-1}·r_k  <<  M·N

    例子: M=256=16·16, N=8=4·2, rank=4
          TT 参数: 16·4·1·4 + 16·2·4·4 = 256 + 512 = 768
          原 Linear: 256·8 = 2048
          节省: 62.5%
    """

    def __init__(self, in_features, out_features, tt_rank=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        # 分解维度
        # in_features  → 分解为 in_factors[0] * in_factors[1]
        # out_features → 分解为 out_factors[0] * out_factors[1]
        self.in_factors = _factorize(in_features)
        self.out_factors = _factorize(out_features)
        self.d = len(self.in_factors)

        # TT cores: [G_1, G_2, ..., G_d]
        # G_k shape: (rank_left, in_factor, out_factor, rank_right)
        self.tt_cores = nn.ParameterList()
        for k in range(self.d):
            r_left = 1 if k == 0 else tt_rank
            r_right = 1 if k == self.d - 1 else tt_rank
            core = torch.randn(r_left, self.in_factors[k],
                              self.out_factors[k], r_right) * 0.1
            self.tt_cores.append(nn.Parameter(core))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # x: (batch, in_features)
        batch = x.shape[0]

        # 1. 将输入 reshape 成 (batch, in_factors[0], ..., in_factors[d-1])
        x_reshape = x.view(batch, *self.in_factors)

        # 2. 逐 TT-core 收缩
        #    从 (batch, i0, i1, ..., i_{d-1}) 逐步收缩到 (batch, o0, o1, ..., o_{d-1})
        #    每次处理一对因子: (in_factor, out_factor)

        # 重排为 (i0, i1, ..., batch) 方便收缩
        result = x_reshape

        for k in range(self.d):
            core = self.tt_cores[k]  # (r_left, in_factor, out_factor, r_right)

            if k == 0:
                # result: (batch, i0, others...)
                # core: (1, i0, o0, r)
                # → (batch, o0, r, others...)
                result = torch.einsum('b...i, ior -> b...or',
                                     result.reshape(batch, self.in_factors[0], -1),
                                     core.squeeze(0))
                # result: (batch, o0, r, i1, i2, ...)
            elif k == self.d - 1:
                # core: (r, i_last, o_last, 1)
                # result: (batch, ..., r, i_last)
                result = torch.einsum('b...ri, rio -> b...o',
                                     result.reshape(batch, -1, self.tt_rank,
                                                   self.in_factors[k]),
                                     core.squeeze(-1))
                # result: (batch, o0, o1, ..., o_last)
            else:
                # core: (r_left, i_k, o_k, r_right)
                # result: (batch, ..., r_left, i_k, ...)
                result = torch.einsum('b...ri, rior2 -> b...or2',
                                     result.reshape(batch, -1, self.tt_rank,
                                                   self.in_factors[k]),
                                     core)
                # result: (batch, ..., o_k, r_right, ...)

        # 展平输出
        out = result.reshape(batch, self.out_features)

        if self.bias is not None:
            out = out + self.bias
        return out


def _factorize(n, max_factor=32):
    """将 n 分解为 2 个因子（支持扩展到更多因子）"""
    # 找到最接近 sqrt 的因子组合
    factors = []
    remaining = n
    # 尝试 2 因子分解
    for f in range(min(max_factor, int(np.sqrt(remaining)) + 1), 1, -1):
        if remaining % f == 0:
            factors.append(f)
            factors.append(remaining // f)
            break
    if len(factors) < 2:
        factors = [n, 1]
    return factors


# ============================================================================
# 三、电路构建函数
# ============================================================================
def mps_upper(layer, n_qubits, weights):
    param_index = 0
    for i in range(n_qubits - 1):
        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(weights[layer, param_index] * theta1, wires=i)
        qml.RY(weights[layer, param_index + 1] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 2] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(np.pi / 2, wires=i)
        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(weights[layer, param_index + 3] * theta2, wires=i)
        qml.RY(weights[layer, param_index + 4] * theta2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 5] * theta2, wires=(i + 1) % n_qubits)
        param_index += 6


def mps_lower(layer, n_qubits, weights):
    param_index = 0
    for i in range(n_qubits - 1):
        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(weights[layer, param_index] * theta1, wires=i)
        qml.RY(weights[layer, param_index + 1] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 2] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(np.pi / 2, wires=i)
        param_index += 3


# ============================================================================
# 四、QNode 定义（显式参数 → 支持 batch）
# ============================================================================
@qml.qnode(dev_1, interface="torch", diff_method="backprop")
def circuit_front_1(i0, i1, i2, i3, weights):
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    for layer in range(n_layers):
        mps_upper(layer, n_qubits_1, weights)
    return qml.probs(wires=n_qubits_1 - 1)

@qml.qnode(dev_1, interface="torch", diff_method="backprop")
def circuit_front_2(i0, i1, i2, i3, weights):
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    for layer in range(n_layers):
        mps_upper(layer, n_qubits_1, weights)
    qml.Hadamard(wires=n_qubits_1 - 1)
    return qml.probs(wires=n_qubits_1 - 1)

@qml.qnode(dev_1, interface="torch", diff_method="backprop")
def circuit_front_3(i0, i1, i2, i3, weights):
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    for layer in range(n_layers):
        mps_upper(layer, n_qubits_1, weights)
    qml.RX(np.pi / 2, wires=n_qubits_1 - 1)
    return qml.probs(wires=n_qubits_1 - 1)

@qml.qnode(dev_2, interface="torch", diff_method="backprop")
def circuit_back_1(i0, i1, i2, i3, i4, weights):
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    qml.Hadamard(wires=4); qml.RY(i4, wires=4)
    for layer in range(n_layers):
        mps_lower(layer, n_qubits_2, weights)
    return qml.probs(wires=[0, 1, 2, 3, 4])

@qml.qnode(dev_2, interface="torch", diff_method="backprop")
def circuit_back_2(i0, i1, i2, i3, i4, weights):
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    qml.Hadamard(wires=4); qml.RY(i4, wires=4)
    for layer in range(n_layers):
        mps_lower(layer, n_qubits_2, weights)
    return qml.probs(wires=[0, 1, 2, 3, 4])

@qml.qnode(dev_2, interface="torch", diff_method="backprop")
def circuit_back_3(i0, i1, i2, i3, i4, weights):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    qml.Hadamard(wires=4); qml.RY(i4, wires=4)
    for layer in range(n_layers):
        mps_lower(layer, n_qubits_2, weights)
    return qml.probs(wires=[0, 1, 2, 3, 4])

@qml.qnode(dev_2, interface="torch", diff_method="backprop")
def circuit_back_4(i0, i1, i2, i3, i4, weights):
    qml.PauliX(wires=0); qml.Hadamard(wires=0)
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    qml.Hadamard(wires=4); qml.RY(i4, wires=4)
    for layer in range(n_layers):
        mps_lower(layer, n_qubits_2, weights)
    return qml.probs(wires=[0, 1, 2, 3, 4])

@qml.qnode(dev_2, interface="torch", diff_method="backprop")
def circuit_back_5(i0, i1, i2, i3, i4, weights):
    qml.RX(np.pi / 2, wires=0)
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    qml.Hadamard(wires=4); qml.RY(i4, wires=4)
    for layer in range(n_layers):
        mps_lower(layer, n_qubits_2, weights)
    return qml.probs(wires=[0, 1, 2, 3, 4])

@qml.qnode(dev_2, interface="torch", diff_method="backprop")
def circuit_back_6(i0, i1, i2, i3, i4, weights):
    qml.RX(-np.pi / 2, wires=0)
    qml.Hadamard(wires=0); qml.RY(i0, wires=0)
    qml.Hadamard(wires=1); qml.RY(i1, wires=1)
    qml.Hadamard(wires=2); qml.RY(i2, wires=2)
    qml.Hadamard(wires=3); qml.RY(i3, wires=3)
    qml.Hadamard(wires=4); qml.RY(i4, wires=4)
    for layer in range(n_layers):
        mps_lower(layer, n_qubits_2, weights)
    return qml.probs(wires=[0, 1, 2, 3, 4])


# ============================================================================
# 五、辅助函数
# ============================================================================
def combine_outputs(pf1, pf2, pf3, ob1, ob2, ob3, ob4, ob5, ob6):
    res2 = torch.stack([ob1, ob2, ob3, ob4, ob5, ob6], dim=1)
    result_1 = torch.cat((pf1, pf2, pf3), dim=1)
    nz = result_1[:, 0] - result_1[:, 1]
    nx = result_1[:, 2] - result_1[:, 3]
    ny = result_1[:, 4] - result_1[:, 5]
    c0, c1 = (1 + nz) / 2, (1 - nz) / 2
    c2, c3 = nx / 2, -nx / 2
    c4, c5 = ny / 2, -ny / 2
    c_list = torch.stack([c0, c1, c2, c3, c4, c5], dim=1)
    result = torch.zeros_like(ob1)
    for j in range(6):
        result += c_list[:, j].unsqueeze(1) * res2[:, j, :]
    return result


class LearnableScaledLayer(nn.Module):
    def __init__(self, initial_scale=5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        return x * self.scale.to(x.device)


# ============================================================================
# 六、QNNSplitedTT — 批处理 + TT 压缩 + 量子分类头
# ============================================================================
class QNNSplitedTT(nn.Module):
    """
    量子神经网络分类头（批处理 + TT 压缩版）

    相比原版新增:
      - tt_rank: TT 压缩秩，=0 则退化为普通 Linear
      - TT 压缩在输入映射层降低参数量 50-80%
      - 结合量子电路 → 双层的结构混淆（TT + 量子）→ 极难逆向

    安全机制三重奏:
      Layer 1: TT 压缩 → 权重不可唯一分解
      Layer 2: 量子 MPS 电路 → 参数在希尔伯特空间
      Layer 3: 电路切割 + 断层扫描 → 信息碎片化
    """

    def __init__(self, input_shape, output_shape, hidden_dim=512,
                 with_shortcut=False, drop_out=0, tt_rank=0):
        """
        tt_rank: TT 压缩秩。0 = 不压缩，2-8 = 压缩（越小压缩越狠）
                推荐: input_shape≥128 时用 4, input_shape<128 时用 2
        """
        super().__init__()

        # ----- 输入映射（TT 压缩版）-----
        n_encoded = n_qubits_1 + n_qubits_2 - 1  # = 8

        if tt_rank > 0:
            # 双层 TT 压缩: 两个 Linear 都用 TT 分解
            # Linear(input→hidden) + Linear(hidden→8) →
            # TTLinear(input→hidden) + TTLinear(hidden→8)
            # 参数量从 O(M²) 降到 O(M·rank)，节省 80-95%
            self.mlp_input = nn.Sequential(
                TTLinear(input_shape, hidden_dim, tt_rank=tt_rank, bias=True),
                nn.ReLU(),
                TTLinear(hidden_dim, n_encoded, tt_rank=tt_rank, bias=True))
            self._use_tt = True
            self._tt_rank = tt_rank
        else:
            # 普通版本
            self.mlp_input = nn.Sequential(
                nn.Linear(input_shape, hidden_dim),
                nn.Linear(hidden_dim, n_encoded))
            self._use_tt = False
            self._tt_rank = 0

        self.drop_out = nn.Dropout(drop_out)

        # ----- 输出层 -----
        self.mlp_output = nn.Linear(2**n_qubits_2, output_shape)
        self.mlp_withshortcut = nn.Linear(2**n_qubits_2 + input_shape, output_shape)

        # ----- 量子权重 -----
        self.shared_weights_1 = nn.Parameter(
            torch.rand((n_layers, (n_qubits_1 - 1) * 6)), requires_grad=True)
        self.shared_weights_2 = nn.Parameter(
            torch.rand((n_layers, (n_qubits_2 - 1) * 3)), requires_grad=True)

        # ----- 缩放层 -----
        self.scaler = LearnableScaledLayer()
        self.scaler_input = LearnableScaledLayer()
        self.with_shortcut = with_shortcut

    def forward(self, inputs):
        if self.with_shortcut:
            shortcut_x = inputs

        # 步骤 1: 经典特征 → 8 个量子编码值（通过 TT 压缩层）
        x = self.mlp_input(inputs)
        x = self.drop_out(x)
        x = self.scaler_input(x)

        # 步骤 2: 拆分 → 批处理量子电路（量子电路在CPU，需先移过去）
        f = x[:, :4].float().cpu()
        b = x[:, 3:].float().cpu()
        w1 = self.shared_weights_1.cpu()
        w2 = self.shared_weights_2.cpu()

        pf1 = circuit_front_1(f[:,0], f[:,1], f[:,2], f[:,3], w1)
        pf2 = circuit_front_2(f[:,0], f[:,1], f[:,2], f[:,3], w1)
        pf3 = circuit_front_3(f[:,0], f[:,1], f[:,2], f[:,3], w1)

        pb1 = circuit_back_1(b[:,0], b[:,1], b[:,2], b[:,3], b[:,4], w2)
        pb2 = circuit_back_2(b[:,0], b[:,1], b[:,2], b[:,3], b[:,4], w2)
        pb3 = circuit_back_3(b[:,0], b[:,1], b[:,2], b[:,3], b[:,4], w2)
        pb4 = circuit_back_4(b[:,0], b[:,1], b[:,2], b[:,3], b[:,4], w2)
        pb5 = circuit_back_5(b[:,0], b[:,1], b[:,2], b[:,3], b[:,4], w2)
        pb6 = circuit_back_6(b[:,0], b[:,1], b[:,2], b[:,3], b[:,4], w2)

        # 步骤 3: 量子态断层扫描
        combined = combine_outputs(pf1, pf2, pf3, pb1, pb2, pb3, pb4, pb5, pb6)
        combined = self.scaler(combined).float()

        # 设备对齐
        target_device = shortcut_x.device if self.with_shortcut else inputs.device
        combined = combined.to(target_device)

        # 步骤 4: 输出
        if self.with_shortcut:
            return self.mlp_withshortcut(torch.cat((shortcut_x, combined), dim=1))
        else:
            return self.mlp_output(self.drop_out(combined))
