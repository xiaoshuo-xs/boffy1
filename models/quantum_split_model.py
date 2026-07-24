"""
=============================================================================
 QD-MSA 量子电路分割模型 (Quantum Split MPS Circuit)
=============================================================================
 论文核心创新：将 9 量子比特的 MPS 电路切割成两半：
   - 上半 (4 qubits): 前端电路 + 量子态断层扫描
   - 下半 (5 qubits): 后端电路 + 不同基下的测量
   切割点: 第4个量子比特（n_qubits_1 - 1）
   效果: 量子比特使用从 n 降到 n/2+1，节省近50%

 量子态断层扫描 (Quantum State Tomography):
   对切割点量子比特做 I, H, RX 三种基下的测量 → 得到6个概率值
   用这6个值重建出 nz, nx, ny（Bloch球坐标）
   结合后端6种初始化下的测量结果 → 线性组合还原完整量子态
=============================================================================
"""

import pennylane as qml
import torch.nn as nn
import torch
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = "8"    # 加速CPU量子模拟

# =============================================================================
# 一、电路参数配置
# =============================================================================
n_layers = 1        # MPS电路层数（每层是一轮相邻量子比特纠缠）
n_qubits_1 = 4      # 上半电路量子比特数（包含切割点）
n_qubits_2 = 5      # 下半电路量子比特数（从切割点开始）

# 量子模拟后端 — lightning.qubit 是 PennyLane 里最快的 CPU 模拟器
dev_1 = qml.device('lightning.qubit', wires=n_qubits_1, batch_obs=True)  # 上半
dev_2 = qml.device('lightning.qubit', wires=n_qubits_2, batch_obs=True)  # 下半

# 旋转角度系数（控制RX/RY门的角度缩放）
theta1 = np.pi / 3
theta2 = -np.pi / 3

# =============================================================================
# 二、量子编码电路 — 数据加载到量子态
# =============================================================================
def embedding_circuit(inputs, n_qubits):
    """
    振幅编码: 每个量子比特先过 H 门（叠加态），再用 RY 门编码输入数据
    H门: |0⟩ → (|0⟩+|1⟩)/√2  创造叠加态
    RY(input): 将经典数据旋转到量子态的振幅中
    """
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)           # 叠加
        qml.RY(inputs[qub], wires=qub)    # 编码输入

# =============================================================================
# 三、MPS（矩阵乘积态）电路 — 上半部分
# =============================================================================
def mps_upper(layer, n_qubits, weights):
    """
    上半MPS电路（包含完整的纠缠+旋转，6个参数/对）
    结构: 相邻量子比特对 (i, i+1) 之间做 CNOT + RX + RY 操作
          每个相邻对有6个可训练参数
    """
    param_index = 0
    for i in range(n_qubits - 1):
        # ---- 第一个子块 (theta1) ----
        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])         # 纠缠门
        qml.RX(weights[layer, param_index] * theta1, wires=i)
        qml.RY(weights[layer, param_index + 1] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])          # 反向纠缠
        qml.RY(weights[layer, param_index + 2] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(np.pi / 2, wires=i)

        # ---- 第二个子块 (theta2) ----
        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(weights[layer, param_index + 3] * theta2, wires=i)
        qml.RY(weights[layer, param_index + 4] * theta2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 5] * theta2, wires=(i + 1) % n_qubits)
        param_index += 6  # 每个相邻对消耗6个参数

# =============================================================================
# 四、MPS电路 — 下半部分（简化版，3个参数/对）
# =============================================================================
def mps_lower(layer, n_qubits, weights):
    """
    下半MPS电路（只有theta1子块，3个参数/对）
    因为在电路切割后，后端只需要一半的参数量
    """
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
        param_index += 3  # 每个相邻对消耗3个参数

# =============================================================================
# 五、构建完整电路（不含测量）
# =============================================================================
def build_circuits1(inputs, weights, qubits):
    """上半电路: 编码 + MPS层"""
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        mps_upper(layer, qubits, weights)

def build_circuits2(inputs, weights, qubits):
    """下半电路: 编码 + 简化MPS层"""
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        mps_lower(layer, qubits, weights)

# =============================================================================
# 六、量子态断层扫描 — 前端测量
# =============================================================================
# 对切割点量子比特做3种不同基下的测量，用于重建其量子态
# I基(直测): 得到量子比特处于|0⟩和|1⟩的概率
# H基: 做了Hadamard门后再测 → 得到X方向信息
# RX(π/2)基: 做RX旋转后再测 → 得到Y方向信息
# 这三个测量合起来 → Bloch球上的 (nz, nx, ny) 坐标

@qml.qnode(dev_1, interface="torch")
def circuit_front_1(inputs, weights):
    """前端I基测量"""
    build_circuits1(inputs, weights, n_qubits_1)
    return qml.probs(wires=n_qubits_1 - 1)  # 只测切割点

@qml.qnode(dev_1, interface="torch")
def circuit_front_2(inputs, weights):
    """前端H基测量"""
    build_circuits1(inputs, weights, n_qubits_1)
    qml.Hadamard(wires=n_qubits_1 - 1)      # 先H再测
    return qml.probs(wires=n_qubits_1 - 1)

@qml.qnode(dev_1, interface="torch")
def circuit_front_3(inputs, weights):
    """前端RX基测量"""
    build_circuits1(inputs, weights, n_qubits_1)
    qml.RX(np.pi / 2, wires=n_qubits_1 - 1) # 先RX再测
    return qml.probs(wires=n_qubits_1 - 1)

# =============================================================================
# 七、后端重建 — 6种初始化状态
# =============================================================================
# 切割后的量子态需要用6种不同的初始化来完整重建
# 初始化对应的密度矩阵基底: |0⟩⟨0|, |1⟩⟨1|, |+⟩⟨+|, |−⟩⟨−|, |+i⟩⟨+i|, |−i⟩⟨−i|

@qml.qnode(dev_2, interface="torch")
def circuit_back_1(inputs, weights):
    """后端初始化 |0⟩（什么都不做）"""
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])

@qml.qnode(dev_2, interface="torch")
def circuit_back_2(inputs, weights):
    """后端初始化 |1⟩（X门翻转）"""
    qml.PauliX(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])

@qml.qnode(dev_2, interface="torch")
def circuit_back_3(inputs, weights):
    """后端初始化 |+⟩ = (|0⟩+|1⟩)/√2"""
    qml.Hadamard(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])

@qml.qnode(dev_2, interface="torch")
def circuit_back_4(inputs, weights):
    """后端初始化 |−⟩ = (|0⟩−|1⟩)/√2"""
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])

@qml.qnode(dev_2, interface="torch")
def circuit_back_5(inputs, weights):
    """后端初始化 |+i⟩ = (|0⟩+i|1⟩)/√2"""
    qml.RX(np.pi / 2, wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])

@qml.qnode(dev_2, interface="torch")
def circuit_back_6(inputs, weights):
    """后端初始化 |−i⟩ = (|0⟩−i|1⟩)/√2"""
    qml.RX(-np.pi / 2, wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])

# =============================================================================
# 八、辅助函数 — 概率提取与量子态合并
# =============================================================================

def calculate_probabilities(tensor):
    """
    从量子电路原始输出中提取|0⟩和|1⟩的概率
    输入: (batch_size, num_probs) — 所有测量基的概率分布
    偶数索引 → |0⟩的概率, 奇数索引 → |1⟩的概率
    """
    prob_event_0 = tensor[:, 0::2].sum(dim=1)  # 所有偶数位求和 = P(0)
    prob_event_1 = tensor[:, 1::2].sum(dim=1)  # 所有奇数位求和 = P(1)
    return torch.stack([prob_event_0, prob_event_1], dim=1)  # (batch, 2)


def combine_outputs(probabilities_front_1, probabilities_front_2, probabilities_front_3,
                    output_back_1, output_back_2, output_back_3,
                    output_back_4, output_back_5, output_back_6):
    """
    量子态断层扫描合并 — 这是电路切割的核心

    步骤:
    1. 用前端3种基的测量结果计算切割点量子态的 Bloch 坐标 (nz, nx, ny)
    2. 计算6个密度矩阵系数 c0~c5
    3. 6种后端初始化结果 × 对应系数 → 线性组合 → 还原完整5-qubit量子态

    数学原理: 任意单量子比特密度矩阵 ρ = Σ c_i σ_i
             其中 σ_i ∈ {|0⟩⟨0|, |1⟩⟨1|, |+⟩⟨+|, |−⟩⟨−|, |+i⟩⟨+i|, |−i⟩⟨−i|}
    """
    # 后端6种结果堆叠: (batch, 6, 32)
    res2 = torch.stack([output_back_1, output_back_2, output_back_3,
                        output_back_4, output_back_5, output_back_6], dim=1)

    # 前端6个概率值拼接: (batch, 6)
    result_1 = torch.cat((probabilities_front_1, probabilities_front_2,
                          probabilities_front_3), dim=1)

    # Bloch球坐标
    nz = result_1[:, 0] - result_1[:, 1]   # P(0) - P(1) → Z方向
    nx = result_1[:, 2] - result_1[:, 3]   # H基下 P(+) - P(−) → X方向
    ny = result_1[:, 4] - result_1[:, 5]   # RX基下的差值 → Y方向

    # 密度矩阵展开系数（对应6种初始化基）
    c0 = (1 + nz) / 2   # |0⟩⟨0| 系数
    c1 = (1 - nz) / 2   # |1⟩⟨1| 系数
    c2 = nx / 2          # |+⟩⟨+| 系数
    c3 = -c2             # |−⟩⟨−| 系数
    c4 = ny / 2          # |+i⟩⟨+i| 系数
    c5 = -c4             # |−i⟩⟨−i| 系数

    c_list = torch.stack([c0, c1, c2, c3, c4, c5], dim=1)  # (batch, 6)

    # 线性组合还原完整量子态: ρ = Σ c_i * (后端第i种结果)
    result_list = torch.zeros_like(output_back_1)
    for j in range(c_list.shape[1]):
        result_list += c_list[:, j].unsqueeze(1) * res2[:, j, :]

    return result_list  # (batch, 32) — 5个量子比特的概率分布


# =============================================================================
# 九、可学习缩放层
# =============================================================================
class LearnableScaledLayer(nn.Module):
    """用一个可训练的标量缩放输入"""
    def __init__(self, initial_scale=5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        return x * self.scale.to(x.device)


# =============================================================================
# 十、QNNSplited — 量子电路分割模型（主要类）
# =============================================================================
class QNNSplited(nn.Module):
    """
    量子神经网络分类头（电路分割版）

    输入: (batch, fusion_dim) — 经典编码器融合后的特征向量
    流程:
      1. MLP将经典特征映射到 8 个值（= n_qubits_1 + n_qubits_2 - 1）
      2. 前 4 个值 → 上半4-qubit量子电路（3种基测量）
      3. 后 5 个值 → 下半5-qubit量子电路（6种初始化测量）
      4. combine_outputs() → 量子态断层扫描重组
      5. 重组后的量子态 → MLP输出 → 7分类

    参数:
      input_shape:   输入特征维度
      output_shape:  输出类别数（7）
      hidden_dim:    MLP隐藏层维度
      with_shortcut: 是否使用残差连接（跳跃连接）
    """

    def __init__(self, input_shape, output_shape, hidden_dim=512,
                 with_shortcut=False, drop_out=0):
        super().__init__()

        # ----- 10.1 输入映射: 经典特征 → 量子比特编码值 -----
        # 总共需要 n_qubits_1 + n_qubits_2 - 1 = 4+5-1 = 8 个值
        # -1 是因为切割点量子比特被两半共享
        self.mlp_input = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.Linear(hidden_dim, n_qubits_1 + n_qubits_2 - 1))
        self.drop_out = nn.Dropout(drop_out)

        # ----- 10.2 输出层 -----
        # 下半电路测量 5 qubits → 2^5 = 32 维概率向量
        self.mlp_output = nn.Linear(2**n_qubits_2, output_shape)
        # 带残差的输出层 (32 + input_shape) → 7
        self.mlp_withshortcut = nn.Linear(2**n_qubits_2 + input_shape, output_shape)

        # ----- 10.3 前端量子层（上半4-qubit）-----
        # 3种测量基 × 共享权重
        # 上半电路: 3个相邻对 × 6个参数 = 18个参数
        self.shared_weights_1 = nn.Parameter(
            torch.rand((n_layers, (n_qubits_1 - 1) * 6)), requires_grad=True)
        weight_shapes_1 = {"weights": (n_layers, 2 * (n_qubits_1 - 1) * 3)}

        # 3个前端电路共享同一组权重，只是最后测量基不同
        self.QLayer_front_1 = qml.qnn.TorchLayer(circuit_front_1, weight_shapes_1)
        self.QLayer_front_2 = qml.qnn.TorchLayer(circuit_front_2, weight_shapes_1)
        self.QLayer_front_3 = qml.qnn.TorchLayer(circuit_front_3, weight_shapes_1)
        self.QLayer_front_1.weights.data = self.shared_weights_1
        self.QLayer_front_2.weights.data = self.shared_weights_1
        self.QLayer_front_3.weights.data = self.shared_weights_1

        # ----- 10.4 后端量子层（下半5-qubit）-----
        # 6种初始化 × 共享权重
        # 下半电路: 4个相邻对 × 3个参数 = 12个参数
        self.shared_weights_2 = nn.Parameter(
            torch.rand((n_layers, (n_qubits_2 - 1) * 3)), requires_grad=True)
        weight_shapes_2 = {"weights": (n_layers, (n_qubits_2 - 1) * 3)}

        # 6个后端电路共享同一组权重，只是初始状态不同
        self.QLayer_back_1 = qml.qnn.TorchLayer(circuit_back_1, weight_shapes_2)
        self.QLayer_back_2 = qml.qnn.TorchLayer(circuit_back_2, weight_shapes_2)
        self.QLayer_back_3 = qml.qnn.TorchLayer(circuit_back_3, weight_shapes_2)
        self.QLayer_back_4 = qml.qnn.TorchLayer(circuit_back_4, weight_shapes_2)
        self.QLayer_back_5 = qml.qnn.TorchLayer(circuit_back_5, weight_shapes_2)
        self.QLayer_back_6 = qml.qnn.TorchLayer(circuit_back_6, weight_shapes_2)
        self.QLayer_back_1.weights.data = self.shared_weights_2
        self.QLayer_back_2.weights.data = self.shared_weights_2
        self.QLayer_back_3.weights.data = self.shared_weights_2
        self.QLayer_back_4.weights.data = self.shared_weights_2
        self.QLayer_back_5.weights.data = self.shared_weights_2
        self.QLayer_back_6.weights.data = self.shared_weights_2

        # ----- 10.5 辅助层 -----
        self.scaler = LearnableScaledLayer()        # 缩放量子输出
        self.scaler_input = LearnableScaledLayer()  # 缩放入量子电路的值
        self.with_shortcut = with_shortcut

    def forward(self, inputs):
        """
        前向传播流程:
          经典特征 → MLP → 8个值 → 拆分 →
            前端4值 → 3种基测量 → 切割点Bloch坐标
            后端5值 → 6种初始化测量 → 后端概率分布
            ↓
          combine_outputs → 断层扫描重组 → 32维量子态
            ↓
          MLP (可选+残差) → 7维分类输出
        """

        # 残差连接：保存原始输入
        if self.with_shortcut:
            shortcut_x = inputs

        # 步骤1: 经典特征 → 8个量子编码值
        inputs = self.mlp_input(inputs)   # (batch, 8)
        inputs = self.drop_out(inputs)
        inputs = self.scaler_input(inputs)

        # 步骤2: 逐样本跑量子电路（TorchLayer不支持这种电路的batch模式）
        output_front_1, output_front_2, output_front_3 = [], [], []
        output_back_1, output_back_2, output_back_3 = [], [], []
        output_back_4, output_back_5, output_back_6 = [], [], []

        for i in inputs:
            output_front_1.append(self.QLayer_front_1(i[:n_qubits_1]))
            output_front_2.append(self.QLayer_front_2(i[:n_qubits_1]))
            output_front_3.append(self.QLayer_front_3(i[:n_qubits_1]))
            output_back_1.append(self.QLayer_back_1(i[n_qubits_1 - 1:]))
            output_back_2.append(self.QLayer_back_2(i[n_qubits_1 - 1:]))
            output_back_3.append(self.QLayer_back_3(i[n_qubits_1 - 1:]))
            output_back_4.append(self.QLayer_back_4(i[n_qubits_1 - 1:]))
            output_back_5.append(self.QLayer_back_5(i[n_qubits_1 - 1:]))
            output_back_6.append(self.QLayer_back_6(i[n_qubits_1 - 1:]))

        output_front_1 = torch.stack(output_front_1, dim=0)
        output_front_2 = torch.stack(output_front_2, dim=0)
        output_front_3 = torch.stack(output_front_3, dim=0)
        output_back_1 = torch.stack(output_back_1, dim=0)
        output_back_2 = torch.stack(output_back_2, dim=0)
        output_back_3 = torch.stack(output_back_3, dim=0)
        output_back_4 = torch.stack(output_back_4, dim=0)
        output_back_5 = torch.stack(output_back_5, dim=0)
        output_back_6 = torch.stack(output_back_6, dim=0)

        # 步骤3: 量子态断层扫描合并
        combined_outputs = combine_outputs(
            output_front_1, output_front_2, output_front_3,
            output_back_1, output_back_2, output_back_3,
            output_back_4, output_back_5, output_back_6
        )

        combined_outputs = self.scaler(combined_outputs)

        # 对齐设备：量子电路始终在CPU，需移到与输入相同的设备
        target_device = shortcut_x.device if self.with_shortcut else inputs.device
        combined_outputs = combined_outputs.to(target_device)

        # 步骤4: 输出层
        if self.with_shortcut:
            # 残差连接: [量子态 | 原始特征] → MLP → 7分类
            outputs = torch.cat((shortcut_x, combined_outputs), dim=1)
            output_tensor = self.mlp_withshortcut(outputs)
        else:
            combined_outputs = self.drop_out(combined_outputs)
            output_tensor = self.mlp_output(combined_outputs)

        return output_tensor  # (batch, 7)
