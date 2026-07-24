"""
=============================================================================
 公共模型组件 — 经典神经网络编码器
=============================================================================
 包含:
   GRU             — 门控循环单元（时序特征提取）
   GRUWithLinear   — GRU + 线性层（编码器主体）
   MMDL            — 多模态深度学习框架（编码→融合→分类）
   Concat          — 模态拼接融合层
=============================================================================
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


# =============================================================================
# 一、GRU 门控循环单元
# =============================================================================
class GRU(torch.nn.Module):
    """
    标准GRU，支持变长序列（通过 pack_padded_sequence）

    参数:
      indim:       输入特征维度
      hiddim:      GRU隐藏层维度
      dropout:     是否在GRU输出后加Dropout
      flatten:     是否展平输出
      has_padding: 输入是否包含padding（变长序列模式）
      last_only:   只返回最后一个时间步的输出
      batch_first: 输入是否batch在第一维 (batch, seq, feature)
    """

    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.1,
                 flatten=False, has_padding=False, last_only=False,
                 batch_first=True):
        super().__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):
        """
        输入: x — 如果has_padding=True, x = (data, lengths)
                        否则 x = tensor(batch, seq, feature)
        输出: GRU处理后的隐藏状态
        """
        if self.has_padding:
            # 变长序列模式: 用pack_padded_sequence压缩padding
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False)
            out = self.gru(x)[1][-1]  # 取最后层的隐藏状态
        elif self.last_only:
            out = self.gru(x)[1][0]
            return out
        else:
            out, _ = self.gru(x)

        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)

        return out


# =============================================================================
# 二、GRUWithLinear — 编码器主体
# =============================================================================
class GRUWithLinear(torch.nn.Module):
    """
    GRU + 线性投影层
    用于将变长时序特征压缩为固定长度向量

    输入: (batch, seq_len, indim) 或 带padding的元组
    输出: (batch, outdim) 固定长度编码向量

    参数:
      indim:    输入维度
      hiddim:   GRU隐藏维度
      outdim:   输出编码维度
      has_padding: 是否处理padding
      output_each_layer: 是否输出各中间层结果
    """

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1,
                 flatten=False, has_padding=False, output_each_layer=False,
                 batch_first=False):
        super().__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=batch_first)
        self.linear = nn.Linear(hiddim, outdim)         # 投影到输出维度
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.has_padding:
            # 变长序列: 先pack再送GRU, 取最后隐藏状态
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            hidden = self.gru(x)[1][-1]
        else:
            hidden = self.gru(x)[0]

        if self.dropout:
            hidden = self.dropout_layer(hidden)

        out = self.linear(hidden)  # 线性投影

        if self.flatten:
            out = torch.flatten(out, 1)

        if self.output_each_layer:
            return [0, torch.flatten(x, 1), torch.flatten(hidden, 1),
                    self.lklu(out)]
        return out


# =============================================================================
# 三、MMDL — 多模态深度学习框架
# =============================================================================
class MMDL(nn.Module):
    """
    MultiModal Deep Learning 标准框架:
      编码(Encoder) → 融合(Fusion) → 分类(Head)

    三个步骤:
      1. 各模态独立编码（GRUWithLinear）
      2. 编码向量拼接（Concat）
      3. 融合向量过分类头（量子或经典）
    """

    def __init__(self, encoders, fusion, head, has_padding=False):
        """
        encoders:  3个编码器列表 [text_enc, audio_enc, vision_enc]
        fusion:    融合层（如 Concat）
        head:      分类头（如 QNNSplited 或 MLP）
        has_padding: 输入是否带padding信息
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None  # 保存融合输出（调试用）
        self.reps = []       # 保存各模态编码（调试用）

    def forward(self, inputs):
        """
        输入格式:
          has_padding=True:  ([text, audio, vision], [t_len, a_len, v_len])
          has_padding=False: [text, audio, vision]

        输出: (batch, num_classes)
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))

        self.reps = outs  # 保存编码结果

        # 融合
        if self.has_padding:
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)

        self.fuseout = out  # 保存融合结果

        # 分类
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)


# =============================================================================
# 四、Concat — 模态拼接
# =============================================================================
class Concat(nn.Module):
    """
    多模态特征拼接融合
    将各模态编码向量在第1维（特征维）上拼接

    参数:
      masks: 选择哪些模态参与融合 [0,1,2] 表示全部三个模态
    """

    def __init__(self, masks=None):
        super().__init__()
        self.masks = masks  # None表示使用全部模态

    def forward(self, modalities):
        """
        modalities: [text_enc, audio_enc, vision_enc]
        每个都是 (batch, dim_i)
        返回: (batch, sum(dim_i))
        """
        if self.masks is None:
            masks = range(len(modalities))

        flattened = []
        for idx in self.masks:
            modality = modalities[idx]
            flattened.append(torch.flatten(modality, start_dim=1))

        return torch.cat(flattened, dim=1)