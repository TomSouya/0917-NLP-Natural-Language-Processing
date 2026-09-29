# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   2.py
# FileName:      6.py
# Author:       Souya Tom
# Datetime:     2026/9/22 14:46
# Software: 	PyCharm
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import torch
#
# data1 = torch.ones(3, 5, 5).float()
# data2 = torch.zeros(3, 5, 5).float()
# matrix = torch.full_like(data1, 10)
# l_res = torch.tril(data1, diagonal=-1)
# u_res = torch.triu(data1, diagonal=0)
# new_l_res = 1 - l_res
#
# print(u_res)
# print("------------------------")
# print(new_l_res)
#
# data3 = torch.rand(1, 1, 5)
# print(data3.squeeze(-1).shape)

import copy
import time
import math
from matplotlib import pyplot as plt

torch.manual_seed(992)


class Embedding(torch.nn.Module):
    """
    功能：获取每句话中词的词嵌入向量
    实现原理：
    1. 使用可学习的嵌入矩阵将离散的词索引映射为连续的向量空间表示
    2. 对输出进行缩放处理，使数值尺度与后续模块（如位置编码）相匹配
    """

    def __init__(self, vocab, d_model):
        """
        :param vocab: 词表大小（词汇量）
        :param d_model: 词向量维度（嵌入维度）
        初始化说明：
        - 创建形状为[vocab_size, d_model]的嵌入矩阵
        - 矩阵参数会通过优化器在训练过程中自动更新
        """
        super().__init__()
        self.d_model = d_model
        # 创建嵌入层：将词索引映射到d_model维向量空间
        # 嵌入矩阵参数会通过反向传播自动学习
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab,
            embedding_dim=self.d_model
        )

    def forward(self, x):
        """
        :param x: 输入张量，形状为[batch_size, seq_len]，包含词索引
        :return: 缩放后的词嵌入向量，形状为[batch_size, seq_len, d_model]

        缩放处理说明：
        1. 乘以sqrt(d_model)是为了平衡嵌入向量和位置编码的数值范围
        2. 这样设计的依据来自论文《Attention Is All You Need》：
           - 当d_model较大时，未经缩放的嵌入向量可能导致梯度消失
           - 经验表明这种缩放有助于稳定训练过程
        3. 使用math.sqrt替代torch.sqrt是为了避免创建额外的计算图节点
        """
        # 需要消除词嵌入向量量纲
        # 对词嵌入向量进行缩放，使其数值尺度与后续模块匹配。
        # return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(torch.nn.Module):
    """
    功能：给每个词分配一个位置，因为transformer结构中没有循环网络，记不住词之间的位置信息
    实现原理：
    1. 使用正弦和余弦函数生成固定的位置编码（Positional Encoding, PE）
    2. 将PE与词嵌入向量相加，为模型提供序列中词的位置信息
    3. 通过Dropout防止过拟合
    """

    def __init__(self, d_model, max_len=5000, p=0.1):
        """
        :param d_model: 词向量维数（即嵌入维度）
        :param max_len: 一句话中词的最大个数（预生成最大长度的位置编码）
        :param p: Dropout的概率
        """
        super().__init__()

        # 1. 初始化PE矩阵: [max_len, d_model]
        # requires_grad=False 表示PE在训练过程中不更新
        pe = torch.zeros(max_len, d_model, requires_grad=False, dtype=torch.float32)

        # 2. 构建位置序列 pos: [max_len, 1]
        # 例如当max_len=5000时，pos = [[0], [1], [2], ..., [4999]]
        pos = torch.arange(0, max_len, 1).unsqueeze(1)

        # 3. 计算角度分母 div_term: [d_model//2]
        # 公式：div_term = 10000^(-2i/d_model)，其中i是维度索引
        # 通过指数函数和对数运算实现幂运算
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * - (torch.log(torch.tensor(10000.0)) / d_model)
        )

        # 4. 计算位置-维度的乘积: [max_len, d_model//2]
        # pos_val[i,j] = pos[i] * div_term[j]
        pos_val = pos * div_term

        # 5. 奇偶维度分别填充sin和cos
        # 偶数索引列（0,2,4...）填入sin(pos_val)
        pe[:, ::2] = torch.sin(pos_val)  # [max_len, d_model] 的偶数列
        # 奇数索引列（1,3,5...）填入cos(pos_val)
        pe[:, 1::2] = torch.cos(pos_val)  # [max_len, d_model] 的奇数列

        # 6. 扩展PE矩阵维度为 [1, max_len, d_model]
        # 第一个维度是batch维度（广播兼容性）
        pe = pe.unsqueeze(0)  # 1, max_len, d_model

        # 7. 注册为buffer，不会被优化器更新，但会随模型移动到GPU
        self.register_buffer("PE", pe)

        # 8. 创建Dropout层
        self.dropout = torch.nn.Dropout(p)

    def forward(self, data):
        """
        :param data: 输入张量，形状为 [batch_size, seq_len, d_model]
        :return: 添加位置编码后的张量，形状与输入相同
        """
        # 1. 截取与输入序列长度匹配的PE部分
        # self.PE[:, :data.size(1)] 形状: [1, seq_len, d_model]
        # 通过广播机制与data相加：[batch_size, seq_len, d_model] + [1, seq_len, d_model]
        # 2. 应用Dropout
        return self.dropout(data + self.PE[:, :data.size(1)])


class LayerNorm(torch.nn.Module):
    """
    功能：对输入数据进行标准化的映射，再进行缩放和偏移。目的：稳定数据，将数据缩放在合理区间
    """

    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        # 创建gram和beta
        # 在torch.nn.Parameter中的tensor在后续的计算过程中，会更新
        self.gram = torch.nn.Parameter(torch.ones(d_model, requires_grad=True, dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.zeros(d_model, requires_grad=True, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, data):
        """
        :param data: 输入张量，形状为[batch_size, seq_len, d_model]
        :return: 标准化并缩放后的张量，形状与输入相同

        标准化过程详解：
        1. 计算均值μ：在最后一个维度（d_model）上计算均值
           - dim=-1 表示对每个样本的特征维度进行标准化
           - keepdim=True 保持维度一致性（用于广播）
        2. 计算标准差σ：同上，在最后一个维度计算标准差
        3. 标准化：(data - μ)/σ
        4. 仿射变换：γ * 标准化结果 + β
        """
        # 获取输入数据的均值和标准差  # 3, 5, 512
        mean = torch.mean(data, dim=-1, keepdim=True)  # 3, 5, 1
        std = torch.std(data, dim=-1, keepdim=True)  # 3, 5, 1

        # 标准化公式：y = γ * (x - μ)/σ + β
        # ε防止除零错误
        return self.gram * ((data - mean) / (std + self.epsilon)) + self.beta


def self_attention(query, key, value, mask=False, dropout=None):
    """
    进行自注意力机制的计算，中间带有掩码
    softmax(QK^T/sqrt(d_k))V
    :param query: 查询张量，形状为[batch_size, head, seq_len, d_k]
    :param key: 索引张量，形状为[batch_size, head, seq_len, d_k]
    :param value: 数据张量，形状为[batch_size, head, seq_len, d_k]
    :param mask: 是否应用掩码（用于解码时的未来位置遮蔽）
    :param dropout: Dropout层对象（用于注意力权重随机失活）
    :return:
        - 注意力计算结果: 形状[batch_size, head, seq_len, d_k]
        - 注意力权重: 形状[batch_size, head, seq_len, seq_len]
    """

    # 1. 获取维度参数 d_k
    # d_k 是每个注意力头的维度（query 最后一维的大小）
    d_k = query.size(-1)

    # 2. 计算注意力分数 QK^T / sqrt(d_k)
    # 公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    # 实现细节：
    # - key.transpose(-1, -2) 将 key 的最后两个维度转置（seq_len, d_k -> d_k, seq_len）
    # - 矩阵乘法后结果形状：[batch_size, head, seq_len, seq_len]
    # - 除以 sqrt(d_k) 是为了防止内积过大导致 softmax 饱和（缩放点积注意力）
    score = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k))

    # 3. 应用掩码（仅在解码阶段使用）
    if mask:
        # a. 创建掩码矩阵（上三角矩阵）
        # - torch.ones_like(score) 创建与 score 同形状的全1矩阵
        # - torch.triu(matrix, diagonal=1) 保留主对角线及以上的元素，其余置0
        # - 1 - ... 将上三角部分置0，下三角部分置1（形成下三角掩码）
        mask_matrix = torch.ones_like(score)
        mask_matrix = 1 - torch.triu(mask_matrix, diagonal=1)

        # b. 应用掩码：将被掩码的位置设为极小值（-1e9）
        # - masked_fill(condition, value) 将 condition 为 True 的位置设为 value
        # - 这里 mask_matrix == 0 表示需要被掩码的位置（未来位置）
        score = score.masked_fill(mask_matrix == 0, -1e9)

    # 4. 计算注意力权重
    # - 在最后一个维度（seq_len）进行 softmax 归一化
    # - 结果表示每个位置对其他位置的关注程度
    att_weight = torch.softmax(score, dim=-1)

    # 5. 应用 Dropout（训练时随机失活注意力权重）
    if dropout is not None:
        att_weight = dropout(att_weight)

    # 6. 计算最终输出：注意力权重与 value 相乘
    # - att_weight [batch_size, head, seq_len, seq_len]
    # - value [batch_size, head, seq_len, d_k]
    # - 矩阵乘法后结果形状：[batch_size, head, seq_len, d_k]
    return torch.matmul(att_weight, value), att_weight


class MultiHeadAttention(torch.nn.Module):
    """
    功能：多头自注意力，计算输入数据，词和词之间的关系的特征。将以前的自注意力计算分成多个head(头)的计算
    实现原理：
    1. 将输入数据线性投影到多个子空间（多头）
    2. 每个头独立计算自注意力
    3. 将结果拼接并通过线性层映射回原始维度
    数学公式：
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    其中 head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    """

    def __init__(self, d_model, head, p=0.1):
        """
        :param d_model: 输入数据的最后一维的维数
        :param head: 分成的头数
        :param p: 概率
        初始化说明：
        - 断言检查d_model能被head整除（确保每个头的维度是整数）
        - 每个头的维度d_k = d_model / head
        - 创建三个线性层分别生成Q/K/V
        - 输出层将多头结果映射回原始维度
        """
        super().__init__()
        # 校验输入数据的最后一维的维数，分成的头数能够整除
        assert d_model % head == 0
        # 定义每个头的输出维数
        self.d_k = d_model // head
        # 定义头数
        self.head = head
        # 定义三个线性层，获取QKV
        self.linear_q = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = torch.nn.Linear(in_features=d_model, out_features=d_model)
        # 输出层
        self.out = torch.nn.Linear(in_features=d_model, out_features=d_model)
        # 随机识活
        self.dropout = torch.nn.Dropout(p)
        # 定义属性，主要接收注意力权重
        self.att_weight = None

    def forward(self, query, key, value, mask=False):
        """
        :param query: 输入的数据，形状[batch_size, seq_len, d_model]
        :param key: 输入的数据，形状[batch_size, seq_len, d_model]
        :param value: 输入的数据，形状[batch_size, seq_len, d_model]
        :param mask: 控制自注意力机制中使用进行掩码
        :return: 多头注意力输出，形状[batch_size, seq_len, d_model]

        维度转换详解：
        1. 输入形状：[batch_size, seq_len, d_model]
        2. 线性变换后：[batch_size, seq_len, d_model]
        3. reshape后：[batch_size, seq_len, head, d_k]
        4. 转置后：[batch_size, head, seq_len, d_k]
        5. 注意力计算后：[batch_size, head, seq_len, d_k]
        6. 重组后：[batch_size, seq_len, d_model]
        """
        # 获取输入的数据的batch_size
        batch_size = query.size(0)

        # 根据输入的数据，经过线性变换，获取对应的QKV
        # 输入数据的形状为：3, 5, 512--->3, 5, 512--->将获取到的qkv进行多头的拆分-->3, 5, head, d_k-->batch_size=3, head, seq_len=5, d_k
        # batch_size=3, head, seq_len=5, d_k： 有三句话，每句话分成head个头，每个头有5个词，每个词以d_K维的数据表示
        q = self.linear_q(query).contiguous().view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        k = self.linear_k(key).contiguous().view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        v = self.linear_v(value).contiguous().view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        # 自注意力机制的计算-->进行多头自注意力机制的计算
        att_res, self.att_weight = self_attention(q, k, v, mask, self.dropout)

        # 对注意力计算的结果，改变形状，batch_size=3, head, seq_len=5, d_k
        # batch_size=3, head, seq_len=5, d_k-->batch_size=3, seq_len=5, head, d_k-->batch_size=3, seq_len=5, head * d_k
        att_res = att_res.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 经过线性变换，获取输出结果
        out = self.out(att_res)
        return out


class FeedForward(torch.nn.Module):
    """
    前馈神经网络：功能给Transformer结构注入非线性的因素。让模型提取特征更强
    实现原理：
    1. 使用两层全连接网络（FFN）实现非线性映射
    2. 数学公式：FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    3. 通常中间层维度d_ff远大于d_model（如4×d_model）
    4. 与自注意力机制并行工作，形成Transformer的核心组件
    """

    def __init__(self, d_model, d_ff=2048, p=0.1):
        """
        :param d_model: 输入数据的维数（通常为512或768等）
        :param d_ff: 中间层的神经元个数（默认2048，通常为d_model的4倍）
        :param p: Dropout的概率（默认0.1）
        初始化说明：
        - 第一个线性层将输入映射到高维空间（d_model → d_ff）
        - 第二个线性层将高维特征映射回原始维度（d_ff → d_model）
        - Dropout层防止过拟合，随机失活中间层的输出
        - PyTorch默认使用Kaiming初始化权重（适用于ReLU激活）
        """
        super().__init__()
        self.fc = torch.nn.Linear(in_features=d_model, out_features=d_ff)
        self.out = torch.nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, data):
        """
        :param data: 输入张量，形状为[batch_size, seq_len, d_model]
        :return: 经过前馈网络处理后的张量，形状与输入相同

        计算过程详解：
        1. 第一层线性变换：data × W₁ + b₁ → [batch_size, seq_len, d_ff]
        2. ReLU激活函数：max(0, x) → 保留正向特征，增强非线性
        3. Dropout应用：随机失活d_ff维度的输出，防止过拟合
        4. 第二层线性变换：dropout_result × W₂ + b₂ → [batch_size, seq_len, d_model]
        """
        return self.out(self.dropout(torch.relu(self.fc(data))))


class ResidualConnection(torch.nn.Module):
    """
    残差连接：防止梯度消失，增强深层网络的训练稳定性
    实现原理：
    1. 对输入数据先进行LayerNorm标准化
    2. 通过子层（如Attention或FFN）处理
    3. 应用Dropout防止过拟合
    4. 最后将处理结果与原始输入相加（残差连接）
    数学公式：Output = x + Dropout(Sublayer(LayerNorm(x)))
    """

    def __init__(self, d_model, epsilon=1e-6, p=0.1):
        """
        :param d_model: 特征维度（需要标准化的维度）
        :param epsilon: LayerNorm中的数值稳定常数（防止除零）
        :param p: Dropout的概率
        初始化说明：
        - 创建LayerNorm层：对特征维度进行标准化
        - 创建Dropout层：随机失活子层输出，防止过拟合
        - 这两个模块在残差连接中是必须的组件
        """
        super().__init__()
        # 创建归一化层的对象（LayerNorm）
        self.norm = LayerNorm(d_model, epsilon)
        # 创建Dropout层
        self.dropout = torch.nn.Dropout(p)

    def forward(self, data, sub_layer):
        """
        :param data: 输入张量，形状为[batch_size, seq_len, d_model]
        :param sub_layer: 可调用的子层模块（如MultiHeadAttention或FeedForward）
        :return: 残差连接后的输出，形状与输入相同

        执行流程详解：
        1. LayerNorm标准化：对输入数据进行特征维度标准化
           - 保证子层输入分布稳定
           - 形状保持不变：[batch_size, seq_len, d_model]
        2. 子层处理：执行传入的子层操作（如自注意力或前馈网络）
        3. Dropout应用：随机失活子层输出（训练时）
        4. 残差相加：将处理结果与原始输入相加
           - 保证维度一致（子层输出与输入形状相同）
        """
        # 公式：Output = x + Dropout(Sublayer(LayerNorm(x)))
        return data + self.dropout(sub_layer(self.norm(data)))


class EncoderLayer(torch.nn.Module):
    """
    实现Encoder的一个层级
    功能：构成Transformer编码器的基础模块，包含自注意力机制和前馈网络
    结构组成：
    1. 多头自注意力层：捕获序列中词与词之间的依赖关系
    2. 前馈神经网络：引入非线性特征转换
    3. 两个残差连接模块：保证梯度传播并增强模型稳定性
    数学公式：
    EncoderLayer(x) = LayerNorm(x + Residual1(Attention(LayerNorm(x)))) + Residual2(FFN(...))
    """

    def __init__(self, d_model, head, d_ff, epsilon=1e-6, p=0.1):
        """
        :param d_model: 输入数据的维度（词向量维度）
        :param head: 多头注意力的头数
        :param d_ff: 前馈网络中间层的神经元数量
        :param epsilon: LayerNorm中的数值稳定常数
        :param p: Dropout的概率
        初始化说明：
        - 构建多头自注意力模块（MultiHeadAttention）
        - 构建前馈神经网络模块（FeedForward）
        - 构建两个残差连接模块（ResidualConnection）
        - 所有组件参数通过优化器自动更新
        """
        super().__init__()
        # 多头自注意力机制的计算
        self.attention = MultiHeadAttention(d_model, head, p)
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff, p)
        # 残差连接 * 2
        self.residual1 = ResidualConnection(d_model, epsilon, p)
        self.residual2 = ResidualConnection(d_model, epsilon, p)

    def forward(self, data, mask=False):
        """
        :param data: 输入张量，形状为[batch_size, seq_len, d_model]
        :param mask: 是否应用掩码（默认False，仅解码时需要）
        :return: 经过编码器层级处理后的张量，形状与输入相同

        数据流详解：
        1. 第一个残差连接：
           - 对输入进行LayerNorm标准化
           - 执行多头自注意力计算（Q=K=V=data）
           - 应用Dropout
           - 与原始输入相加形成残差
        2. 第二个残差连接：
           - 对上一步结果进行LayerNorm标准化
           - 执行前馈神经网络计算
           - 应用Dropout
           - 与输入相加形成第二个残差
        """
        # 多头自注意力机制的计算残差连接
        # 子层函数：lambda x: self.attention(x, x, x, mask)
        # 表示Q=K=V=x，这是自注意力的标准用法
        residual1_res = self.residual1(data, lambda x: self.attention(x, x, x, mask))

        # 前馈神经网络的残差连接
        # 直接传入feed_forward模块
        residual2_res = self.residual2(residual1_res, self.feed_forward)

        return residual2_res


class Encoder(torch.nn.Module):
    """
    Encoder: 编码器中EncoderLayer要堆叠6次
    功能：构建Transformer编码器主体，通过堆叠多层EncoderLayer提取序列特征
    结构组成：
    1. 多个EncoderLayer堆叠（默认6层）
    2. 最终LayerNorm标准化层
    数学公式：
    Encoder(x) = LayerNorm(EncoderLayer_6(...(EncoderLayer_1(x))))
    """

    def __init__(self, d_model, head, d_ff, layers=6, epsilon=1e-6, p=0.1):
        """
        :param d_model: 输入数据的维度（词向量维度）
        :param head: 多头注意力的头数
        :param d_ff: 前馈网络中间层的神经元数量
        :param layers: EncoderLayer的堆叠层数（默认6）
        :param epsilon: LayerNorm中的数值稳定常数
        :param p: Dropout的概率
        初始化说明：
        - 创建包含多个EncoderLayer的ModuleList
        - 每个EncoderLayer独立初始化（使用copy.deepcopy）
        - 添加最终的LayerNorm层（用于输出标准化）
        - 所有组件参数通过优化器自动更新
        """
        super().__init__()
        # 创建堆叠的EncoderLayer列表
        self.encoder_layers = torch.nn.ModuleList(
            [copy.deepcopy(EncoderLayer(d_model, head, d_ff, epsilon, p)) for _ in range(layers)]
        )
        # 最终的LayerNorm层（输出标准化）
        self.norm = LayerNorm(d_model, epsilon)

    def forward(self, data, mask=False):
        """
        :param data: 输入张量，形状为[batch_size, seq_len, d_model]
        :param mask: 是否应用掩码（默认False，仅解码时需要）
        :return: 经过编码器处理后的张量，形状与输入相同

        数据流详解：
        1. 依次通过每个EncoderLayer：
           - 每个EncoderLayer包含：
             * 多头自注意力机制（捕获序列依赖）
             * 前馈神经网络（非线性特征转换）
             * 两个残差连接（保证梯度传播）
        2. 最终应用LayerNorm标准化：
           - 对输出进行标准化处理
           - 保持维度不变：[batch_size, seq_len, d_model]
        """
        # 逐层处理输入数据
        for encoder_layer in self.encoder_layers:
            data = encoder_layer(data, mask)
        # 最终标准化输出
        return self.norm(data)


class DecoderLayer(torch.nn.Module):
    """
    实现Decoder的一个层级
    功能：构成Transformer解码器的基础模块，包含掩码自注意力、编码器-解码器注意力和前馈网络
    结构组成：
    1. 掩码多头自注意力层：处理目标序列（防止未来信息泄露）
    2. 多头注意力层：与编码器输出进行交互（捕获源-目标依赖）
    3. 前馈神经网络：引入非线性特征转换
    4. 三个残差连接模块：保证梯度传播并增强模型稳定性
    数学公式：
    DecoderLayer(x) =
        LayerNorm(x + Residual1(MaskedAttention(LayerNorm(x)))) →
        LayerNorm(...) + Residual2(Attention(..., encoder_out)) →
        LayerNorm(...) + Residual3(FFN(...))
    """

    def __init__(self, d_model, head, d_ff, epsilon=1e-6, p=0.1):
        """
        :param d_model: 输入数据的维度（词向量维度）
        :param head: 多头注意力的头数
        :param d_ff: 前馈网络中间层的神经元数量
        :param epsilon: LayerNorm中的数值稳定常数
        :param p: Dropout的概率
        初始化说明：
        - 构建掩码多头注意力模块（处理目标序列的自注意力）
        - 构建多头注意力模块（与编码器输出交互）
        - 构建前馈神经网络模块
        - 构建三个残差连接模块
        - 所有组件参数通过优化器自动更新
        """
        super().__init__()
        # 掩码多头注意力（处理目标序列的自注意力，防止未来信息泄露）
        self.mask_attention = MultiHeadAttention(d_model, head, p)
        # 多头注意力（与编码器输出交互）
        self.attention = MultiHeadAttention(d_model, head, p)
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff, p)
        # 创建三个残差连接模块
        self.residual1 = ResidualConnection(d_model, epsilon, p)  # 掩码注意力后
        self.residual2 = ResidualConnection(d_model, epsilon, p)  # 编码器-解码器注意力后
        self.residual3 = ResidualConnection(d_model, epsilon, p)  # 前馈网络后

    def forward(self, target_data, encoder_out, source_mask=False, target_mask=True):
        """
        :param target_data: 目标序列输入，形状为[batch_size, seq_len, d_model]
        :param encoder_out: 编码器输出，形状为[batch_size, seq_len, d_model]
        :param source_mask: 源序列掩码（通常用于解码时的特定任务）
        :param target_mask: 目标序列掩码（默认True，应用下三角掩码防止未来信息泄露）
        :return: 经过解码器层级处理后的张量，形状与输入相同

        数据流详解：
        1. 第一个残差连接（掩码自注意力）：
           - 对目标数据进行LayerNorm标准化
           - 执行掩码多头自注意力计算（Q=K=V=target_data）
           - 应用Dropout并形成残差
        2. 第二个残差连接（编码器-解码器注意力）：
           - 对上一步结果进行LayerNorm标准化
           - 执行多头注意力计算（Q=target_data, K/V=encoder_out）
           - 应用Dropout并形成残差
        3. 第三个残差连接（前馈网络）：
           - 对上一步结果进行LayerNorm标准化
           - 执行前馈神经网络计算
           - 应用Dropout并形成残差
        """
        # 1. 掩码自注意力（防止目标序列看到未来位置）
        residual1_res = self.residual1(
            target_data,
            lambda x: self.mask_attention(x, x, x, target_mask)  # Q=K=V=x
        )

        # 2. 编码器-解码器注意力（与编码器输出交互）
        residual2_res = self.residual2(
            residual1_res,
            lambda x: self.attention(x, encoder_out, encoder_out, source_mask)  # Q=x, K/V=encoder_out
        )

        # 3. 前馈神经网络
        residual3_res = self.residual3(residual2_res, self.feed_forward)

        return residual3_res


class Decoder(torch.nn.Module):
    """
    Decoder: 解码器中DecoderLayer要堆叠6次
    功能：构建Transformer解码器主体，通过堆叠多层DecoderLayer处理目标序列并生成输出
    结构组成：
    1. 多个DecoderLayer堆叠（默认6层）
    2. 最终LayerNorm标准化层
    数学公式：
    Decoder(x) = LayerNorm(DecoderLayer_6(...(DecoderLayer_1(x, encoder_out))))
    """

    def __init__(self, d_model, head, d_ff, layers=6, epsilon=1e-6, p=0.1):
        """
        :param d_model: 输入数据的维度（词向量维度）
        :param head: 多头注意力的头数
        :param d_ff: 前馈网络中间层的神经元数量
        :param layers: DecoderLayer的堆叠层数（默认6）
        :param epsilon: LayerNorm中的数值稳定常数
        :param p: Dropout的概率
        初始化说明：
        - 创建包含多个DecoderLayer的ModuleList
        - 每个DecoderLayer独立初始化（使用copy.deepcopy）
        - 添加最终的LayerNorm层（用于输出标准化）
        - 所有组件参数通过优化器自动更新
        """
        super().__init__()
        # 创建堆叠的DecoderLayer列表
        self.decoder_layers = torch.nn.ModuleList(
            [copy.deepcopy(DecoderLayer(d_model, head, d_ff, epsilon, p)) for _ in range(layers)]
        )
        # 最终的LayerNorm层（输出标准化）
        self.norm = LayerNorm(d_model, epsilon)

    def forward(self, target_data, encoder_out, source_mask=False, target_mask=True):
        """
        :param target_data: 目标序列输入，形状为[batch_size, seq_len, d_model]
        :param encoder_out: 编码器输出，形状为[batch_size, seq_len, d_model]
        :param source_mask: 源序列掩码（通常用于解码时的特定任务）
        :param target_mask: 目标序列掩码（默认True，应用下三角掩码防止未来信息泄露）
        :return: 经过解码器处理后的张量，形状与输入相同

        数据流详解：
        1. 依次通过每个DecoderLayer：
           - 每个DecoderLayer包含：
             * 掩码多头自注意力（防止目标序列看到未来位置）
             * 编码器-解码器注意力（与编码器输出交互）
             * 前馈神经网络（非线性特征转换）
             * 三个残差连接（保证梯度传播）
        2. 最终应用LayerNorm标准化：
           - 对输出进行标准化处理
           - 保持维度不变：[batch_size, seq_len, d_model]
        """
        # 逐层处理输入数据
        for decoder_layer in self.decoder_layers:
            target_data = decoder_layer(target_data, encoder_out, source_mask, target_mask)
        # 最终标准化输出
        return self.norm(target_data)


class Output(torch.nn.Module):
    """
    输出层：将Transformer解码器输出映射到词汇表空间
    功能：
    1. 将解码器输出的d_model维向量转换为词汇表维度
    2. 应用LogSoftmax生成概率分布（用于交叉熵损失计算）
    数学公式：
    Output(x) = LogSoftmax(Linear(x))
    其中：
    - Linear(x) = W·x + b
    - LogSoftmax(x_i) = log(exp(x_i)/Σ_j exp(x_j))
    """

    def __init__(self, d_model, vocab):
        """
        :param d_model: 解码器输出的维度（通常与词嵌入维度一致）
        :param vocab: 目标词汇表的大小
        初始化说明：
        - 创建线性层：将d_model维向量映射到词汇表维度
        - 使用LogSoftmax替代常规Softmax：
          * 数值稳定性更好
          * 直接输出log概率（适配NLLLoss）
        - PyTorch默认使用Kaiming初始化权重
        """
        super().__init__()
        # 创建线性层：d_model → vocab
        self.out = torch.nn.Linear(in_features=d_model, out_features=vocab)
        # 使用LogSoftmax替代常规Softmax：
        # 1. 数值稳定性更好（避免exp溢出）
        # 2. 直接输出log概率（适配NLLLoss）
        # 3. 在最后一个维度（词汇表维度）进行计算
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, data):
        """
        :param data: 解码器输出张量，形状为[batch_size, seq_len, d_model]
        :return: 词汇表概率分布，形状为[batch_size, seq_len, vocab]

        计算过程详解：
        1. 线性变换：
           - 将d_model维特征映射到vocab维
           - 公式：output = W·data + b
           - 形状转换：[batch_size, seq_len, d_model] → [batch_size, seq_len, vocab]
        2. LogSoftmax归一化：
           - 在词汇表维度（dim=-1）进行归一化
           - 公式：log(softmax(x)) = log(exp(x_i)/Σ_j exp(x_j))
           - 保证每个时间步的概率分布总和为1
        """
        return self.softmax(self.out(data))


class Transformer(torch.nn.Module):
    """
    Transformer 模型：完整的序列到序列架构
    功能：
    1. 编码器处理源序列（如英文句子）
    2. 解码器生成目标序列（如中文翻译）
    3. 输出层将解码器输出映射到目标词汇表
    结构组成：
    - 编码器（Encoder）：提取源序列特征
    - 解码器（Decoder）：生成目标序列并利用编码器信息
    - 输出层（Output）：生成词汇表概率分布
    数学公式：
    Transformer(x, y) = Output(Decoder(y, Encoder(x)))
    """

    def __init__(self, d_model, head, d_ff, vocab, layers=6, epsilon=1e-6, p=0.1):
        """
        :param d_model: 模型的隐藏维度（词嵌入维度）
        :param head: 多头注意力的头数
        :param d_ff: 前馈网络中间层维度
        :param vocab: 目标词汇表大小
        :param layers: 编码器/解码器层数（默认6）
        :param epsilon: LayerNorm中的数值稳定常数
        :param p: Dropout概率
        初始化说明：
        - 创建编码器模块：处理源序列
        - 创建解码器模块：处理目标序列并交互编码器输出
        - 创建输出层：将解码器输出转换为词汇表概率
        - 所有组件参数通过优化器自动更新
        """
        super().__init__()
        # 编码器模块：提取源序列特征
        self.encoder = Encoder(d_model, head, d_ff, layers, epsilon, p)
        # 解码器模块：生成目标序列并利用编码器信息
        self.decoder = Decoder(d_model, head, d_ff, layers, epsilon, p)
        # 输出层：将解码器输出映射到词汇表空间
        self.out = Output(d_model, vocab)

    def forward(self, source_data, target_data, source_mask=False, target_mask=True):
        """
        :param source_data: 源序列输入，形状为[batch_size, seq_len, d_model]
        :param target_data: 目标序列输入，形状为[batch_size, seq_len, d_model]
        :param source_mask: 源序列掩码（通常用于特殊任务）
        :param target_mask: 目标序列掩码（默认True，防止未来信息泄露）
        :return: 词汇表概率分布，形状为[batch_size, seq_len, vocab]

        数据流详解：
        1. 编码器处理：
           - 输入源序列 → 编码器 → 获取上下文表示(encoder_out)
           - encoder_out: [batch_size, seq_len, d_model]
        2. 解码器处理：
           - 输入目标序列 + encoder_out → 解码器 → 获取目标序列表示(decoder_out)
           - decoder_out: [batch_size, seq_len, d_model]
        3. 输出层处理：
           - decoder_out → 输出层 → 词汇表概率分布
           - out: [batch_size, seq_len, vocab]
        """
        # 1. 编码器处理源序列
        encoder_out = self.encoder(source_data, source_mask)

        # 2. 解码器处理目标序列并交互编码器输出
        decoder_out = self.decoder(target_data, encoder_out, source_mask, target_mask)

        # 3. 输出层生成词汇表概率分布
        out = self.out(decoder_out)

        return out


if __name__ == '__main__':
    # data = torch.zeros(1, 100, 20)
    # positional = PositionalEncoding(20, p=0)
    # res = positional(data) # 1, 200, 20
    # plt.plot(range(100), res[0, :, 4:8])
    # plt.show()
    """
    主函数模块：用于验证Transformer模型各组件功能及性能测试
    功能说明：
    1. 测试词嵌入 + 位置编码 + 编码器 + 解码器 + 输出层的完整流程
    2. 验证各模块输入输出的维度一致性
    3. 测量Transformer模型前向传播耗时
    4. 提供可视化调试示例（已注释）
    """

    # ============= [编码器测试部分] =============
    # 模拟源序列输入：batch_size=5, seq_len=100（词索引）
    data = torch.randint(1, 20000, (5, 100))

    # 初始化词嵌入层：将词索引映射到512维向量空间
    # 词汇表大小50000，d_model=512
    embedding = Embedding(50000, 512)

    # 初始化位置编码层：为序列添加位置信息
    positional_encoder = PositionalEncoding(512)

    # 初始化编码器：6层EncoderLayer，8头注意力，d_ff=2048
    encoder = Encoder(512, 8, 2048, 6)

    # 前向传播流程：
    # 1. 词嵌入：[5, 100] → [5, 100, 512]
    embedding_res = embedding(data)
    # 2. 添加位置编码：[5, 100, 512]
    positional_encoder_res = positional_encoder(embedding_res)
    # 3. 编码器处理：[5, 100, 512] → [5, 100, 512]
    encoder_res = encoder(positional_encoder_res)
    print("编码器输出形状:", encoder_res.shape)  # 验证维度一致性

    # ============= [解码器测试部分] =============
    # 模拟目标序列输入：batch_size=5, seq_len=120（词索引）
    target_data = torch.randint(1, 20000, (5, 120))

    # 初始化目标序列的词嵌入和位置编码
    de_embedding = Embedding(50000, 512)
    de_positional_encoder = PositionalEncoding(512)

    # 目标序列处理流程：
    # 1. 词嵌入 + 位置编码：[5, 120] → [5, 120, 512]
    target_data = de_positional_encoder(de_embedding(target_data))

    # 初始化解码器：6层DecoderLayer，8头注意力，d_ff=2048
    decoder = Decoder(512, 8, 2048, 6)

    # 解码器处理流程：
    # 1. 输入：[5, 120, 512] + 编码器输出[5, 100, 512]
    # 2. 输出：[5, 120, 512]
    decoder_res = decoder(target_data, encoder_res)
    print("解码器输出形状:", decoder_res.shape)  # 验证维度一致性

    # ============= [输出层测试部分] =============
    # 初始化输出层：将d_model=512映射到55000词汇表
    out = Output(512, 55000)

    # 输出层处理流程：
    # 1. 输入：[5, 120, 512]
    # 2. 输出：[5, 120, 55000]（log概率分布）
    out_res = out(decoder_res)
    print("输出层概率分布形状:", out_res.shape)  # 验证维度一致性

    # ============= [完整Transformer模型测试] =============
    # 记录开始时间
    start_time = time.time()

    # 初始化完整Transformer模型：
    # - d_model=768（大模型配置）
    # - head=12（对应论文中12头注意力）
    # - d_ff=2048（中间层维度）
    # - vocab=50000（目标词汇表大小）
    # - layers=12（12层堆叠，对应论文中base/large模型）
    transformer = Transformer(768, 12, 2048, 50000, 12)

    # 模拟输入数据：
    # - 源序列：batch_size=5, seq_len=100
    # - 目标序列：batch_size=5, seq_len=120
    data = torch.randint(1, 20000, (5, 100))
    target_data = torch.randint(1, 20000, (5, 120))

    # 源序列预处理：
    # 1. 词嵌入：[5, 100] → [5, 100, 768]
    # 2. 添加位置编码：[5, 100, 768]
    embedding = Embedding(50000, 768)
    positional_encoder = PositionalEncoding(768)
    source_data = positional_encoder(embedding(data))

    # 目标序列预处理：
    # 1. 词嵌入：[5, 120] → [5, 120, 768]
    # 2. 添加位置编码：[5, 120, 768]
    de_embedding = Embedding(50000, 768)
    de_positional_encoder = PositionalEncoding(768)
    target_data = de_positional_encoder(de_embedding(target_data))

    # 完整模型前向传播：
    # 1. 输入：source_data [5, 100, 768], target_data [5, 120, 768]
    # 2. 输出：[5, 120, 50000]（log概率分布）
    res = transformer(source_data, target_data)
    print("完整模型输出形状:", res.shape)

    # 输出模型处理耗时（秒）
    print("模型前向传播耗时:", time.time() - start_time)
