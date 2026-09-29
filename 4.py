# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   pythonlearn
# FileName:      4.py
# Author:       Souya Tom
# Datetime:     2026/9/20 16:59
# Software: 	PyCharm
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import torch

"""
    软注意力机制：根据输入数据，获取源数据中词和词之间的关系特征
    seq2seq的注意力机制的计算流程：
        K：Encoder(编码器)中GRU最后一个时间步的隐藏状态的输出。表示每个时间步的索引
        V: Encoder(编码器)中GRU每个时间步的预测结果，组合而成的一个向量。表示具体的数据
        Q: Decoder(解码器)中每个时间步的输入
        1：QK计算相似度，得到相似度的分数，对相似度的分数计算注意力权重。
            attention_weight = softmax(Linear(cat([Q, K])))
        2: 注意力权重和V相乘得到上下文向量
            context_vector = attention_weight @ V
        3: 上下文向量和Q再进行拼接，拼接之后再进行线性变换。结果当前Decoder中GRU的输入
            data = Linear(cat([context_vector, Q]))
"""


class Seq2SeqSoftAttention(torch.nn.Module):
    def __init__(self, q_size, k_size, v_size, seq_len, out_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=q_size + k_size, out_features=seq_len)
        self.fc2 = torch.nn.Linear(in_features=v_size * 2, out_features=out_size)

    def forward(self, q, k, v):
        # q,k拼接
        cat_res = torch.cat((q, k), dim=-1)
        # 再线性变换
        fc1_res = self.fc1(cat_res)
        # 调用softmax获取注意力权重
        att_weight = torch.softmax(fc1_res, dim=-1)
        # 注意力权重@v，得到上下文向量
        matmul_res = torch.matmul(att_weight, v)
        # 上下文向量和q拼接
        cat_res1 = torch.cat((matmul_res, q), dim=-1)
        # 再线性变换
        res = self.fc2(cat_res1)
        return res


"""
        自注意力机制：获取一句话中，词和词之间的关系特征
         1: 获取QK的相似度 QK.T/sqrt(d_k) # 将score值缩放在合理区间，让梯度健康的传播
         2: 获取注意力权重：softmax(QK.T/sqrt(d_k))
         3: 注意力权重和v相乘：softmax(QK.T/sqrt(d_k)) @ v
"""


def self_attention(q, k, v):
    # 获取d_k
    d_k = q.size(-1)
    # 处理k的转置
    k_T = k.transpose(-2, -1)
    # 计算qk之间的相似度分数
    score = torch.bmm(q, k_T) / torch.sqrt(torch.tensor(d_k))
    # 获取注意力权重
    att_weight = torch.softmax(score, dim=-1)
    # 计算注意力的结果
    res = torch.bmm(att_weight, v)
    return res


if __name__ == '__main__':
    # GRU的隐藏状态：num_layers, batch_size, hidden_size
    # GRU的输出：batch_size=1, seq_len, hidden_size
    k = torch.rand(1, 1, 128)
    v = torch.rand(1, 10, 128)  # 1, 1, 10-->1,1,128
    # GRU的每个时间步输入：batch_size, 1, dim
    q = torch.rand(1, 1, 128)

    # print(torch.cat((q, k), dim=-1))  # 1,1,256

    attention = Seq2SeqSoftAttention(128, 128, 128, 10, 128)
    res = attention(q, k, v)
    print(res.shape)

    q = torch.rand(5, 100, 512)
    k = torch.rand(5, 100, 512)
    v = torch.rand(5, 100, 512)
    res = self_attention(q, k, v)
    print(res.shape)
