# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   2.py
# FileName:      5.py
# Author:       Souya Tom
# Datetime:     2026/9/21 17:37
# Software: 	PyCharm
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import re
import torch
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def clear_data(data):
    res = re.sub(r"([.!?])", r" \1", data)  # replace
    res = re.sub(r'[^a-zA-Z0.!?]+', ' ', res)
    return res


def get_data():
    with open('./data/eng-fra-v2.txt', 'r') as f:
        lines = f.readlines()

    lines = [[clear_data(i) for i in line.strip().split('\t')] for line in lines]

    english_word2idx = {"_GO": 0, "_EOS": 1}
    english_word_count = 2
    french_word2idx = {"_GO": 0, "_EOS": 1}
    french_word_count = 2
    for line in lines:
        english = line[0]
        french = line[1]
        for word in english.split(' '):
            if word not in english_word2idx.keys():
                english_word2idx[word] = english_word_count
                english_word_count += 1
        for word in french.split(' '):
            if word not in french_word2idx.keys():
                french_word2idx[word] = french_word_count
                french_word_count += 1

    # print(english_word_count)  # 2803
    # print(french_word_count)  # 4345
    english_idx2word = {idx: word for word, idx in english_word2idx.items()}
    french_idx2word = {idx: word for word, idx in french_word2idx.items()}
    return english_word2idx, english_idx2word, french_word2idx, french_idx2word, english_word_count, french_word_count, lines


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sentences, english_word2idx, french_word2idx):
        super().__init__()
        self.sentences = sentences
        self.english_word2idx = english_word2idx
        self.french_word2idx = french_word2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        index = min(max(0, index), len(self.sentences) - 1)
        english = self.sentences[index][0]
        french = self.sentences[index][1]
        english_word_idx = [self.english_word2idx[word] for word in english.split(' ')]
        french_word_idx = [self.french_word2idx[word] for word in french.split(' ')]

        english_word_idx.append(self.english_word2idx["_EOS"])
        french_word_idx.append(self.french_word2idx["_EOS"])
        return torch.tensor(english_word_idx), torch.tensor(french_word_idx)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, dim):
        '''
        :param vocab_size: 词表的大小
        :param dim: 词向量的维度
        '''
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)
        self.gru = torch.nn.GRU(input_size=dim, hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, data, hidden):
        embedding = self.embedding(data)
        gru_res, h = self.gru(embedding, hidden)
        return gru_res, h

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 128)


class Decoder(torch.nn.Module):
    def __init__(self, q_size, k_size, v_size, seq_len, vocab_size, dim, p=0.1):
        '''
        :param q_size:  Q的维度
        :param k_size:  K的维度
        :param v_size:  V的维度
        :param seq_len:  句子的长度
        :param vocab_size:  法文词的个数
        :param dim:  每个词的维度
        :param p:  随机失活的概率
        '''
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)
        self.dropout = torch.nn.Dropout(p=p)
        self.fc1 = torch.nn.Linear(in_features=q_size + k_size, out_features=seq_len)
        self.fc2 = torch.nn.Linear(in_features=v_size * 2, out_features=dim)
        self.gru = torch.nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.out = torch.nn.Linear(in_features=dim, out_features=vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, data, hidden, encoder_output):
        '''
        解码器的前向传播
        使用注意力机制的计算，计算之后，使用GRU进行每个时间步的计算，计算完之后使用线程层进行预测
        :param data:  解码器的输入 Decoder每个时间步的输入，经过Embedding之后，获取到Q
        :param hidden:  解码器的隐藏状态 Encoder最后一个时间步的隐藏状态，当做K, shape=1, 1, 128
        :param encoder_output:  编码器的输出 Encoder每个时间步的预测结果，组成上下文向量，当做V, shape=1, 20, 128
                                Encoder每个时间步的预测结果还需要进行后处理，将形状改变为能够计算的形状
        '''
        query = self.dropout(self.embedding(data))
        # 获取计算注意力权重
        attention_weight = torch.softmax(self.fc1(torch.cat((query, hidden), dim=-1)), dim=-1)
        # 计算上下文向量
        context_vector = torch.bmm(attention_weight, encoder_output)
        # 拼接
        cat_res = torch.cat((context_vector, query), dim=-1)
        # 线性变换
        res = self.fc2(cat_res)
        relu_res = torch.relu(res)
        # 使用GRU计算
        gru_res, h = self.gru(relu_res, hidden)
        output = self.out(gru_res[0])
        return self.softmax(output), h, attention_weight


def train():
    # 获取数据
    english_word2idx, english_idx2word, french_word2idx, french_idx2word, english_word_count, french_word_count, lines = get_data()
    # 创建数据集
    data_set = Dataset(lines, english_word2idx, french_word2idx)
    # 创建DataLoader
    data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
    # 创建模型
    encoder = Encoder(vocab_size=english_word_count, dim=128)
    decoder = Decoder(q_size=128, k_size=128, v_size=128, seq_len=20, vocab_size=french_word_count, dim=128)
    # 定义优化器和损失函数
    optim = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": 1e-4},
        {"params": decoder.parameters(), "lr": 1e-4}
    ])
    # 定义损失函数
    loss = torch.nn.NLLLoss()

    # 定义epoch进行模型的训练
    epochs = 10

    for epoch in range(epochs):
        # 统计损失的变量
        total_loss = 0
        iters = 0
        # 开启模型训练
        encoder.train()
        decoder.train()
        # 获取数据，进行模型训练
        for i1, (x, y) in enumerate(data_loader, 1):
            # 定义Encoder的数据
            encoder_hidden = encoder.init_hidden(batch_size=1)
            # 获取Encoder的输出
            encoder_out, encoder_h = encoder(x, encoder_hidden)

            # 定义Decoder的输入
            decoder_hidden = encoder_h
            # 处理上下文向量，Value, 必须满足句子的长度为20
            context_vector = torch.zeros(1, 20, 128)
            for i in range(encoder_out.shape[1]):
                context_vector[:, i] = encoder_out[:, i]

            # 处理Decoder每个时间步输入和计算
            decoder_input = torch.tensor([[french_word2idx["_GO"]]])
            # 定义变量，统计每句话的损失值
            sequence_loss = 0
            # 定义变量，统计每句话中的词的个数
            sequence_word_count = 0

            teacher_forcing_ratio = 0.5

            # 确定时间步
            for y_i in y[0]:
                # 获取Decoder每个时间步的输出
                decoder_out, decoder_h, att_weight = decoder(decoder_input, decoder_hidden, context_vector)
                # 根据Decoder的输出，获取预测结果【法文单词的下标】
                pred_index = torch.argmax(decoder_out, dim=-1)
                # 决定下一步输入
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                # 处理后续时间步的输入
                if use_teacher_forcing:
                    decoder_input = torch.tensor([[y_i]])  # 真实标签
                else:
                    decoder_input = pred_index.detach().view(1, 1)  # 模型预测
                decoder_hidden = decoder_h
                # 统计损失
                sequence_loss += loss(decoder_out, torch.tensor([y_i]))  # item()带有.detach()动作
                sequence_word_count += 1
                # 如果预测的结果为1，结束循环
                if y_i == 1:
                    break

            # teacher_forcing = True if random.randint(1, 10) > 5 else False
            # if teacher_forcing:
            #     # teacher forcing
            #     # 确定时间步
            #     for y_i in y[0]:
            #         # 获取Decoder每个时间步的输出
            #         decoder_out, decoder_h, att_weight = decoder(decoder_input, decoder_hidden, context_vector)
            #         # 根据Decoder的输出，获取预测结果【法文单词的下标】
            #         # pred_index = torch.argmax(decoder_out, dim=-1)
            #         # 处理后续时间步的输入
            #         decoder_input = torch.tensor([[y_i.item()]])
            #         decoder_hidden = decoder_h
            #         # 统计损失
            #         sequence_loss += loss(decoder_out, torch.tensor([y_i]))  # item()带有.detach()动作
            #         sequence_word_count += 1
            # else:
            #     # 自训练
            #     # 确定时间步
            #     for y_i in y[0]:
            #         # 获取Decoder每个时间步的输出
            #         decoder_out, decoder_h, att_weight = decoder(decoder_input, decoder_hidden, context_vector)
            #         # 根据Decoder的输出，获取预测结果【法文单词的下标】
            #         pred_index = torch.argmax(decoder_out, dim=-1)
            #         # 处理后续时间步的输入
            #         decoder_input = torch.tensor([[pred_index.item()]])
            #         decoder_hidden = decoder_h
            #         # 统计损失
            #         sequence_loss += loss(decoder_out, torch.tensor([y_i]))  # item()带有.detach()动作
            #         sequence_word_count += 1
            #         # 如果预测的结果为1，结束循环
            #         if pred_index.item() == 1:
            #             break

            # # 确定时间步
            # for y_i in y[0]:
            #     # 获取Decoder每个时间步的输出
            #     decoder_out, decoder_h, att_weight = decoder(decoder_input, decoder_hidden, context_vector)
            #     # 根据Decoder的输出，获取预测结果【法文单词的下标】
            #     pred_index = torch.argmax(decoder_out, dim=-1)
            #     # 处理后续时间步的输入
            #     decoder_input = torch.tensor([[pred_index.item()]])
            #     decoder_hidden = decoder_h
            #     # 统计损失
            #     sequence_loss += loss(decoder_out, torch.tensor([y_i]))  # item()带有.detach()动作
            #     sequence_word_count += 1
            #     # 如果预测的结果为1，结束循环
            #     if pred_index.item() == 1:
            #         break

            # 统计平均损失
            mean_loss = sequence_loss / sequence_word_count

            # 反向传播
            optim.zero_grad()
            mean_loss.backward()
            optim.step()

            total_loss += mean_loss
            iters += 1
            if i1 % 100 == 0:
                print(f"第{epoch + 1}轮训练，损失值为：{total_loss / iters}")
                total_loss = 0
                iters = 0
        # 模型的保存
        torch.save(encoder.state_dict(), f"./data/encoder_{epoch}")
        torch.save(decoder.state_dict(), f"./data/decoder_{epoch}")


def valid():
    # 确定验证数据
    valid_data = [
        ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
        ['i m more than a friend .', 'je suis plus qu une amie .'],
        ['she is beautiful like her mother .', 'elle est belle comme sa mere .']
    ]
    # 获取数据
    english_word2idx, english_idx2word, french_word2idx, french_idx2word, english_word_count, french_word_count, lines = get_data()
    # 创建Encoder和Decoder对象
    # 创建模型
    encoder = Encoder(english_word_count, 128)
    decoder = Decoder(128, 128, 128, 20, french_word_count, 128)
    # 加载模型的参数
    encoder.load_state_dict(torch.load("./data/encoder_9"))
    decoder.load_state_dict(torch.load("./data/decoder_9"))
    # 开启模型的预测
    encoder.eval()
    decoder.eval()
    # 获取验证集中的所有的句子
    for sentence in valid_data:
        # 获取英文和法文
        english = sentence[0]
        french = sentence[1]
        # 对英文处理
        english_data_idx = [english_word2idx[word] for word in english.split(" ")]
        english_data_idx.append(english_word2idx["_EOS"])

        # 处理Encoder的输入
        encoder_input = torch.tensor([english_data_idx])
        # 处理Encoder的隐藏状态输入
        encoder_hidden = encoder.init_hidden(1)
        # 获取Encoder模型的输出
        encoder_out, encoder_h = encoder(encoder_input, encoder_hidden)
        # 处理Decoder的输入
        # 隐藏状态
        decoder_hidden = encoder_h
        # 处理上下文向量
        context_vector = torch.zeros(1, 20, 128)
        for i in range(encoder_out.shape[1]):
            context_vector[:, i] = encoder_out[:, i]

        # 统计每个时间步的预测结果
        pred_list = []
        # Decoder第一个时间步的输入
        decoder_input = torch.tensor([[french_word2idx["_GO"]]])

        # 创建全0矩阵统计注意力权重值
        att_zero_matrix = torch.zeros(20, 20)

        for i in range(20):
            decoder_out, decoder_h, att_weight = decoder(decoder_input, decoder_hidden, context_vector)

            # plt.matshow(att_weight[0].detach().numpy())
            # plt.show()
            att_weight = att_weight[0]  # 1，20
            # 将注意力权重的值填充到前面的全0矩阵中
            att_zero_matrix[i] = att_weight[0]
            # 获取预测结果
            pred_index = torch.argmax(decoder_out, dim=-1)
            # 处理第二个时间步的输入
            decoder_input = torch.tensor([[pred_index.item()]])
            decoder_hidden = decoder_h
            # 存储预测结果
            pred_list.append(pred_index.item())
            # 遇见结束符就停止预测
            if pred_index.item() == 1:
                break

        att_zero_matrix = att_zero_matrix[:i, :]
        plt.matshow(att_zero_matrix.detach().numpy())
        plt.show()
        print(f"英文={english}")
        print(f"法文={french}")
        print(f"预测={' '.join([french_idx2word[idx] for idx in pred_list])}")
        print("-" * 100)


if __name__ == '__main__':
    # train()
    valid()
    # decoder = Decoder(128, 128, 128, 20, 1024, 128)
    # input = torch.tensor([[0]])  # batch_size1, seq_len1
    # h = torch.rand(1,1,128)
    # v = torch.rand(1,20,128)
    # res, _, _ = decoder(input, h, v)
    # print(res.shape)