# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   pythonlearn
# FileName:      3.py
# Author:       Souya Tom
# Datetime:     2026/9/19 18:21
# Software: 	PyCharm
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import torch

torch.manual_seed(992)

# 输入数据形状：batch_size, seq_len, input_size
# 输入隐藏状态形状：num_layers, batch_size, hidden_size  全0初始化
# 计算：tanh(wx+b+wh+b)
rnn = torch.nn.RNN(input_size=5, hidden_size=4, num_layers=1, batch_first=True, bidirectional=False)
# 输出数据的形状：batch_size，seq_len，hidden_size
# 隐藏状态输出数据的形状：num_layers，batch_size，hidden_size

data = torch.rand(3, 10, 5)
hidden = torch.zeros(1, 3, 4)

res, h = rnn(data, hidden)
print(res[:, -1, :])  # 3,10,10
print(h)

"""
    LSTM(input_size, hidden_size, num_layers, batch_first, bidirectional)
        input_size: 输入数据最后一维的维数
        hidden_size：长周期和短周期的最后一维的维数
        num_layers：长周期和短周期的层数

        输入数据的形状：batch_size, seq_len, input_size
            输出：batch_size, seq_len, hidden_size
                bidirectional如果为True: batch_size, seq_len, hidden_size*2
        长周期和短周期形状：num_layers, batch_size, hidden_size(神经元的个数)默认全0初始化
            bidirectional如果为True: num_layers*2, batch_size, hidden_size
            输出：num_layers, batch_size, hidden_size
                bidirectional如果为True: num_layers*2, batch_size, hidden_size

        初始化的h, C
        遗忘门 f = sigmoid(wx+b+wh+b)
        输入门 i = sigmoid(wx+b+wh+b)
        候选的长周期的状态：hat_c = tanh(wx+b+wh+b)
            更新最新的细胞状态：C_new = f * C + i * hat_c
        输出门：o = sigmoid(wx+b+wh+b)
            out = o * tanh(C_new)

    bidirectional: 
"""

# lstm = torch.nn.LSTM(input_size=128, hidden_size=2, num_layers=1, batch_first=True)
# """for name, param in lstm.named_parameters():
#     print(name)
#     print(param.shape)
#     print("-"*100)"""
#
# data = torch.rand(10, 20, 128)
# c = h = torch.zeros(1, 10, 2)
#
# res, (c1, h1) = lstm(data, (c, h))
# print(res.shape)  # 10， 20， 2
# print(c1.shape) # 1, 10, 2
# print(h1.shape) # 1, 10, 2


lstm = torch.nn.LSTM(input_size=128, hidden_size=2, num_layers=1, batch_first=True, bidirectional=True)
"""for name, param in lstm.named_parameters():
    print(name)
    print(param.shape)
    print("-"*100)"""

data = torch.rand(10, 20, 128)
c = h = torch.zeros(1 * 2, 10, 2)

res, (c1, h1) = lstm(data, (c, h))
print(res.shape)  # 10， 20， 2
print(c1.shape)  # 1, 10, 2
print(h1.shape)  # 1, 10, 2

"""
    GRU(input_size=128, hidden_size=2, num_layers=1, batch_first=True)
        输入数据的形状：batch_size, seq_len, input_size
            输出：batch_size, seq_len, hidden_size
                bidirectional如果为True: batch_size, seq_len, hidden_size*2
        隐藏状态形状：num_layers, batch_size, hidden_size(神经元的个数)默认全0初始化
            bidirectional如果为True: num_layers*2, batch_size, hidden_size
            输出：num_layers, batch_size, hidden_size
                bidirectional如果为True: num_layers*2, batch_size, hidden_size


        重置门：r = sigmoid(wx+b+wh+b)
            候选的隐藏状态 hat_h = tanh(wx+b+w(r*h)+b)
        更新门：z = sigmoid(wx+b+wh+b)
            输出：out = (1-z)*h + z * hat_h

"""

gru = torch.nn.GRU(input_size=1, hidden_size=3, num_layers=1, batch_first=True, bidirectional=True)
"""for name, param in gru.named_parameters():
    print(name)
    print(param.shape)
    print("-"*100)"""

data = torch.rand(10, 20, 1)
h = torch.zeros(2, 10, 3)

res, h1 = gru(data, h)
print(res.shape)  # 10,20,3
print(h.shape)  # 1, 10, 3

# -*-coding:utf-8 -*- #
import string

from torch.utils.data import DataLoader


def get_data():
    # 读取文件中的数据
    with open("./data/name_classfication.txt", encoding="utf-8") as f:
        lines = f.readlines()

    # 定义两个列表，存储姓和国家
    names, countries = [], []
    # 定义一个列表，存储唯一的国家
    unique_country = []
    # 获取每行数据
    for line in lines:
        # 对每行数据进行分割，获取姓和国家
        name, country = line.strip().split("\t")
        # 将姓名和国家存储在列表中
        names.append(name)
        countries.append(country)
        # 判断国家是否不在unique_country中，才存储
        if country not in unique_country:
            unique_country.append(country)
    return names, countries, unique_country


# 构建数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, names, countries, unique_country):
        super().__init__()
        # 创建属性
        self.names = names
        self.countries = countries
        self.unique_country = unique_country

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # 处理下标
        idx = min(max(0, idx), len(self.names) - 1)
        # 获取x和y的数据
        x_data = self.names[idx]  # zhang
        y_data = self.countries[idx]
        # 根据y_data获取国家的下标，并转成tensor
        y = torch.tensor(self.unique_country.index(y_data), dtype=torch.int64)
        # 处理x  abcd.....---->string.ascii_letters
        chars = string.ascii_letters + " ',"
        # 创建一个全0的tensor矩阵，行为姓的长度，列为字母的个数
        zero_data = torch.zeros(len(x_data), len(chars), dtype=torch.float32)
        # 循环去除姓中的字母，再获取字母在chars中对应的下标索引，然后更改全0矩阵中字母对应索引位置的值
        # Abl
        for i in range(len(x_data)):
            # print(i, x_data[i], end=" ")
            # print(chars.index(x_data[i]))
            zero_data[i, chars.index(x_data[i])] = 1
        return zero_data, y


if __name__ == '__main__':
    names, countries, unique_country = get_data()

    dataset = Dataset(names, countries, unique_country)  # __init__
    # print(dataset[0])  # __getitem___
    # print(len(dataset))  # __len__
    dataloader = DataLoader(dataset, batch_size=1)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break

from tqdm import tqdm


class RNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # # zhang--1,5,55-->1,5,128--->1,128
        self.rnn = torch.nn.RNN(input_size=55, hidden_size=128, num_layers=1, batch_first=True)
        # FC层使用RNN输出数据中，最后一个时间步的数据进行预测
        self.fc = torch.nn.Linear(in_features=128, out_features=18)

    def forward(self, data, hidden):
        # rnn_res==>batch_size, seq_len, hidden_size
        rnn_res, h = self.rnn(data, hidden)
        # 取出最后一个时间的输出
        rnn_res = rnn_res[:, -1, :]
        # 使用全连接层预测
        return self.fc(rnn_res), h

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 128)


class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # # zhang--1,5,55-->1,5,128--->1,128
        self.lstm = torch.nn.LSTM(input_size=55, hidden_size=128, num_layers=1, batch_first=True)
        # FC层使用RNN输出数据中，最后一个时间步的数据进行预测
        self.fc = torch.nn.Linear(in_features=128, out_features=18)
        self.softmax = torch.nn.LogSoftmax(dim=-1)  # NLLLoss

    def forward(self, data, hidden):
        # rnn_res==>batch_size, seq_len, hidden_size
        lstm_res, h = self.lstm(data, hidden)
        # 去除最后一个时间的输出
        lstm_res = lstm_res[:, -1, :]
        # 使用全连接层预测
        return self.softmax(self.fc(lstm_res)), h

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 128), torch.zeros(1, batch_size, 128)


def train():
    # 获取数据
    names, countries, unique_country = get_data()
    # 创建DataSet
    dataset = Dataset(names, countries, unique_country)
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # 创建模型
    rnn_model = RNNModel()
    lstm_model = LSTMModel()
    # 创建优化器  # Adam,AdamW
    # optim = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    optim = torch.optim.Adam(
        [{"params": rnn_model.parameters(), "lr": 1e-4},
         {"params": lstm_model.parameters(), "lr": 1e-4}])
    # 创建损失函数
    rnn_loss = torch.nn.CrossEntropyLoss()
    lstm_loss = torch.nn.NLLLoss()
    # 定义epoch进行模型的训练
    epochs = 10
    for i in range(epochs):
        # 定义变量统计损失
        rnn_total_loss = 0
        lstm_total_loss = 0
        iters = 0
        tq = tqdm(range(len(dataloader)), desc="训练模型")
        # 获取数据进行模型的训练
        for x, y in dataloader:
            # 获取隐藏桩体
            rnn_hidden = rnn_model.init_hidden(len(y))
            lstm_hidden = lstm_model.init_hidden(len(y))
            # 获取模型的输出
            rnn_out, rnn_h = rnn_model(x, rnn_hidden)
            lstm_out, lstm_h = lstm_model(x, lstm_hidden)
            # 计算损失
            rnn_loss_val = rnn_loss(rnn_out, y)
            lstm_loss_val = lstm_loss(lstm_out, y)
            # 反向传播
            optim.zero_grad()
            rnn_loss_val.backward()
            lstm_loss_val.backward()
            optim.step()

            rnn_total_loss += rnn_loss_val.item()
            lstm_total_loss += lstm_loss_val.item()
            iters += 1

            tq.update()
        tq.close()
        print(f"第{i + 1}轮训练，RNN-损失值为：{rnn_total_loss / iters}, LSTM-损失值为:{lstm_total_loss / iters}")

    # 保存模型
    torch.save(rnn_model.state_dict(), "./data/rnn.pth")
    torch.save(lstm_model.state_dict(), "./data/lstm.pth")


def pred(name="Zhang"):
    # 获取国家数据
    _, __, unique_country = get_data()
    # 将姓转成one-hot
    chars = string.ascii_letters + " ',"
    # 创建一个全0的矩阵
    zero_matrix = torch.zeros(len(name), len(chars))
    # 将姓中的字母对应位置改1
    for i in range(len(name)):
        zero_matrix[i, chars.index(name[i])] = 1

    # 创建网络结构
    model = RNNModel()
    # 初始化隐藏状态
    h = model.init_hidden(1)
    # 加载网络参数
    model.load_state_dict(torch.load("./data/rnn.pth"))
    # 获取网络结构的输出  # 1, 5, 55
    out, _ = model(zero_matrix.unsqueeze(0), h)  # batch, 18
    # 获取预测结构最大值的下标
    # index = torch.argmax(out, dim=-1)
    # print(index, unique_country[index.item()])
    v, i = torch.topk(out, 3, dim=-1)
    # 获取对应的国家
    for v1, i1 in zip(v[0], i[0]):
        print(v1, i1, unique_country[i1.item()])


if __name__ == '__main__':
    # train()
    pred("john")
    # pass


# data = torch.rand(10)
# print(data)
# print(torch.argmax(data))
#
# v, i = torch.topk(data, 3)
# print(v)
# print(i)

# data = torch.rand(2, 10)
# print(data)
# print(torch.argmax(data, dim=-1))
#
# v, i = torch.topk(data, 3, dim=-1)
# print(v)
# print(i)


data = torch.rand(2, 10)
print(data)
print(torch.argmin(data, dim=-1))

v, i = torch.topk(data, 3, dim=-1, largest=False)
print(v)
print(i)
