# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   2.py
# FileName:      7.py
# Author:       Souya Tom
# Datetime:     2026/9/23 14:06
# Software: 	PyCharm
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# 将预训练模型的路径设置为全局变量
path = r"./data/bert-base-chinese"

# 创建AutoTokenizer对象
tokenizer = AutoTokenizer.from_pretrained(path)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out = torch.nn.Linear(in_features=768, out_features=2)
        self.pre_model = AutoModel.from_pretrained(path)
        # 冻结
        for param in self.pre_model.parameters():
            param.requires_grad_(False)

    def forward(self, data):
        with torch.no_grad():
            out = self.pre_model(**data)

        last_hidden_state = out.last_hidden_state
        attention_mask = data["attention_mask"]
        attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        pooler = torch.sum(last_hidden_state * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), 1e-9)
        return self.out(pooler)


# 构建Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path="./data/train.csv"):
        super().__init__()
        # 使用hugging face的load_dataset加载数据
        data_set = load_dataset(path="csv", data_files=file_path, split="train")
        # 过滤数据
        data_set = data_set.filter(lambda x: len(x["text"]) > 40)
        # 将data_set设置为属性
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        # 处理idx
        idx = min(max(0, idx), len(self.data_set) - 1)
        # 获取text
        sentence = self.data_set[idx]["text"]
        # 将text分成2句话
        sen1 = sentence[:20]
        sen2 = sentence[20:40]
        label = 1

        # 定义一个变量，相当于概率
        state = False if random.randint(1, 10) > 5 else True
        # 如果概率变量为True时，构造返例
        if state:
            # idx不能为当前传入的idx，在获取数据
            idx = random.randint(0, idx - 1) if idx >= 10 else random.randint(idx + 1, len(self.data_set) - 1)
            # 获取text
            sentence_other = self.data_set[idx]["text"]
            sen2 = sentence_other[20:40]
            label = 0

        return [sen1, sen2], label


def collate_func(data):
    # 获取句子和目标
    sentences = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 获取tokenizer的结果
    token_res = tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=42)
    # print("".join(tokenizer.convert_ids_to_tokens(token_res["input_ids"][0])))
    return token_res, torch.tensor(labels).long()


def train():
    # 创建Dataset
    data_set = Dataset()
    # 创建DataLoader
    data_loader = DataLoader(data_set, batch_size=8, collate_fn=collate_func, drop_last=True)
    # 创建模型
    model = Model()
    # 定义损失函数和优化器
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(params=model.parameters(), lr=7e-4)

    # 训练模型
    for epoch_idx in range(10):
        total_loss = 0
        iters = 0

        model.train()
        for i, (x, y) in enumerate(data_loader, 1):
            out = model(x)
            loss_val = loss(out, y)
            optim.zero_grad()
            loss_val.backward()
            optim.step()

            total_loss += loss_val
            iters += 1
            if i % 10 == 0:
                print(f"第{epoch_idx + 1}轮训练，损失值为：{total_loss / iters}")
                total_loss = 0
                iters = 0

        torch.save(model.state_dict(), f"./data/model_{epoch_idx}.pth")


def valid():
    # 创建Dataset
    data_set = Dataset("./data/validation.csv")
    # 创建DataLoader
    data_loader = DataLoader(data_set, batch_size=8, collate_fn=collate_func, drop_last=True)
    # 创建模型
    model = Model()
    # 加载模型参数
    model.load_state_dict(torch.load("./data/model_9.pth"))
    # 定义变量统计结果
    accuracy = 0
    length = 0
    # 开启模型的预测
    model.eval()
    for i, (x, y) in enumerate(data_loader, 1):
        out = model(x)

        accuracy += (out.argmax(dim=-1) == y).sum()
        length += len(y)
        if i % 10 == 0:
            print(f"准确度为：{accuracy / length}")
            accuracy = 0
            length = 0


def pred(sen1="我看过朋友的还可以，但是我订的书迟迟未到", sen2="法调节，前台服务僵化。用早餐时，服务员居然叫我回房间把拖鞋换成"):
    # 数据预处理
    sentence = [sen1[:20], sen2[:20]]
    # 创建DataLoader
    data = tokenizer([sentence], return_tensors="pt", padding="max_length", truncation=True, max_length=42)
    # 创建模型
    model = Model()
    # 加载模型参数
    model.load_state_dict(torch.load("./data/model_9.pth"))
    # 开启模型的预测
    model.eval()
    # 获取模型的输出
    out = model(data)
    # 获取模型的预测
    pred_res = out.argmax(dim=-1)
    print(pred_res)


if __name__ == '__main__':
    # train()
    # valid()
    pred()
