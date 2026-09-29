# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   pythonlearn
# FileName:      2.py
# Author:       Souya Tom
# Datetime:     2026/9/17 14:19
# Software: 	PyCharm
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import jieba
import jieba.posseg as pseg
import pandas as pd
import fasttext
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import joblib

'''
jieba分词。
    安装：pip install jieba

    精确分词：
        jieba.lcut(text, cut_all=False): 将分词的结果以列表的方式返回
        jieba.cut(text, cut_all=False): 将分词的结果以生成器的方式返回
    全模式分词：可能会分割语义
        jieba.lcut(text, cut_all=True): 将分词的结果以列表的方式返回
        jieba.cut(text, cut_all=True): 将分词的结果以生成器的方式返回
    搜索分词：
        jieba.lcut_for_search(text)
        jieba.cut_for_search(text)
    按照词性分词：
        jieba.posseg.lcut(text)
        jieba.posseg.cut(text)
    使用指定的词表分词：
        jieba.load_userdict(path)
'''

jieba.setLogLevel(logging.NOTSET)
text = "即使人生是一场悲剧，也应该笑着把人生演完。"
np.random.seed(992)

cut = jieba.lcut(text, cut_all=False)


# print(cut)


# cut = jieba.cut(text, cut_all=False)
# print(cut)  # <generator object Tokenizer.cut at 0x000002786627D540>


# 精确分词方法
def precise_cut(text):
    """
    使用精确模式进行中文分词。

    参数:
        text (str): 需要分词的文本

    返回:
        list: 分词后的结果列表
    """
    return jieba.lcut(text, cut_all=False)


# 全模式分词方法
def full_cut(text):
    """
    使用全模式进行中文分词（可能会分割语义）。

    参数:
        text (str): 需要分词的文本

    返回:
        list: 分词后的结果列表
    """
    return jieba.lcut(text, cut_all=True)


# 搜索引擎优化分词方法
def search_cut(text):
    """
    使用搜索引擎优化模式进行中文分词。

    参数:
        text (str): 需要分词的文本

    返回:
        list: 分词后的结果列表
    """
    return jieba.lcut_for_search(text)


# 词性标注分词方法
def pos_cut(text):
    """
    对文本进行词性标注和分词。

    参数:
        text (str): 需要分词的文本

    返回:
        generator: 包含词语及其词性的生成器
    """
    return pseg.cut(text)


# 使用指定词表分词方法
def custom_dict_cut(text, dict_path):
    """
    加载自定义词典并使用指定词典进行中文分词。

    参数:
        text (str): 需要分词的文本
        dict_path (str): 自定义词典文件的路径

    返回:
        list: 分词后的结果列表
    """
    jieba.load_userdict(dict_path)
    return jieba.lcut(text, cut_all=False)


# # 示例调用
# print("精确分词示例:", precise_cut(text))
# print("全模式分词示例:", full_cut(text))
# print("搜索分词示例:", search_cut(text))
#
# print("\n词性标注分词示例:")
# for word, flag in pos_cut(text):
#     print(f"{word} -> {flag}")
#
# # 如果需要测试自定义词典分词，请提供自定义词典的路径
# custom_dict_path = "./data/word.txt"
# print("自定义词典分词示例:", custom_dict_cut(text, custom_dict_path))

'''
文本的tensor表示：
        one-hot(独热编码)：词对应位置以1表示，其余为以0表示
          one-hot：严重分割语义，使用不多
            安装tensorflow: pip install tensorflow==2.15.0[了解]
                创建Tokenizer对象
                tokenizer = Tokenizer()
                训练：tokenizer.fit_on_texts(词汇列表)
                获取词对应的下标：tokenizer.word_index
                获取下标对应的词: tokenizer.index_word
                词汇列表中所有词对应的下标：tokenizer.texts_to_sequences(词汇列表)
            
        word2vec: 词以向量的表示
            word2vec：先训练模型，使用词获取模型的参数当做词对应的向量。
            训练模型的模式：
                CBOW: 根据上下文预测中间词
                skipgram: 根据中间词预测上下文
                
            fasttext: 下载 pip install fasttext-wheel
            使用时：
                1：训练模型
                    fasttext.train_unsupervised(path, # 词汇存放的文件路径，每个词之间以空格隔开
                                                model="训练模型的模式",  # 训练模型的模式:CBOW,skipgram
                                                dim,  # 每个词的向量维度
                                                minCount=1,  # 过滤低频词，一个词最少要出现多少词
                                                epoch,  # 训练的轮次
                                                ws,     # 窗口大小
                                                thread, # 线程
                                                lr....) # 学习率
                    
                    获取：
                        get_word_vector(word): 获取词对应的向量
                        get_words(): 获取模型所有的词
                        get_nearest_neighbors(word, n): 获取一个词和它相邻的n个词
                        
                        保存：save_model()
                        加载：load_model()
                    
        embedding：词以向量的表示-词嵌入层
            torch.nn.Embedding(num_embeddings=唯一词的个数, embedding_dim=每个词以多少维的向量表示)
            
            # 词汇向量的展示
            summary = SummaryWriter()
            # 给summary添加词嵌入数据
            summary.add_embedding(embedding.weight.data, 词汇列表)
            # 关闭summary
            summary.close()
            
            启动web服务器：tensorborad --logdor=runs --host=0.0.0.0

'''

unique = []
for word in cut:
    if word not in unique:
        unique.append(word)

# print(unique)
one_hot = []
for i in range(len(unique)):
    # 创建一个全0的列表
    zero_list = [0] * len(unique)
    # 更改
    zero_list[i] = 1
    # zero_list追加到前面的列表
    one_hot.append(zero_list)
    # print(unique[i], zero_list)
# print(one_hot)

# 使用tensorflow实现独热编码
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_one_hot_with_tensorflow(text):
    """
    使用TensorFlow的Tokenizer实现独热编码。

    参数:
        text (str): 需要处理的文本字符串

    返回:
        list: 独热编码表示的词向量列表
    """
    # 分词
    cut = jieba.lcut(text, cut_all=False)

    # 去重获取唯一词汇列表
    unique = []
    for word in cut:
        if word not in unique:
            unique.append(word)

    # 创建Tokenizer对象
    tokenizer = Tokenizer()
    # 训练
    tokenizer.fit_on_texts(unique)
    # 保存tokenizer
    # joblib.dump(tokenizer, "./data/token.pth")

    # 加载模型
    tokenizer = joblib.load("./data/token.pth")
    # 获取模型中的word_index, index_word
    print(tokenizer.word_index)
    print(tokenizer.index_word)

    # 实现one-hot
    one_hot = []
    for i in range(len(unique)):
        zero_list = [0] * len(unique)
        index = tokenizer.word_index[unique[i]] - 1
        zero_list[index] = 1
        one_hot.append(zero_list)
    sequences = tokenizer.texts_to_sequences(unique)
    print("sequences:", sequences)
    # print("padded", pad_sequences(sequences))
    print("matrix", tokenizer.texts_to_matrix(unique))  # 二维数组
    # 截断与填充：texts_to_matrix 生成的是等长向量（列数为词表大小），它不会处理句子长度（即不会把文本变成序列）。
    # 如果你需要序列（(batch, maxlen)），应该使用 texts_to_sequences 配合 pad_sequences
    return one_hot


new_cut = pd.get_dummies(cut, dtype=int, drop_first=False)


# print(new_cut)


# word2vec
def train_word2vec_model(text, model_type, vector_size=100, min_count=1, window=5, epochs=10, thread=4):
    """
    使用fasttext训练word2vec模型。

    参数:
        text (str): 需要处理的文本字符串，每个词之间以空格隔开。
        model_type (str): 训练模型的模式，可以是 "cbow" 或 "skipgram"。
        vector_size (int): 每个词的向量维度。
        min_count (int): 过滤低频词，一个词最少要出现多少次才会被保留。
        window (int): 窗口大小，即考虑前后多少个词作为上下文。
        epochs (int): 训练的轮数。
        thread (int): 训练的线程数。

    返回:
        fasttext.FastText: 训练好的fasttext模型。
    """
    # 写入训练数据
    # 使用精确分词将文本分割为词语列表，并用空格连接后写入训练文件
    with open("./data/data.txt", mode="w", encoding="utf-8") as f:
        f.write(" ".join(precise_cut(text)))

    # 训练模型
    model = fasttext.train_unsupervised(
        "./data/data.txt",  # 训练数据文件路径
        model=model_type,  # 指定训练模式（CBOW或Skipgram）
        dim=vector_size,  # 词向量维度
        min_count=min_count,  # 忽略出现次数低于该值的词
        ws=window,  # 上下文窗口大小
        epoch=epochs,  # 训练迭代次数
        thread=thread  # 并行线程数量
    )

    # 示例操作（可选）
    # 获取特定词语的向量表示（示例为"笑"）
    print("词向量示例:", model.get_word_vector("笑"))
    # 查找与指定词语最相似的3个词（包含相似度分数）
    print("相似词示例:", model.get_nearest_neighbors("笑", 3))
    # 获取句子的向量表示（示例句子已分词）
    print("句子向量示例:", model.get_sentence_vector("即使 人生 是"))

    # 保存训练完成的模型到指定路径
    model.save_model(f"./data/model_{model_type}.pth")

    return model


# embedding
import torch
from torch.utils.tensorboard import SummaryWriter


def create_embedding(unique_words, embedding_dim):
    """
    创建词嵌入层并可视化词向量。

    参数:
        unique_words (list): 唯一的词汇列表。
        embedding_dim (int): 每个词的向量维度。

    返回:
        torch.nn.Embedding: 词嵌入层对象。
    """
    # 创建词嵌入层
    embedding = torch.nn.Embedding(num_embeddings=len(unique_words), embedding_dim=embedding_dim)

    # 打印词嵌入层参数
    print("Embedding Layer Weights:")
    print(embedding.weight.data)

    # 示例输入：假设两个句子，每个句子包含两个词（使用它们的索引）
    input_indices = torch.tensor([[0, 1], [1, 2]])  # batch_size=2, sequence_length=2
    embedded_sentences = embedding(input_indices)
    print("\nEmbedded Sentences:")
    print(embedded_sentences)  # 输出形状为 (batch_size, sequence_length, embedding_dim)

    # 可视化词向量
    summary_writer = SummaryWriter()
    summary_writer.add_embedding(
        embedding.weight.data,
        metadata=unique_words,
        label_img=None,
        global_step=None
    )
    summary_writer.close()

    return embedding


# 示例调用
# 确保 'cut' 已经定义，即已经对文本进行了分词处理
# unique_words 是从 cut 中提取的唯一词列表
# 调用示例：create_embedding(unique, embedding_dim=4)

'''
    seaborn as sns
        seaborn下载：pip install seaborn
        底层matplotlib
    sns.scatterplot(data, x, y): 散点图
    sns.lineplot(data, x, y): 折线图
    sns.barplot(data, x, y): 柱状图
    sns.stripplot(data, x, y): 在x列中y的统计
    sns.countplot(data, x): 统计x列的每个类别的个数
    sns.displot(data, x, kind="hist", bins=柱子个数)：直方图
    sns.heatmap(data, annot=True, cmap="coolwarm", fmt=".2f"): 热力图
    WordCloud: pip install wordcloud
        1: 创建WordCloud对象
                wordcloud = WordCloud(字体, 词云中词的个数, 设置背景色)
                img = wordcloud.generate(字符串)
'''

arr = np.random.randint(1, 10, (100, 3))
data = pd.DataFrame(data=arr, columns=["A", "B", "C"])
data["D"] = np.random.randint(1, 3, (100,))
# print(data)
# 散点图
# sns.scatterplot(data, x="A", y="B")
# 折线图
# sns.lineplot(data, x="A", y="B")
# 柱状图
# sns.barplot(data, x="A", y="B")
# 在x列中y的统计
# sns.stripplot(data, x="D", y="B")
# 统计某列类别的个数
# sns.countplot(data, x="D")
# 直方图
# sns.displot(data, x="C", kind="hist", bins=100)

# 热力图
# res_data = np.random.rand(4, 3)
# sns.heatmap(res_data, annot=True, cmap="coolwarm", fmt=".3f")
# plt.show()

plt.style.use('fivethirtyeight')

train_data = pd.read_csv('data/train.tsv', sep='\t')
test_data = pd.read_table('data/dev.tsv', sep='\t')
# print(train_data)
# print(test_data)
train_data['len'] = train_data["sentence"].apply(lambda x: len(x))
test_data['len'] = test_data["sentence"].apply(lambda x: len(x))
# print(train_data)

# 获取标签数量分布
# sns.countplot(train_data, x="label")
# plt.show()
# sns.countplot(test_data, x="label")
# plt.show()

# 获取句子长度分布
"""
使用seaborn绘制训练集和测试集的句子长度分布可视化
"""


# 绘制训练集句子长度分布直方图，按标签分组显示
# - data: 指定数据源为train_data数据框
# - x: 横轴使用"len"列（句子长度）
# - kind: 设置图表类型为直方图(histogram)
# - kde: 开启核密度估计曲线
# - rug: 在横轴显示数据点分布刻度线
# - hue: 按"label"列进行分组着色
# - bins: 设置直方图柱体数量为500
# sns.displot(data=train_data, x="len", kind="hist", kde=True, rug=True, hue="label", bins=500)

# 绘制测试集句子长度分布直方图，使用并排柱状图对比
# - multiple='dodge': 设置不同类别间柱状图并排显示（而非堆叠）
# - rug=False: 关闭测试集的横轴刻度线显示
# 其余参数与训练集保持一致以确保可视化风格统一
# sns.displot(data=test_data, multiple='dodge', x="len", kind="hist", kde=True, rug=False, hue="label", bins=500)

# 显示所有绘制的可视化图表
# plt.show()

# 获取正负样本长度散点分布
def get_label_len_scatter(train_data, test_data):
    """
    合并训练集和测试集的句子长度分布可视化
    """
    # 创建标识列区分数据集来源
    train_data['data_type'] = 'Train'
    test_data['data_type'] = 'Test'

    # 合并数据集
    combined_data = pd.concat([train_data, test_data])

    # 使用统一参数绘制双数据集分布
    # sns.displot(
    #     data=combined_data,
    #     x="len",
    #     hue="data_type",  # 用不同颜色区分数据集
    #     kind="hist",
    #     kde=True,
    #     hue_order=["Train", "Test"],  # 强制图例顺序
    #     palette={"Train": "red", "Test": "black"},  # 自定义颜色
    #     multiple="dodge",  # 并排显示直方图
    #     legend=True,              # 显示图例
    #     # bins=500,
    #     height=6,
    #     aspect=1.5
    # )

    # 添加标签分布对比
    sns.stripplot(
        data=combined_data,
        x="label",
        y="len",
        hue="data_type",
        hue_order=["Train", "Test"],  # 强制图例顺序
        palette={"Train": "red", "Test": "black"},  # 自定义颜色
        legend=True,  # 显示图例
        dodge=True,
        size=2
    )

    plt.tight_layout()
    plt.show()


# 获取不同词汇总数统计
def get_vocabulary_count(data, sentence_column="sentence"):
    """
    统计数据集中所有句子的总词汇数（不重复）。

    参数:
        data (DataFrame): 包含文本数据的数据框。
        sentence_column (str): 数据框中包含句子的列名，默认为"sentence"。

    返回:
        int: 数据集中所有唯一词汇的总数。
    """

    # test_words_len = len(set(chain(*map(lambda x: jieba.lcut(x), test_data["sentence"]))))
    # 使用精确分词对每条句子进行分词，并将所有词语收集到一个列表中
    all_words = []
    for sentence in data[sentence_column]:
        words = precise_cut(sentence)
        all_words.extend(words)

    # 使用集合去重，获取唯一词汇列表
    unique_words = list(set(all_words))

    # 返回唯一词汇数量
    return len(unique_words)


# 获取数据集中的形容词，然后使用WordCloud生成词云
def get_adjective_words_and_generate_wordcloud(data):
    # 初始化一个空列表存储形容词
    adjective_words = []

    # 遍历训练集中每条句子
    for sentence in data['sentence']:
        # 对句子进行分词并标注词性
        for word, flag in pseg.cut(sentence):
            if flag == 'a':  # 如果词性为形容词
                adjective_words.append(word)  # 将形容词添加到列表中

    # 创建WordCloud对象，设置字体路径、背景颜色和最大显示词汇数
    word_cloud = WordCloud(
        font_path=r'C:\Windows\Fonts\simfang.ttf',  # 指定中文字体路径
        background_color='lightpink',  # 设置背景颜色
        max_words=2000  # 最多显示的词数量
    )

    # 使用形容词列表生成词云图像
    img = word_cloud.generate(" ".join(adjective_words))

    # 显示词云图像
    plt.imshow(img)
    plt.axis("off")  # 关闭坐标轴
    plt.grid(False)  # 关闭网格线
    plt.show()


if __name__ == '__main__':
    # "one_hot:", create_one_hot_with_tensorflow(text)
    # train_word2vec_model(text, model_type="cbow", vector_size=100, min_count=1, window=5, epochs=10, thread=4)
    # create_embedding(unique, 4)

    # get_label_len_scatter(train_data, test_data)

    # 示例调用：计算训练集和测试集的词汇总数
    # train_vocab_count = get_vocabulary_count(train_data)
    # test_vocab_count = get_vocabulary_count(test_data)
    #
    # print(f"训练集词汇总数: {train_vocab_count}")  # 12162
    # print(f"测试集词汇总数: {test_vocab_count}")  # 6857

    # get_adjective_words_and_generate_wordcloud(train_data)
    get_adjective_words_and_generate_wordcloud(test_data)
