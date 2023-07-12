from collections import defaultdict

import numpy as np
import pandas as pd
import os
import torch
from nltk import word_tokenize
from torch import nn
from torchvision import transforms
from transformers import AdamW, get_linear_schedule_with_warmup, AutoFeatureExtractor, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split, Dataset
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
from gensim.models import Word2Vec

# 忽略警告信息
warnings.filterwarnings("ignore")

torch.manual_seed(27)
np.random.seed(27)


# 处理jpg图片为三维数组的方法, tokenizer方法定义
# feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义图像预处理变换
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 自定义的Dataset类
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 根据guid(str类型)读取对应的jpg txt文件
def get_info(guid):
    file_path = 'data'
    jpg_path = os.path.join(file_path, f'{int(guid)}.jpg')
    txt_path = os.path.join('processed_data', f'{int(guid)}.txt')
    # jpg = Image.open(jpg_path).resize((600, 1100))
    jpg = image_transform(Image.open(jpg_path))

    with open(txt_path, 'r', encoding='utf-8') as f:
        txt = f.readline().strip().lower()

    return jpg, txt


# 获取训练集和测试集
# 列表中的每个元素格式为{'guid':int, 'tag':int, 'jpg':image, 'txt':str}
def get_data():
    states = {'negative': 0, 'neutral': 1, 'positive': 2}

    train = pd.read_csv('train.txt')
    train_data = []
    for guid, tag in train.values:
        dict_i = {}
        dict_i['guid'] = int(guid)
        dict_i['tag'] = states[tag]
        dict_i['jpg'], dict_i['txt'] = get_info(guid)
        train_data.append(dict_i)

    test = pd.read_csv('test_without_label.txt')
    test_data = []
    for guid, tag in test.values:
        dict_i = {}
        dict_i['guid'] = int(guid)
        dict_i['tag'] = None
        dict_i['jpg'], dict_i['txt'] = get_info(guid)
        test_data.append(dict_i)

    return train_data, test_data


# 处理text 并获得vocab词汇表(word->index)
def process_data(train_data, test_data):
    vocab = defaultdict()

    stop_words = [',', '.', '(', ')', ':', '-', '!', '?', '&', '#', '*', '@', '^', '/', '[', ']', '"', "'", ';', '%',
                  '$']

    for data in train_data:
        tokens = word_tokenize(data['txt'])
        text = []
        for w in tokens:
            if w not in stop_words:  # 过滤掉无意义的标点符号
                if w not in vocab:
                    vocab[w] = len(vocab) + 1  # word -> index
                text.append(vocab[w])
        text.extend([0] * (30 - len(text)))  # padding
        data['txt'] = text[:30]

    for data in test_data:
        tokens = word_tokenize(data['txt'])
        text = []
        for w in tokens:
            if w not in stop_words:  # 过滤掉无意义的标点符号
                if w not in vocab:
                    vocab[w] = len(vocab) + 1  # word -> index
                text.append(vocab[w])
        text.extend([0] * (30 - len(text)))  # padding
        data['txt'] = text[:30]

    return train_data, test_data, vocab


# 合并列表形成张量 处理jpg图片为三维张量 txt文本为max_length长度的张量
def collate_fn(data):
    guids = [i['guid'] for i in data]
    tags = [i['tag'] for i in data]
    # jpgs = [np.array(i['jpg']) for i in data]
    jpgs = [i['jpg'].numpy() for i in data]
    jpgs = np.array(jpgs)
    txts = [i['txt'] for i in data]

    # return guids, None if tags[0] is None else torch.LongTensor(tags), jpgs, txts
    return guids, None if tags[0] is None else torch.LongTensor(tags), torch.Tensor(jpgs), torch.LongTensor(txts)


# 根据训练集和测试集，将训练集划分出验证集，并将三个数据集转化为Dataloader类型
def get_dataloader(train_data, test_data, batch_size=16):
    train_length = int(len(train_data) * 0.8)
    val_length = len(train_data) - train_length
    train_data, val_data = random_split(dataset=train_data, lengths=[train_length, val_length])

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


# 训练Word2Vec模型并存储
def train_Word2Vec(train_data, test_data):
    corpus = []
    train_corpus = [data['txt'] for data in train_data]
    test_corpus = [data['txt'] for data in test_data]
    corpus.extend(train_corpus)
    corpus.extend(test_corpus)

    model = Word2Vec(corpus, window=5, min_count=1, workers=4, vector_size=150)

    model.save('word2vec.model')


if __name__ == '__main__':
    train_data, test_data = get_data()
    train_data, test_data, vocab = process_data(train_data, test_data)
    train_Word2Vec(train_data, test_data)
