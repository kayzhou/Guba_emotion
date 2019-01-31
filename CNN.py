#-*- coding: utf-8 -*-

"""
Created on 2018-12-13 11:07:53
@author: https://kayzhou.github.io/
"""

import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter
from torch import autograd, optim

from tweet_process import TwPro

# logging.basicConfig(filename="log/train-11302018.log", format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)


class Config:

    def __init__(self):
        self.train_file = "data/labelled_split/labels_text_random.txt"
        self.train_batch_size = 128

        self.learning_rate = 0.001
        self.window_size = 3
        self.num_classes = 2

        self.num_epochs = 10
        self.train_steps = None

        self.summary_interval = 100


class Dataset:

    def __init__(self, filepath, batch_size):
        self._file = open(filepath)
        self._batch_size = batch_size
        self._count = 0
        self._file_num = 1

        self._wv1 = None
        self._wv2 = None

        self._reset()

    def read_wv1(self):
        print("Loading wv1 ...")
        return Word2Vec.load("model/guba_word2vec.model")

    def read_wv2(self):
        """
        加载ACL2018词向量
        """
        word_vec = {}
        print('加载词向量中 ...')
        for i, line in enumerate(open('model/sgns.financial.word')):
            if i <= 10:
                continue
            if i > 150000:
                break
            words = line.strip().split(' ')
            word = words[0]
            word_vec[word] = np.array([float(num) for num in words[1:]])
    #         except UnicodeDecodeError:
    #             print("编码问题，行 {}".format(i))
        print('加载词完成！一共 {}个词'.format(len(word_vec)))
        return word_vec

    def wv1(self, words):
        v = np.zeros(100 * 300).reshape(100, 300)
        _index = 0
        for w in words:
            if _index >= 100:
                break
            if w in self._wv1.wv:
                v[_index] = self._wv1.wv[w]
                _index += 1
        return v

    def wv2(self, words):
        v = np.zeros(100 * 300).reshape(100, 300)
        _index = 0
        for w in words:
            if _index >= 100:
                break
            if w in self._wv2:
                v[_index] = self._wv2[w]
                _index += 1
        return v

    # 迭代时候每次先调用__iter__，初始化
    # 接着调用__next__返回数据
    # 如果没有buffer的时候，就补充数据_fill_buffer
    # 如果buffer补充后仍然为空，则停止迭代
    def _save(self):
        if not self._wv1:
            self._wv1 = self.read_wv1()
        if not self._wv2:
            self._wv2 = self.read_wv2()

        self._reset()

        labels = []
        X = []
        tw = TwPro()

        for line in self._file:
            try:
                label, sentence = line.strip().split("\t")
            except ValueError:
                continue

            label = label.strip()
            if label == "-":
                label = 0
            elif label == "x":
                continue

            print(label)
            words = tw.process_tweet(sentence)
            sequence1 = self.wv1(words)
            sequence2 = self.wv2(words)
            labels.append(label)
            X.append([sequence1, sequence2])

        np.save("data/train/X.npy", np.array(X))
        np.save("data/train/Y.npy", np.array(labels))

    def _load(self):
        X = np.load("data/train/X.npy")
        y = np.load("data/train/Y.npy")
        return X, y

    def _reset(self):
        self._buffer = None
        self._count = 0
        self._file_num = 1
        self._buffer = []
        self._buffer_iter = None


# -------------- MODEL -------------- #
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # 2 in- channels, 32 out- channels, 3 * 300 windows size
        self.conv = torch.nn.Conv2d(2, 64, kernel_size=(3, 300), groups=2)
        self.f1 = nn.Linear(6272, 128)
        self.f2 = nn.Linear(128, 64)
        self.f3 = nn.Linear(64, 32)
        self.f4 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = torch.squeeze(out)
        out = F.max_pool1d(out, 2)
        out = out.view(-1, 2 * 32 * 98)  # 98 is after pooling
        out = F.relu(self.f1(out))
        out = F.relu(self.f2(out))
        out = F.relu(self.f3(out))
        out = F.relu(self.f4(out))
        # print(out.size())

        probs = F.softmax(out, dim=1)
        # print(probs)
        classes = torch.max(probs, 1)[1]

        return probs, classes


def train(model, train_set, test_set):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter(log_dir="log")

    epoch = 0
    step = 0

    # making dataset
    train_data = []
    batch_size = 128
    X, y = train_set._load()

    for i in range(len(y)):
        if i == 0:
            label_batch = []
            sequence_batch = []

        elif i % 128 == 0:
            train_data.append({"sequences": torch.Tensor(sequence_batch),
                               "labels": torch.LongTensor(label_batch)})
            label_batch = []
            sequence_batch = []

        label_batch.append(int(y[i]))
        sequence_batch.append(X[i])

    if label_batch:
        train_data.append({"sequences": torch.Tensor(sequence_batch),
                           "labels": torch.LongTensor(label_batch)})

    print("finished dataset!")

    for epoch in range(1, config.num_epochs + 1):
        logging.info(
            "==================== Epoch: {} ====================".format(epoch))
        running_losses = []

        for batch in train_data:

            sequences = batch["sequences"]
            labels = batch["labels"]

            # Predict
            try:
                probs, classes = model(sequences)
            except:
                print(sequences.size(), labels.size())
                print("发生致命错误！")

            # Backpropagation
            optimizer.zero_grad()
            losses = loss_function(probs, labels)
            losses.backward()
            optimizer.step()

            # Log summary
            running_losses.append(losses.data.item())
            if step % config.summary_interval == 0:
                loss = sum(running_losses) / len(running_losses)
                writer.add_scalar("train/loss", loss, step)
                logging.info("step = {}, loss = {}".format(step, loss))
                running_losses = []

            step += 1

        # Classification report
        # test_X = test_set["sequences"]
        # test_labels = test_set["labels"]
        # probs, y_pred = model(test_X)
        # target_names = ['pro-hillary', 'pro-trump']
        # logging.info("{}".format(classification_report(test_labels, y_pred, target_names=target_names)))

        # Save
        torch.save(model, "model/11292018-model-epoch-{}.pkl".format(epoch))

        epoch += 1


config = Config()

if __name__ == "__main__":
    train_set = Dataset(config.train_file, config.train_batch_size)
    # train_set._save()
    # test_set = train_set.get_testdata()
    model = CNNClassifier()
    train(model, train_set, None)
