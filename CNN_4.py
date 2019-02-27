#-*- coding: utf-8 -*-

"""
Created on 2018-12-13 11:07:53
@author: https://kayzhou.github.io/
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter
from torch import autograd, optim
from tqdm import tqdm

from tweet_process import TwPro

# logging.basicConfig(filename="log/log-20190208.log", format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)
logging.basicConfig(format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)


class Config:
    def __init__(self):
        self.train_file = "data/labelled_split/tmp.txt"
        self.train_batch_size = 128
        self.embedding_size = 100 # word embedding

        self.learning_rate = 0.001
        self.window_size = 3
        # self.num_classes = 2

        self.num_epochs = 20
        self.train_steps = None

        self.summary_interval = 10
config = Config()


class Dataset:

    def __init__(self, filepath, batch_size):
        self._file = filepath
        self._batch_size = batch_size
        self._wv1 = None
        self._wv2 = None

    def read_wv1(self):
        print("Loading wv1 ...")
        return Word2Vec.load("model/guba_word2vec.model")

    def read_wv2(self):
        """
        加载ACL2018词向量
        """
        word_vec = {}
        print('Loading wv2 ...')
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
        print('Loaded! There are {} words.'.format(len(word_vec)))
        return word_vec

    def wv1(self, words):
        v = np.zeros(config.embedding_size * 300).reshape(config.embedding_size, 300)
        _index = 0
        for w in words:
            if _index >= config.embedding_size:
                break
            if w in self._wv1.wv:
                v[_index] = self._wv1.wv[w]
                _index += 1
        return v

    def wv2(self, words):
        v = np.zeros(config.embedding_size * 300).reshape(config.embedding_size, 300)
        _index = 0
        for w in words:
            if _index >= config.embedding_size:
                break
            if w in self._wv2:
                v[_index] = self._wv2[w]
                _index += 1
        return v

    def _save(self):
        if not self._wv1:
            self._wv1 = self.read_wv1()
        if not self._wv2:
            self._wv2 = self.read_wv2()

        labels = []
        X = []
        # tw = TwPro()
        for line in tqdm(open(self._file)):
            try:
                label, sentence = line.strip().split("\t")
            except ValueError:
                continue

            # 5- 分类
            # label = label.strip()
            # if label == "-":
            #     label = 0
            # elif label == "x":
            #     continue
            # else:
            #     label = int(label)

            # 2- 分类
            # if label == "-":
            #     label = 0
            # elif label == "x":
            #     continue
            # else:
            #     label = 1

            # 4- 分类
            # label = label.strip()
            # if label == "x" or label == "-":
            #     continue
            # else:
            #     label = int(label) - 1

            # 5- 分类
            label = label.strip()
            if label == "x":
                label = 0
            elif label == "-":
                continue
            else:
                label = int(label)


            labels.append(label)
            words = sentence.split()
            X.append([self.wv1(words), self.wv2(words)])
            # X.append([self.wv1(words), self.wv2(words), len(words)]) # 添加文本长度

        train_rate = 0.8
        Aha = int(len(labels) * train_rate)
        np.save("data/4-train.npy", np.array([X[: Aha], labels[: Aha]]))
        np.save("data/4-test.npy", np.array([X[Aha: ], labels[Aha: ]]))


    def _load(self):
        # disk or data
        if not Path("data/4-train.npy").exists():
            self._save()
        print("Loading ...")
        X_train, y_train = np.load("data/5-train.npy")
        X_test, y_test = np.load("data/5-test.npy")
        print("Finished!")
        return X_train, y_train, X_test, y_test


# -------------- MODEL -------------- #
class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        # 2 in- channels, 32 out- channels, 3 * 300 windows size
        self.conv = torch.nn.Conv2d(2, 64, kernel_size=(3, 300), groups=2)
        self.f1 = nn.Linear(32 * (config.embedding_size - 2), 128)
        self.f2 = nn.Linear(128, 64)
        self.f3 = nn.Linear(64, 32)
        self.f4 = nn.Linear(32, 5)

    def forward(self, x):
        # print(x.size())
        out = self.conv(x)
        out = F.relu(out)
        # print(out.size())
        out = torch.squeeze(out)
        # print(out.size())
        out = F.max_pool1d(out, 2)
        # print(out.size())
        out = out.view(-1, 32 * (config.embedding_size - 2))
        # print(out.size())
        out = F.relu(self.f1(out))
        # print(out.size())
        out = F.relu(self.f2(out))
        out = F.relu(self.f3(out))
        out = F.relu(self.f4(out))
        # print(out.size())

        probs = out
        # probs = F.log_softmax(out, dim=1) # 不需要，因为使用了交叉信息墒
        # probs = out
        # print(probs, probs.size())
        # print(probs.size())
        classes = torch.max(probs, -1)[1]

        return probs, classes


def train(model, train_set, test_set):
    """
    开始训练
    """
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter(log_dir="log")

    epoch = 0
    step = 0

    # making dataset
    train_data = []
    batch_size = config.train_batch_size
    X, y, X_test, y_test = train_set._load()

    X_test = torch.Tensor([X_test[i] for i in range(len(X_test))])
    y_test = torch.LongTensor([y_test[i] for i in range(len(y_test))])

    sequence_batch = []
    label_batch = []
    
    for i in tqdm(range(len(y))):
        label_batch.append(y[i])
        sequence_batch.append(X[i])    

        if len(label_batch) % batch_size == 0: # 存够了
            train_data.append(
                {
                    "sequences": torch.Tensor(sequence_batch),
                    "labels": torch.LongTensor(label_batch)
                }
            )
            label_batch = []
            sequence_batch = []

    # 扫尾工作, 未满batch_size
    if label_batch:
        train_data.append(
            {
                "sequences": torch.Tensor(sequence_batch),
                "labels": torch.LongTensor(label_batch)
            }
        )

    print("finished dataset!")

    for epoch in range(1, config.num_epochs + 1):
        logging.info("================= Epoch: {} =================".format(epoch))
        running_losses = []

        for batch in train_data:
            sequences = batch["sequences"]
            labels = batch["labels"]
            # print(sequences.size(), labels.size())

            # Predict
            probs, classes = model(sequences)

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
        probs, y_pred = model(X_test)
        target_names = ['non-', 'anger', "joy", "sadness", "fear"]
        # target_names = ['anger', "joy", "sadness", "fear"]
        logging.info("{}".format(classification_report(y_test, y_pred, target_names=target_names)))

        epoch += 1

    # Save
    # torch.save(model, "model/shit{}.pkl".format(epoch))


if __name__ == "__main__":
    train_set = Dataset(config.train_file, config.train_batch_size)
    # train_set._save()
    # test_set = train_set.get_testdata()
    model = CNNClassifier()
    train(model, train_set, None)
