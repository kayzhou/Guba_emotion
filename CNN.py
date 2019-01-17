#-*- coding: utf-8 -*-

"""
Created on 2018-12-13 11:07:53
@author: https://kayzhou.github.io/
"""

import sys
import numpy as np
from gensim.models import Word2Vec
import word2vecReader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim
import logging
# logging.basicConfig(filename="log/train-11302018.log", format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)
logging.basicConfig(format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)

from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report


class Config:

    def __init__(self):
        self.train_file = "data/labelled_split/labels_text.txt"
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
        print("Loading wv2 ...")
        return Word2Vec.load("model/sgns.financial.word")

    def wv1(self, line):
        v = np.zeros(40 * 400).reshape(40, 400)
        words = line.strip().split(" ")
        _index = 0
        for w in words:
            if _index >= 40:
                break
            if w in self._wv1.wv:
                v[_index] = self._wv1.wv[w]
                _index += 1
        return v

    def wv2(self, line):
            v = np.zeros(40 * 400).reshape(40, 400)
            words = line.strip().split(" ")
            _index = 0
            for w in words:
                if _index >= 40:
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
        count = 0

        labels = []
        X = []
        for line in self._file:
            try:
                label, sentence = line.strip().split("\t")
            except ValueError:
                continue

            label = int(label.strip())
            sequence1 = self.wv1(sentence)
            sequence2 = self.wv2(sentence)
            labels.append(label)
            X.append([sequence1, sequence2])
            count += 1
            if count % (self._batch_size * 100) == 0:
                np.save("/media/alex/data/train_data/X_{}.npy".format(int(count /
                self._batch_size / 100)), np.array(X))
                np.save("/media/alex/data/train_data/Y_{}.npy".format(int(count /
                self._batch_size / 100)), np.array(labels))
                labels = []
                X = []
                print(count)

    def __iter__(self):
        self._reset()
        return self

    def _fill_buffer(self):
        if self._count == 0 and self._file_num <= 189:
            self._buffer = []
            # print("load file {} ...".format(self._file_num))
            X = np.load("/media/alex/data/train_data/X_{}.npy".format(self._file_num))
            Y = np.load("/media/alex/data/train_data/Y_{}.npy".format(self._file_num))
            self._file_num += 1
            self._count += Y.shape[0]
            for i in range(Y.shape[0]):
                self._buffer.append((Y[i], X[i]))
            self._buffer_iter = iter(self._buffer)
            # print("loading finished.")

    def __next__(self):
        self._fill_buffer() # 每次读1024个batch作为buffer
        if self._count == 0: # After filling, still empty, stop iter!
            raise StopIteration

        label_batch = []
        sequence_batch = []

        for label, sequence in self._buffer_iter:
            self._count -= 1
            label_batch.append(label)
            sequence_batch.append(sequence)
            if len(label_batch) == self._batch_size:
                break
        return {"sequences": torch.Tensor(sequence_batch), "labels": torch.LongTensor(label_batch)}

    def _reset(self):
        self._buffer = None
        self._count = 0
        self._file_num = 1
        self._buffer = []
        self._buffer_iter = None

    def save_testdata(self):
        if not self._wv1:
            self._wv1 = self.read_wv1()
        if not self._wv2:
            self._wv2 = self.read_wv2()

        labels = []
        sequences = []
        for line in open("data/0-test.txt"):
            labels.append(0)
            sequences.append([self.wv1(line), self.wv2(line)])
        for line in open("data/1-test.txt"):
            labels.append(1)
            sequences.append([self.wv1(line), self.wv2(line)])
        np.save("/media/alex/data/train_data/X_test.npy", np.array(sequences))
        np.save("/media/alex/data/train_data/Y_test.npy", np.array(labels))

    def get_testdata(self):
        return {"sequences": torch.Tensor(np.load("/media/alex/data/train_data/X_test.npy")),
                "labels": torch.LongTensor(np.load("/media/alex/data/train_data/Y_test.npy"))}


class Dataset2:
    def __init__(self, filepath, batch_size):
        self._file = open(filepath)
        self._wv1 = self.read_wv1()
        self._wv2 = self.read_wv2()
        self._batch_size = batch_size

        self._file.seek(0)
        self._buffer = []
        self._buffer_iter = None
        self._buff_count = 0
        self._file_num = 0

        self._reset()

    def wv1(self, line):
        v = np.zeros(40 * 400).reshape(40, 400)
        words = line.strip().split(" ")
        _index = 0
        for w in words:
            if _index >= 40:
                break
            if w in self._wv1.wv:
                v[_index] = self._wv1.wv[w]
                _index += 1
        return v

    def wv2(self, line):
            v = np.zeros(40 * 400).reshape(40, 400)
            words = line.strip().split(" ")
            _index = 0
            for w in words:
                if _index >= 40:
                    break
                if w in self._wv2:
                    v[_index] = self._wv2[w]
                    _index += 1
            return v

    def read_wv1(self):
        print("Loading wv1 ...")
        return Word2Vec.load("model/word2vec.mod")

    def read_wv2(self):
        print("Loading wv2 ...")
        return word2vecReader.Word2Vec.load_word2vec_format(
            "/media/alex/data/word2vec_twitter_model/word2vec_twitter_model.bin", binary=True)

    # 迭代时候每次先调用__iter__，初始化
    # 接着调用__next__返回数据
    # 如果没有buffer的时候，就补充数据_fill_buffer
    # 如果buffer补充后仍然为空，则停止迭代

    def __iter__(self):
        self._reset()
        return self

    def _fill_buffer(self):
            if self._buff_count > 0:
                return 1

            train_filename = "train_data/train_{:0>2d}".format(0)
            # 遍历文件
            with open(train_filename) as f:
                for line in f:
                    try:
                        label, sentence = line.strip().split("\t")
                    except ValueError:
                        continue
                    label = int(label.strip())
                    sequence1 = self.wv1(sentence)
                    sequence2 = self.wv2(sentence)

                    self._buff_count += 1
                    self._buffer.append((label, [sequence1, sequence2]))

            self._file_num += 1
            self._buffer_iter = iter(self._buffer)
            self._buffer = []

            if self._file_num > 18:
                return 0
            else:
                return 1

    def __next__(self):
        if self._fill_buffer() == 0:
            raise StopIteration

        label_batch = []
        sequence_batch = []
        for label, sequence in self._buffer_iter:
            self._buff_count -= 1
            label_batch.append(label)
            sequence_batch.append(sequence)
            if len(label_batch) == self._batch_size:
                break
        return {"sequences": torch.Tensor(sequence_batch), "labels": torch.LongTensor(label_batch)}

    def _reset(self):
        self._file.seek(0)
        self._buffer = []
        self._buffer_iter = None
        self._buff_count = 0
        self._file_num = 0

    def get_testdata(self):
        labels = []
        sequences = []
        for line in open("train_data/0-test.txt"):
            labels.append(0)
            sequences.append([self.wv1(line), self.wv2(line)])
        for line in open("train_data/1-test.txt"):
            labels.append(1)
            sequences.append([self.wv1(line), self.wv2(line)])
        return torch.LongTensor(labels), torch.Tensor(sequences)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # 2 in- channels, 32 out- channels, 3 * 400 windows size
        self.conv = torch.nn.Conv2d(2, 64, kernel_size=(3, 400), groups=2)
        self.f1 = nn.Linear(1216, 128)
        self.f2 = nn.Linear(128, 64)
        self.f3 = nn.Linear(64, 32)
        self.f4 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = torch.squeeze(out)
        out = F.max_pool1d(out, 2)
        out = out.view(-1, 2 * 32 * 19) # 9 is after pooling
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

    for epoch in range(1, config.num_epochs + 1):
        logging.info("==================== Epoch: {} ====================".format(epoch))
        running_losses = []
        for batch in train_set:

            sequences = batch["sequences"]
            labels = batch["labels"]

        #     # Predict
        #     try:
        #         probs, classes = model(sequences)
        #     except:
        #         print(sequences.size(), labels.size())
        #         print("发生致命错误！")

        #     # Backpropagation
        #     optimizer.zero_grad()
        #     losses = loss_function(probs, labels)
        #     losses.backward()
        #     optimizer.step()

        #     # Log summary
        #     running_losses.append(losses.data.item())
        #     if step % config.summary_interval == 0:
        #         loss = sum(running_losses) / len(running_losses)
        #         writer.add_scalar("train/loss", loss, step)
        #         logging.info("step = {}, loss = {}".format(step, loss))
        #         running_losses = []

        #     step += 1

        # # Classification report
        # test_X = test_set["sequences"]
        # test_labels = test_set["labels"]
        # probs, y_pred = model(test_X)
        # target_names = ['pro-hillary', 'pro-trump']
        # logging.info("{}".format(classification_report(test_labels, y_pred, target_names=target_names)))

        # # Save
        # torch.save(model, "model/11292018-model-epoch-{}.pkl".format(epoch))

        epoch += 1


config = Config()

if __name__ == "__main__":
    train_set = Dataset2(config.train_file, config.train_batch_size)
    test_set = train_set.get_testdata()
    model = CNNClassifier()
    train(model, train_set, test_set)

