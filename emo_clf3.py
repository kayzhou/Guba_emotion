import os
import json
import torch
import glob
from collections import Counter

import numpy as np
from tqdm import tqdm_notebook as tqdm

from myclf import *

# import jieba
from thulac import thulac
thu = thulac(user_dict='data/emo-words.txt', seg_only=True)

from gensim.models import Word2Vec
model = Word2Vec.load("model/guba_word2vec.model")


def load_stopword():
    """
    加载停用词集合
    """
    return set(json.load(open('data/stopword-zh.json')))

# stop_word = load_stopword()

def load_label_sentence():
    """
    加载原始文本
    """
    sentences = []
    labels = []
    in_dir = 'data/labelled_split/'
    for in_name in glob.glob(in_dir + '*.txt'):
        for i, line in enumerate(open(in_name)):
            if line.strip() == '': continue
            label = line.split('\t')[0]
            s= line.split('\t')[1]
            # 1234：四种情绪，-：没有情绪，x：不确定
            if label in ['1', '2', '3', '4', '-']:
                if label == '-' or label == 'x':
                    labels.append('0')
                else:
                    labels.append(label)
            sentences.append(s)

    return labels, sentences


labels, sentences = load_label_sentence()

def Info_gain_of_term(v_ci, v_ci_t, v_ci_non_t, pr_t):
    """
    计算信息增益，需要每类的概率，句子出现t是Ci类的概率，不出现t是Ci的概率，存在t的概率
    """
    def info_entropy(p):
        if p == 0:
            return 0
        else:
            return -p * np.log(p)

    gain = 0
    for i in range(len(v_ci)):
        gain = gain + (info_entropy(v_ci[i]) - pr_t * info_entropy(v_ci_t[i]) - (1 - pr_t) * info_entropy(v_ci_non_t[i]))
    return gain


def get_word_freq():
    """
    统计高频词汇
    """
    stopwords = load_stopword()
    words_freq = {}
    words_ci = {} # 出现某个词，是某类的概率，此问题有五类
    class_num = 5
    labels_num = [0] * class_num
    labels, sentences = load_label_sentence()

    for y, s in zip(labels, sentences):

        # 统计每个类别的数量
        labels_num[int(y)] += 1
        # 分词
        for w in thu.cut(s):
            w = w[0]
            # 停用词等过滤
            if w == '' or w in stopwords or w.isdigit():
                continue
            elif w in words_freq:
                words_freq[w] += 1
                words_ci[w][int(y)] += 1
            else:
                words_freq[w] = 1
                words_ci[w] = [0] * class_num
                words_ci[w][int(y)] += 1

    # 数量转概率
    num2pro = lambda nums: [num / sum(nums) for num in nums]

    # 每类上的概率
    v_ci = num2pro(labels_num)

    word_gain = {}
    for w in words_ci.keys():
        word_ci = words_ci[w]

        v_ci_t = num2pro(word_ci) # 句子出现t是Ci类的概率

        non_word_ci = [labels_num[i] - word_ci[i] for i in range(class_num)] # 不是t时候的各类数量
        v_ci_non_t = num2pro(non_word_ci) # 句子不出现t是Ci的概率

        pr_t = words_freq[w] / sum(labels_num) # 存在t的概率

        Gt = Info_gain_of_term(v_ci, v_ci_t, v_ci_non_t, pr_t)

        word_gain[w] = Gt


    word_gain = sorted(word_gain.items(), key=lambda d: d[1], reverse=True) # 根据信息增益排序
    with open('data/word_gain_freq.txt', 'w') as f:
        for w, gain in word_gain:
            if words_freq[w] >= 5:
                print(w, gain, words_freq[w], sep='\t', file=f)


def make_features_onehot():

    def load_word_list(first=2400):
        word_list = []
        for i, line in enumerate(open('data/word_gain_freq.txt')):
            if i >= first:
                break
            try:
                w, gain, freq = line.strip().split('\t')
            except ValueError:
                print('读取词向量出错：行 {}'.format(i))
            word_list.append(w)
        print('词向量大小', len(word_list))
        return word_list

    word_list = load_word_list()

    print('---- 我的词表 ----')
    i = 0
    with open('data/train/onehot-180912.txt', 'w') as f:
        for y, s in zip(labels, sentences):
            i += 1
            if not i % 1000:
                print('行 ->', i)
            vec = np.zeros(len(word_list))
            for w in thu.cut(s):
                w = w[0]
                # print(w)
                try:
                    _i = word_list.index(w)
                    vec[_i] = 1
                except ValueError:
                    pass

            f.write(y + '\t' + ','.join(['{:.1f}'.format(num) for num in list(vec)]) + '\n')
    print('总行数：', i)


def load_word_vec():
    """
    加载ACL2018词向量
    """
    word_vec = {}
    print('加载词向量中 ...')
    for i, line in enumerate(open('data/sgns.financial.word')):
        if i <= 10:
            continue
        if i > 200000:
            break
        words = line.strip().split(' ')
        word = words[0]
        word_vec[word] = np.array([float(num) for num in words[1:]])
#         except UnicodeDecodeError:
#             print("编码问题，行 {}".format(i))
    print('加载词完成！一共 {}个词'.format(len(word_vec)))
    return word_vec


def make_features_ACLwv():
    word_vec = load_word_vec()
    i = 0
    # 建立训练文件：ACL的wv
    print('---- ACL wv ----')
    with open('data/train/ACL-180912.txt', 'w') as f:
        for y, s in zip(labels, sentences):
            i += 1
            if not i % 1000:
                print('行 -> {}'.format(i))
            count = 0
            vec = np.zeros(300)

            for w in thu.cut(s): # 对分词结果进行处理
                w = w[0]
    #             if w in stop_word:
    #                 continue
                if w in word_vec:
                    vec += word_vec[w]
                    count += 1

            if count != 0:
                vec = vec / count

    #         if count > 0:
            f.write(y + '\t' + ','.join(['{:.6f}'.format(num) for num in list(vec)]) + '\n')
    print('总行数：', i)


def make_features_mywv():
    i = 0
    # 建立训练文件: 我的wv
    print('---- 我的wv ----')
    with open('data/train/wv-180912.txt', 'w') as f:
        for y, s in zip(labels, sentences):
            i += 1
            if not i % 1000:
                print('行 -> {}'.format(i))
            count = 0
            vec = np.zeros(300)

            for w in thu.cut(s): # 对分词结果进行处理
                w = w[0]
                if w in model.wv:
                    vec += model.wv[w]
                    count += 1

            if count != 0:
                vec = vec / count

    #         if count > 0:
            f.write(y + '\t' + ','.join(['{:.6f}'.format(num) for num in list(vec)]) + '\n')
    print('总行数：', i)


def load_train_data(in_name, num=4):
    """
    加载训练数据
    """
    X = []
    y = []
    for line in open(in_name):
        label, vec = line.strip().split('\t')
        # 高兴
        if num == 3:
            if label == '2':
                label = '1'
            # 没有情绪
            elif label == '0':
                label = '0'
            # 负面
            else:
                label = '-1'
        elif num == 4:
            if label == '0':
                continue

        x = np.array([float(v) for v in vec.split(',')])
        y.append(label)
        X.append(x)
    X = np.array(X)
    y = np.array(y)
    return X, y


def stack_X_y(X1, y1, X2, y2, out_name=0):
    print(X1.shape, y1.shape, X2.shape, y2.shape)
    if len(y1) != len(y2):
        print('两列表长度不同，不同合并。')
        return -1
    _len = len(X1)
    X = []
    for i in range(_len):
        xi= np.hstack([X1[i], X2[i]])
        X.append(xi)
    X = np.array(X)
    y = np.array(y1)

    if out_name != 0:
        with open(out_name, 'w') as f:
            for xi, yi in zip(X, y):
                f.write(yi + '\t' + ','.join(['{:.6f}'.format(num) for num in list(xi)]) + '\n')
    print('合并数据完成。')
    return X, y


def train():
    """
    调参
    """
    # 合并数据
    X1, y1 = load_train_data('data/train/onehot-180912.txt')
    X2, y2 = load_train_data('data/train/ACL-180912.txt')
    X1, y1 = stack_X_y(X1, y1, X2, y2)
    X3, y3 = load_train_data('data/train/wv-180912.txt')
    X, y = stack_X_y(X1, y1, X3, y3, out_name='data/train/all-180912.txt')
    print(X.shape, y.shape)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)


    # 初始化分类器
    test_classifiers = ['LR', 'DT', 'GBDT']
    classifiers = {'NB':naive_bayes_classifier,
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                   'RF':random_forest_classifier,
                   'DT':decision_tree_classifier,
                  'SVM':svm_classifier,
                'SVMCV':svm_cross_validation,
                 'GBDT':gradient_boosting_classifier
    }

    for classifier in test_classifiers:
        print('******************* {} ********************'.format(classifier))
        if classifier == "GBDT":
            clf = GradientBoostingClassifier(learning_rate=0.05, max_depth=5)
            clf.fit(X_train, y_train)
        else:
            clf = classifiers[classifier](X_train, y_train)

        # CV
        print('accuracy of CV:', cross_val_score(clf, X, y, cv=5).mean())

        # 模型评估
        y_pred = []
        for i in range(len(X_test)):
            y_hat = clf.predict(X_test[i].reshape(1, -1))
            y_pred.append(y_hat[0])
        print(classification_report(y_test, y_pred))


def train_model():
    X, y = load_train_data('data/train/train_data_ACL-20180712.txt')
    clf = LogisticRegression(penalty='l2')
    print(X.shape, y.shape)
    clf.fit(X, y)
    # 保存模型
    joblib.dump(clf, "emo-LR-v1.model")



if __name__ == "__main__":
    get_word_freq() # 词分析
    make_features_onehot()
    make_features_ACLwv()
    make_features_mywv()
    train()
    # train_model()
