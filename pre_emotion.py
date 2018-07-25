import json
import linecache
import os
import re

import jieba
import numpy as np
from acora import AcoraBuilder

from emotion_cla.emo_cls import classify
from emotion_cla.separate import separate

in_dir = 'data/tweet'
out_dir = 'data/tweet_emo'
builder = AcoraBuilder([line.strip() for line in open('data/emoji.txt')])
ac = builder.build()


def load_labelled():
    lines = set()
    for i in range(5):
        for line in open('data/content_3000/{}.txt'.format(i)):
            lines.add(line.strip())
    return lines
have_lines = load_labelled()


def random_ids(in_name, out_name, lens):
    '''
    随机选择文本的行
    '''
    global have_lines
    out_file = open(out_name, 'a')
    ids = set()
    _max = len(open(in_name).readlines())
    while len(ids) < lens:
        num = int(_max * np.random.random())
        if num in ids:
            continue
        line = linecache.getline(in_name, num)
        y, X = line.strip().split('\t')
        if line not in have_lines:
            ids.add(num)
            out_file.write(X + '\n')
    out_file.close()
    return ids


def pre_label():
    '''
    打预标签
    '''
    for i, in_name in enumerate(os.listdir(in_dir)):
        print(i)
        stock_name = in_name
        in_name = os.path.join(in_dir, in_name)
        for j, line in enumerate(open(in_name)):
            d = json.loads(line)
            d['content_pre_emo'] = classify(separate(d['content']))
            d['title_pre_emo'] = classify(separate(d['title']))
            with open('{}/{}'.format(out_dir, stock_name), 'a') as f:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')


def get_train_data(in_name):
    for line in open(in_name):
        d = json.loads(line.strip())
        content = d['content']
        # title = d['title']
        # t_emo = d['title_pre_emo']
        c_emo = d['content_pre_emo']

        # 标题和内容中要有一个有表情符
        # if not (re.search('\\[\\S+\\]', title) or re.search('\\[\\S+\\]', content)):

        # bingo = False
        # for kw, pos in ac.finditer(content):
        #     bingo = True
        #     break

        # if not re.search('\\[\\S+\\]', content):
        #     print('不满足要求 ...')
        #     continue

        bingo = True
        if bingo:
            # 内容长度5到200
            if 10 < len(content) < 200:
                with open('data/content/{}.txt'.format(c_emo), 'a') as f:
                    f.write(str(c_emo) + '\t' + content + '\n')

        # with open('data/title/{}.txt'.format(t_emo), 'a') as f:
        #     f.write(str(t_emo) + '\t' + title + '\n')


def label_split(in_name):
    """
    分割数据，用于数据标注划分
    """

    index = 0
    for line in open(in_name):
        with open(in_name[:-4] + '-({}).txt'.format(int(index / 500 + 1)), 'a') as f:
            f.write(line)
        print(index, int(index / 500 + 1))
        index += 1


def what_the_fuck():
    """
    将已经标注的数据按情绪分类
    """
    labels = []
    in_dir = 'data/labelled'

    for in_name in os.listdir(in_dir):
        _in = os.path.join(in_dir, in_name)
        # print(_in)
        for i, line in enumerate(open(_in)):
            if line.strip() == '':
                continue
            label = line.split('\t')[0]
            s= line.split('\t')[1]
            # 1234：四种情绪，-：没有情绪，x：不确定
            if label in ['1', '2', '3', '4', '-']:
                if label == '-':
                    label = '0'
                with open('data/labelled_split/{}.txt'.format(label), 'a') as f:
                    f.write(line)


if __name__ == '__main__':
    # for line in open('data/random_ids.txt'):
    # # for line in open('data/_id.txt'):
    #     line = line.strip().split(',')[0]
    #     print(line)
    #     in_name = 'data/tweet_emo/' + line.strip() + '.txt'
    #     get_train_data(in_name)


    # random_ids('data/_id.txt', 100)
    # get_train_data('data/002446.txt')

    for i in range(5):
        random_ids('data/content/{}.txt'.format(i), 'data/content_sample_3000/{}.txt'.format(i), 3000)


    # for i in range(1, 5):
    #     label_split('data/content_3000/{}.txt'.format(i))

    # what_the_fuck()
