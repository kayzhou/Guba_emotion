from sklearn.externals import joblib
from thulac import thulac
import os
import json
import numpy as np
from tqdm import tqdm

thu = thulac(seg_only=True)
clf = joblib.load('emo-LR-v1.model')
in_dir = 'data/tweet'
out_dir = 'data/tweet_emo_v1-20180711'


def load_word_vec():
    """
    加载ACL2018词向量
    """
    word_vec = {}
    print('加载词向量中 ...')
    for i, line in enumerate(open('data/sgns.financial.word')):
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


word_vec = load_word_vec() # 词向量

def emo_predict(sentence):
    def sentence2vector(sentence):
        global word_vec
        vector = np.zeros(300)
        count = 0
        for w in thu.cut(sentence):
            w = w[0]
            if w in word_vec:
                vector += word_vec[w]
                count += 1
            if count > 0:
                vector = vector / count
        return vector.reshape(1, -1)

    X = sentence2vector(sentence)
    y_hat = clf.predict(X)
    return y_hat


'''
打预标签
'''
for i, in_name in tqdm(enumerate(os.listdir(in_dir))):
    print(i, in_name)
    in_name = os.path.join(in_dir, in_name)
    for j, line in enumerate(open(in_name)):
        d = json.loads(line)
        content = d['content']
        if 10 < len(content) < 200:
            y_hat = emo_predict(content)[0]
            with open('data/content/{}.txt'.format(y_hat), 'a') as f:
                f.write(str(y_hat) + '\t' + content + '\n')



