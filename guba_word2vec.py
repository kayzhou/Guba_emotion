import glob
import json
from collections import defaultdict

from gensim import corpora
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec, TfidfModel, FastText
from thulac import thulac
from tqdm import tqdm

thu = thulac(user_dict='data/emo-words.txt', seg_only=True)


def to_words(s):
    return [w[0] for w in thu.cut(s)]


def cut_them():
    in_name = '/home/kayzhou/Project/Guba_analysis/data/content/tweets.txt'
    with open('/home/kayzhou/Project/Guba_analysis/data/content/cuted_tweets.txt', 'w') as f:
        for i, line in tqdm(enumerate(open(in_name))):
            if i % 10000 == 0:
                print(i)
            f.write(' '.join(to_words(line.strip())) + '\n')


def get_train_data():
    """
    加载所有数据
    """
    # stop_word = load_stopword()
    in_name = '/home/kayzhou/Project/Guba_analysis/data/content/cuted_tweets.txt'
    d = []
    for i, line in enumerate(open(in_name)):
        if i % 10000 == 0:
            print(i)
        d.append(line.strip().split(' '))
    return d


if __name__ == '__main__':
    cut_them()
    corpus = LineSentence('/home/kayzhou/Project/Guba_analysis/data/content/cuted_tweets.txt')
    print(type(corpus))

    print('最终开始训练 ... ...')
    model = Word2Vec(corpus, size=300, window=5, min_count=5, workers=8)
    # model = FastText(corpus, size=300, window=5, min_count=5, iter=10)
    model.save("model/guba_word2vec.model")
