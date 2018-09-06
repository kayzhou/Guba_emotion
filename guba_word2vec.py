import glob
import json
from collections import defaultdict

from gensim import corpora
from gensim.models import Word2Vec, TfidfModel
from thulac import thulac

thu = thulac(seg_only=True)


def load_stopword():
    """
    加载停用词集合
    """
    return set(json.load(open('data/stopword-zh.json')))


def to_words(s):
    return [w[0] for w in thu.cut(s)]


def get_train_data():
    """
    加载所有数据
    """
    # stop_word = load_stopword()
    in_name = '/home/kayzhou/Project/Guba_analysis/data/content/tweets.txt'
    for i, line in enumerate(open(in_name)):
        words = to_words(line.strip())
        yield words


processed_corpus = get_train_data()
model = Word2Vec(processed_corpus, size=300, window=8, min_count=5, workers=8)
model.save("word2vec.model")
