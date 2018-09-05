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


def get_train_data():
    """
    加载所有数据
    """
    texts = []
    # stop_word = load_stopword()
    in_dir = '/home/kayzhou/Project/Guba_analysis/data/content/'
    for in_name in glob.glob(in_dir + '*.txt'):
        # print(in_name)
        # if not in_name.endswith('0.txt'): continue
        for i, line in enumerate(open(in_name)):
            # if i > 100: break
            words = []
            if line.strip() == '':
                continue
            s = line.strip().split('\t')[1]
            # print(s)
            for w in thu.cut(s):  # 分词
                w = w[0]
                # if w in stop_word:
                # continue
                words.append(w)
            texts.append(words)

    # print(texts)
    # Count word frequencies
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    return processed_corpus


def to_words(s):
    words = []
    for w in thu.cut(s):  # 分词
        w = w[0]
        words.append(w)
    return words


processed_corpus = get_train_data()
# print(processed_corpus)
# dictionary = corpora.Dictionary(processed_corpus)
# print(dictionary)
# print(dictionary.token2id)
# new_doc = to_words("开始加仓")
# new_vec = dictionary.doc2bow(new_doc)
# print(new_vec)
# bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# tfidf = TfidfModel(bow_corpus)
# print(tfidf[new_vec])

model = Word2Vec(processed_corpus, size=300, window=8, min_count=5, workers=8)
model.save("word2vec.model")
