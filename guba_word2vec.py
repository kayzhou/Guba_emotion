import glob
import json
from collections import defaultdict

from gensim import corpora, models
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
        if in_name != '0.txt': continue
        for i, line in enumerate(open(in_name)):
            words = []
            if i > 100: break
            if line.strip() == '':
                continue
            s = line.split('\t')[1]
            for w in thu.cut(s):  # 分词
                w = w[0]
                # if w in stop_word:
                # continue
                words.append(w)

            texts.append(words)

    # Count word frequencies
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [ [token for token in text if frequency[token] > 1] for text in texts ]
    return processed_corpus


processed_corpus = get_train_data()
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
print(dictionary.token2id)


new_doc = "等一个涨停"
new_vec = dictionary.doc2bow(new_doc)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
print(tfidf[dictionary.doc2bow("system minors".lower().split())])
