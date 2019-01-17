from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import os
import ujson as json
from thulac import thulac
from tqdm import tqdm
import sys
thu = thulac(user_dict='local_data/emo-words.txt', seg_only=True)

def to_words(s):
    return [w[0] for w in thu.cut(s) if w[0] != "\n"]

# print(common_texts)
cnt = 0
in_dir = "data/guba-tweet"
with open("data/cuted_tweet.txt", "w") as f:
    for in_name in os.listdir(in_dir):
        if in_name.endswith(".txt"):
            in_name = os.path.join(in_dir, in_name)
            for line in open(in_name):
                d = json.loads(line)
                cnt += 1
                if cnt % 100000 == 0:
                    print(cnt)
                if d["content"] != "":
                    sen = d["content"].replace("\r\n", "")
                    # print(sen)
                    f.write(' '.join(to_words(sen)) + "\n")


# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
# print(documents)
# model = Doc2Vec(documents, vector_size=400, window=3, min_count=10, workers=4)
# fname = get_tmpfile("my_doc2vec_model")
# model.save(fname)
