from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import os
import json
from thulac import thulac
from tqdm import tqdm
thu = thulac(user_dict='data/emo-words.txt', seg_only=True)

def to_words(s):
    return [w[0] for w in thu.cut(s)]

# print(common_texts)
in_dir = "/Volumes/Disk_B/data/origin/tweet"

docs = []
for in_name in tqdm(os.listdir(in_dir)):
    if in_name.endswith(".txt"):
        # print(in_name)
        in_name = os.path.join(in_dir, in_name)
        for line in open(in_name):
            d = json.loads(line)
            if d["content"] != "":
                docs.append(to_words(d["content"]))

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
# print(documents)
model = Doc2Vec(documents, vector_size=400, window=3, min_count=10, workers=4)
fname = get_tmpfile("my_doc2vec_model")
model.save(fname)