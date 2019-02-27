# -*- coding: utf-8 -*-
# Author: Kay Zhou
# Date: 2019-02-27 18:01:25

## find the character

from collections import Counter
from tqdm import tqdm

cnt = Counter()
for line in tqdm(open("data/traindata_for_word2vec.txt")):
    words = set([w for w in line.strip() if w != " "])
    for w in words:
        cnt[w] += 1

for tup in cnt.most_common(10000):
    print(tup[0], tup[1], file="data/charactor_count.txt")
