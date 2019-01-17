#-*- coding: utf-8 -*-

"""
Created on 2018-11-19 14:51:58
@author: https://kayzhou.github.io/
"""
import os
import sqlite3
import sys
import json

from thulac import thulac
from tqdm import tqdm

from TwProcess import CustomTweetTokenizer

thu = thulac(user_dict='data/emo-words.txt', seg_only=True)


class TwPro:

    def __init__(self):
        self.I_am = "best"

    def process_tweet(self, tweet_text, tokenizer=CustomTweetTokenizer(preserve_case=False,
                                        reduce_len=False,
                                        strip_handles=False,
                                        normalize_usernames=False,
                                        normalize_urls=True,
                                        keep_allupper=False)):

        words = " ".join([w[0] for w in thu.cut(tweet_text)])
        tokens = tokenizer.tokenize(words)
        return tokens


if __name__ == "__main__":

    tw = TwPro()

    # with open("data/train.txt", "w") as f:
    #     for line in tqdm(open("data/labelled_split/labels_text_random.txt")):
    #         label, text = line.strip().split("\t")
    #         tweet = tw.process_tweet(text)
    #         f.write(label + "\t" + " ".join(tweet) + "\n")

    with open("data/traindata_for_word2vec.txt", "w") as f:
        for in_name in tqdm(os.listdir("data/guba-tweet")):
            in_name = os.path.join("data/guba-tweet", in_name)
            if in_name.endswith(".txt"):
                for line in open(in_name):
                    text = json.loads(line.strip())["content"]
                    tweet = tw.process_tweet(text)
                    f.write(label + "\t" + " ".join(tweet) + "\n")
