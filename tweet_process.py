#-*- coding: utf-8 -*-

"""
Created on 2018-11-19 14:51:58
@author: https://kayzhou.github.io/
"""
import sqlite3
from TwProcess import CustomTweetTokenizer
from tqdm import tqdm
from thulac import thulac
from tqdm import tqdm
import sys
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

    with open("data/train.txt", "w") as f:
        for line in tqdm(open("data/labelled_split/labels_text_random.txt")):
            label, text = line.strip().split("\t")
            tweet = tw.process_tweet(text)
            f.write(label + "\t" + " ".join(tweet) + "\n")