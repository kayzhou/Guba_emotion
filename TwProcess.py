#-*- coding: utf-8 -*-

"""
Created on 2018-11-19 14:45:24
@author: https://kayzhou.github.io/
"""


from nltk.corpus import stopwords
import collections
from nltk import precision, recall, BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import ngrams
from itertools import chain
import numpy as np
from string import punctuation


def bag_of_words(words):
    return dict([(word, True) for word in words])

def bag_of_words_and_bigrams(words):
    bigrams = ngrams(words, 2)
    return bag_of_words(chain(words, bigrams))

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))

def bag_of_words_in_set(words, goodwords):
    return bag_of_words(set(words) & set(goodwords))

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)



#==============================================================================
# Custom Tokenizer for tweets
#==============================================================================

from nltk.tokenize.casual import TweetTokenizer, _replace_html_entities, remove_handles, \
                                reduce_lengthening, HANG_RE, WORD_RE, EMOTICON_RE
import re

def normalize_mentions(text):
    """
    Replace Twitter username handles with '@USER'.
    """
    pattern = re.compile(r"(^|(?<=[^\w.-]))@[A-Za-z_]+\w+")
    return pattern.sub('@USER', text)


def normalize_urls(text):
    """
    Replace urls with 'URL'.
    """
    pattern = re.compile(r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")
    # first shorten consecutive punctuation to 3
    # to avoid the pattern to hang in exponential loop in extreme cases.
    text = HANG_RE.sub(r'\1\1\1', text)

    return pattern.sub('URL', text)


def _lowerize(word, keep_all_upper=False):
    if EMOTICON_RE.search(word):
        return word
    elif word.isupper() and keep_all_upper:
        return word
    elif word == 'URL':
        return word
    elif word == '@USER':
        return word
    else:
        return word.lower()

def _stock_code(word):
    if word.isdigit() and len(word) == 6:
        return "[STOCK_CODE]"
    else:
        return word


class CustomTweetTokenizer(TweetTokenizer):
    """ Custom tweet tokenizer based on NLTK TweetTokenizer"""

    def __init__(self, preserve_case=False, reduce_len=True, strip_handles=False,
                 normalize_usernames=True, normalize_urls=True, keep_allupper=True):

        TweetTokenizer.__init__(self, preserve_case=preserve_case, reduce_len=reduce_len,
                                strip_handles=strip_handles)

        self.keep_allupper = keep_allupper
        self.normalize_urls = normalize_urls
        self.normalize_usernames = normalize_usernames

        if normalize_usernames:
            self.strip_handles = False

        if self.preserve_case:
            self.keep_allupper = True


    def tokenize(self, text):
        """
        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings;

        Normalizes URLs, usernames and word lengthening depending of the
        attributes of the instance.

        """
        # Fix HTML character entities:
        text = _replace_html_entities(text)

        # Remove or replace username handles
        if self.strip_handles:
            text = remove_handles(text)
        elif self.normalize_usernames:
            text = normalize_mentions(text)

        if self.normalize_urls:
            # Shorten problematic sequences of characters
            text = normalize_urls(text)

        # Normalize word lengthening
        if self.reduce_len:
            text = HANG_RE.sub(r'\1\1\1', text)
            text = reduce_lengthening(text)

        # Tokenize:
        safe_text = HANG_RE.sub(r'\1\1\1', text)
        words = WORD_RE.findall(safe_text)

        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        # lower words but keep words that are all upper cases
        if not self.preserve_case:
            words = [_lowerize(w, self.keep_allupper) for w in words]

        words = [_stock_code(w) for w in words]

        return words


#==============================================================================
# Direct classification of tweet text
#==============================================================================

def classifyText(text, classifier, tokenizer=CustomTweetTokenizer(),
                  feature_extractor=bag_of_words_and_bigrams,
                  label_inv_mapper={0 : 'neg' , 1 : 'pos'},
                  uncertainty_threshold=0.5,
                  polarity_threshold=0.5):


    labels = [label_inv_mapper[c] for c in  classifier.classes_]


    if isinstance(text, str):
        #single text

        tokens = tokenizer.tokenize(text)

        features = feature_extractor(tokens)

        proba = classifier.predict_proba(features)

        proba = proba.flatten()

        if np.abs(proba[0]-proba[1]) > uncertainty_threshold and np.max(proba) > polarity_threshold:

            predicted_label = labels[np.argmax(proba)]

        else:

            predicted_label = 'N/A'


    elif isinstance(text, list):
        # list of multiple texts

        tokens = map(tokenizer.tokenize, text)
        features = map(feature_extractor, tokens)

        proba = classifier.predict_proba(features)

        predicted_label = np.zeros(len(text), dtype='<U3')
        predicted_label[:] = 'N/A'

        mask = np.all([np.max(proba,axis=1) > polarity_threshold,
                       np.abs(proba[:,0]-proba[:,1]) > uncertainty_threshold], axis=0)

        predicted_label[mask] = [labels[i] for i in np.argmax(proba[mask], axis=1)]


    return predicted_label, proba



# list of official Twitter Clients (should be updated regularly)
official_twitter_clients = ['Twitter for iPhone',
 'Twitter for Android',
 'Twitter Web Client',
 'Twitter for iPad',
 'Mobile Web (M5)',
 'TweetDeck',
 'Mobile Web',
 'Mobile Web (M2)',
 'Twitter for Windows',
 'Twitter for Windows Phone',
 'Twitter for BlackBerry',
 'Twitter for Android Tablets',
 'Twitter for Mac',
 'Twitter for BlackBerry®',
 'Twitter Dashboard for iPhone',
 'Twitter for  iPhone',
 'Twitter Ads',
 'Twitter for  Android',
 'Twitter for Apple Watch',
 'Twitter Business Experience',
 'Twitter for Google TV',
 'Chirp (Twitter Chrome extension)',
 'Twitter for Samsung Tablets',
 'Twitter for MediaTek Phones',
 'Google',
 'Facebook',
 'Twitter for Mac',
 'iOS',
 'Instagram',
 'Vine - Make a Scene',
 'Tumblr']
