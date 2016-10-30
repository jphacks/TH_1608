# -*- coding: utf-8 -*-
# とりあえずclassias-tagするのでサーバで
# 入力は複数ツイートを標準入力で

import commands
import re
import sys


def n_gram(s, n):
    x = ['_'.join(s[i:i + n]) for i in range((len(s) - n + 1))]
    return x

# 確率のリストを返します


def sentences2probs(sentences, model):
    probs = []
    i = 0
    for sent in sentences:
        i += 1
        words = sent.rstrip().split()
        features = []
        for word in words:
            features.append(word)

        for x in n_gram(words, 2):
            features.append(x)

        for x in n_gram(words, 3):
            features.append(x)
        line = ' '.join(features)
        check = commands.getoutput(
            "echo '+1 {}' | classias-tag -m {} -p".format(line, model))
        probs.append(float(check.split(':')[1]))
        # なんか遅いので30ツイートくらいでやめます
        if i > 30:
            break
    return probs


# 確率を逐次表示します
def sent2prob(sentences, model):
    for sent in sentences:
        words = sent.rstrip().split()
        features = []
        for word in words:
            features.append(word)

        for x in n_gram(words, 2):
            features.append(x)

        for x in n_gram(words, 3):
            features.append(x)
        line = ' '.join(features)
        check = commands.getoutput(
            "echo '-1 {}' | classias-tag -m {} -p".format(line, model))
        print(float(check.split(':')[1]))

if __name__ == '__main__':
    model = '/home/asano/twitter_model3'
    sent2prob(sys.stdin, model)
