# -*- coding: utf-8 -*-
# classias使いますよ
import commands
import re
import sys


# とりあえずclassias-tagするのでサーバで
# 入力は複数ツイートを標準入力で

def n_gram(s, n):
    x = ['_'.join(s[i:i + n]) for i in range((len(s) - n + 1))]
    return x

# 確率のリストを返します


def sentences2probs(sentences, model):
    pat = re.compile(r'[-+]1:(0.\d+)')
    probs = []
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
            "echo +1 {} | classias-tag -m {} -p".format(line, model))
        probs.append(float(pat.match(check).group(1)))
    return probs


# 確率を逐次表示します
def sent2prob(sentences, model):
    pat = re.compile(r'[-+]1:(0.\d+)')
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
            "echo +1 {} | classias-tag -m {} -p".format(line, model))
        print(float(pat.match(check).group(1)))

if __name__ == '__main__':
    model = '/home/asano/twitter_model3'
    sent2prob(sys.stdin, model)
