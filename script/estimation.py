# -*- coding: utf-8 -*-

import sys

import numpy as np
from scipy import stats

from sent2prob import sentences2probs


# 確率のリストを受け取って別れたかどうかを推定します。


def main(probs, width, threshold, n_tweets):

    window = np.ones(width) / float(width)
    sma_lis = np.convolve(probs, window, 'valid')

    broken_up = False
    for i in range(len(sma_lis) - 1):
        if sma_lis[i + 1] - sma_lis[i] > threshold:
            broken_up = True

    if broken_up:
        print('!')
    print(stats.hmean(probs[:n_tweets]))

if __name__ == '__main__':
    model = '/home/asano/twitter_model3'
    probs = sentences2probs(sys.stdin, model)

    width = 5  # 移動平均の窓幅
    threshold = 0.1  # 移動平均の降下量の閾値
    n_tweets = 10  # 恋人がいる確率を出すのに使うツイート数
    main(probs, width, threshold, n_tweets)
