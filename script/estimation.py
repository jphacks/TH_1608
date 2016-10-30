# -*- coding: utf-8 -*-

import sys

from scipy import stats

from sent2prob import sentences2probs

# 確率のリストを受け取って別れたかどうかを推定します。実験的なコードです。
model = '/home/asano/twitter_model3'
probs = sentences2probs(sys.stdin, model)
print probs
recent = stats.hmean(probs[:5])
past = stats.hmean(probs[6:])

if recent < past:
    if recent > 0.5 and past - recent > 0.2:
        print '!'
print 'recent:{}'.format(recent)
print 'past:{}'.format(past)
