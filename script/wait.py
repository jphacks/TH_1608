#!/usr/bin/env python

import json
import os
import subprocess
import time

import estimation
from sent2prob import sentences2probs
from user_to_happy_prob import get_recent_tweets


if __name__ == '__main__':
    src_dir = os.path.dirname(os.path.realpath(__file__))
    fname = src_dir + '/../data/buf.txt'
    while True:
        time.sleep(1)
        broken_up = False
        prob = 0.0
        flag = False
        with open(fname) as f:
            text = f.read().strip()
            if text.startswith('@'):
                flag = True
                tweets = list(get_recent_tweets(text, 10))
                ret = subprocess.check_output(
                    u'echo \'{}\' | '
                    u'/home/ryo-t/.pyenv/versions/anaconda3-2.5.0/bin/python '
                    u'{}'.format(u'\n'.join(tweets), src_dir + '/text-normalizer.py'), shell=True)
                normalized_tweets = ret.strip().split('\n')
                probs = sentences2probs(normalized_tweets, '/home/asano/twitter_model3')
                broken_up, prob = estimation.main(probs, width=5, threshold=0.15, n_tweets=10)
                print(text, broken_up, prob)
        if flag:
            with open(fname, 'w') as f:
                f.write(json.dumps({'broken_up': broken_up, 'prob': prob}))
