#!/usr/bin/env python
# Usage: python user-to-happy-prob.py screen_name

import json
import os
import subprocess
import sys

from requests_oauthlib import OAuth1Session


def get_recent_tweets(screen_name, count=5):
    url = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
    params = {'screen_name': screen_name, 'count': count}
    j = json.load(open('/home/ryo-t/.config/wakarepo/keys.json'))
    twitter = OAuth1Session(j['api_key'], j['api_secret'],
                            j['access_token_key'], j['access_token_secret'])
    r = twitter.get(url, params=params)
    j = json.loads(r.text)
    for tweet in j:
        yield tweet['text']


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    screen_name = sys.argv[1]

    for text in get_recent_tweets(screen_name):
        print(text)


if __name__ == '__main__':
    main()
