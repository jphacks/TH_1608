# coding=utf-8

import sys, re

urlRegex = re.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")
screen_name_regex = re.compile(r"@([a-zA-Z0-9_])+")
tweet = sys.argv[1]

tweet = urlRegex.sub("", tweet)
tweet = screen_name_regex.sub("", tweet)
print tweet,
