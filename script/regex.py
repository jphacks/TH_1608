# coding=utf-8

# python regex.py < data/twitter.all > data/tag_twitter_regex.all

import sys
import json
import re

data = sys.stdin
urlRegex = re.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")
screen_name_regex = re.compile(r"@([a-zA-Z0-9_])+")
renzoku_regex = re.compile(r"[A-Za-z]{2,}")

data = sys.stdin
for line in data:
    line_ls = line.split("\t")
    line_ls[1] = urlRegex.sub("", line_ls[1])
    line_ls[1] = screen_name_regex.sub("", line_ls[1])
    print "{}\t{}".format(line_ls[0], line_ls[1].rstrip("\n"))
