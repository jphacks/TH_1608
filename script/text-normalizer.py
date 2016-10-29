#/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import groupby
import os
import re
import subprocess
import sys


stoplist = {'する', 'ある', 'いる', 'おる', 'なる', 'れる', 'られる', 'こと',
            'もの', 'これ', 'あれ', 'それ'}
template = (('surface', 0), ('pos', 1), ('pos1', 2), ('base', 7))


def parse_mecab_one_sent(seq):
    def parse_mecab_line_into_dict(line):
        surface, detail = line.split('\t')
        l = [surface] + detail.split(',')
        return {k: l[i] for k, i in template}

    for startswith_eos, lines in groupby(seq, lambda s: s.startswith('EOS')):
        if not startswith_eos:
            return [parse_mecab_line_into_dict(line) for line in lines]


def main():
    raw_sent = str(sys.stdin.read()).strip()

    raw_sent = re.sub(r'\'', '', raw_sent)

    url_regex = re.compile(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+')
    screen_name_regex = re.compile(r'@([a-zA-Z0-9_])+')
    normalized_sent = screen_name_regex.sub('', url_regex.sub('', raw_sent)).strip()

    if not normalized_sent:
        print()
        return

    cmd = 'echo \'{}\' | mecab -d /home/ryo-t/tmp/mecab-ipadic-neologd'.format(normalized_sent)
    parse_result = subprocess.check_output(cmd, shell=True).decode('utf-8')

    print(' '.join([token['base'] for token in parse_mecab_one_sent(parse_result.split('\n')) if
                    token['pos'] in ('名詞', '動詞', '形容詞') and
                    token['pos1'] != '非自立' and
                    token['base'] not in stoplist and
                    token['base'] != '*']))


if __name__ == '__main__':
    main()
