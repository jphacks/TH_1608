# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np

from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers
import copy

from train import BLSTM


class Predictor(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        xp = cuda.cupy if self.device >= 0 else np

        for batch in it:
            w, l = batch[0]
            xp_words = [xp.array([_w], xp.int32) for _w in w]
            pred = F.softmax(target.predictor(xp_words)).data[0][1]
            print float(pred)


def main(fi):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='model data, saved by train_ptb.py')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    with open('data/vocab_dict.txt', "r") as f_dict:
        vocab = set(unicode(l.split('\t')[0]) for l in f_dict)
        vocab_dict = {w: i for i, w in enumerate(vocab)}

    n_units = args.unit

    blstm = BLSTM(len(vocab_dict) + 1, n_units, 2, train=False)
    model = L.Classifier(blstm)

    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    unk = len(vocab_dict)
    xp = cuda.cupy if args.gpu >= 0 else np
    for line in fi:
        line = unicode(line.rstrip('\n'))
        if not line:
            print ''
            continue
        words = [vocab_dict.get(w, unk) for w in filter(lambda x: len(x) > 0, line.split(' '))]
        xp_words = [xp.array([w], xp.int32) for w in words]
        pred = F.softmax(model.predictor(xp_words)).data[0][1]
        print pred

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main(sys.stdin)
