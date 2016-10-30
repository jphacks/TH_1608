# -*- coding: utf-8 -*-

import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter as reporter_module
from chainer import cuda

import copy
import sys
from gensim.models import word2vec
from tqdm import tqdm


class BLSTM(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_labels, train=True):
        super(BLSTM, self).__init__(
            embed=F.EmbedID(n_vocab, n_units, ignore_label=-1),
            # fl=L.LSTM(n_units, n_units),
            # bl=L.LSTM(n_units, n_units),
            ll=L.Linear(n_units, 2)
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        pass
        # self.fl.reset_state()
        # self.bl.reset_state()

    def __call__(self, x_list):
        e_list = [self.embed(chainer.Variable(x)) for x in x_list]
        # hf_list = [self.fl(e) for e in e_list]
        # hb_list = [self.bl(e) for e in e_list[::-1]][::-1]
        # y1 = [F.concat((_hf, _hb)) for _hf, _hb in zip(hf, hb)]
        # y1 = hf_list[0]
        # for yi in hf_list[1:]:
        #     y1 = F.maximum(yi, y1)
        # y2 = self.ll(F.relu(y1))
        y = self.ll(F.relu(sum(e_list)))
        return y


class MyIterator(chainer.dataset.Iterator):
    def __init__(self, tuple_dataset, batch_size, repeat=True):
        self.dataset = tuple_dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.iteration = 0

    def __next__(self):
        length = len(self.dataset)

        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        idx = self.iteration % (((length - 1) / self.batch_size) + 1)
        datas = self.dataset[idx * self.batch_size:(idx+1) * self.batch_size]
        self.iteration += 1

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        words, labels = zip(*datas)
        self.padding(words)
        return zip(words, labels)

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / float(len(self.dataset))

    def padding(self, words_list):
        max_len = max(len(ws) for ws in words_list)
        for words in words_list:
            words += [-1] * (max_len - len(words))

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class MyUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(MyUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self._is_new_epoch = False
        self.xp = cuda.cupy if device >= 0 else np

    def update_core(self):
        loss = 0
        self._is_new_epoch = False
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        optimizer.target.predictor.reset_state()

        # datas = sorted(train_iter.__next__(), key=lambda x: -len(x))
        if not self._is_new_epoch:
            self._is_new_epoch = self._iterators['main'].is_new_epoch

        words_list, label_list = zip(*train_iter.__next__())
        words_list = self.xp.array(words_list, dtype=self.xp.int32).T
        label_list = self.xp.array(label_list, dtype=self.xp.int32)
        loss = optimizer.target(words_list, label_list)

        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

    @property
    def is_new_epoch(self):
        return self._is_new_epoch


class MyEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        acc = 0
        count = 0
        observation = {}
        with reporter_module.report_scope(observation):
            for batch in it:
                target.predictor.reset_state()
                w, l = batch[0]
                xp = cuda.cupy if self.device >= 0 else np
                xp_words = [xp.array([_w], xp.int32) for _w in w]
                predict = np.argmax(target.predictor(xp_words).data)
                if predict == l:
                    acc += 1
                count += 1

            summary.add({'main/validation/accuracy': acc / float(count)})

        return summary.compute_mean()


def get_dataset(fn, vocab_dict):
    unk = len(vocab_dict)
    words_list = []
    label_list = []
    with open(fn) as fi:
        for line in tqdm(fi, leave=False):
            line = unicode(line.rstrip('\n'))
            attr = line.split('\t')
            if len(attr) < 2:
                continue
            words_list.append([vocab_dict.get(w, unk) for w in filter(lambda x: len(x) > 0, attr[1].split(' '))])
            label_list.append(1 if attr[0] == '+1' else 0)
    return zip(words_list, label_list)


def main(fi):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--w2v', default='../twitter_model.bin',
                        help='')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    with open('data/vocab_dict.txt', "r") as f_dict:
        vocab = set(unicode(l.split('\t')[0]) for l in f_dict)
        vocab_dict = {w: i for i, w in enumerate(vocab)}
        # vocab_dict['<EOS>'] = len(vocab_dict)
        # vocab_dict['<BOS>'] = len(vocab_dict)
    train = get_dataset('data/twitter.train.sort', vocab_dict)
    val = get_dataset('data/twitter.dev', vocab_dict)
    test = get_dataset('data/twitter.test', vocab_dict)

    n_vocab = len(vocab_dict) + 1
    print('#vocab =', n_vocab)

    w2v_model = word2vec.Word2Vec.load(args.w2v)

    train_iter = MyIterator(train, args.batchsize)
    val_iter = MyIterator(val, 1, repeat=False)
    test_iter = MyIterator(test, 1, repeat=False)

    blstm = BLSTM(n_vocab, args.unit, 2)
    model = L.Classifier(blstm)
    for key, index in vocab_dict.iteritems():
        key = unicode(key)
        if key in w2v_model:
            blstm.embed.W.data[index] = w2v_model[key]

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = MyUpdater(train_iter, optimizer, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()
    eval_model.train = False
    trainer.extend(MyEvaluator(
        val_iter, eval_model, device=args.gpu,
        eval_hook=lambda _: eval_model.predictor.reset_state()), priority=100)

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')), priority=90)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/accuracy', 'main/validation/accuracy']
    ), trigger=(1, 'epoch'), priority=80)
    trainer.extend(extensions.ProgressBar(
        update_interval=1 if args.test else 10), priority=0)
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter' + '_{.updater.epoch}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    print('test')
    eval_model.predictor.reset_state()
    evaluator = MyEvaluator(test_iter, eval_model, device=args.gpu)
    result = evaluator()
    print('test accuracy:', result['main/validation/accuracy'])


if __name__ == '__main__':
    main(sys.stdin)
