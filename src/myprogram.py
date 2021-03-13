#!/usr/bin/env python
import time
import_start = time.time()
import os
import sys
from SimpleNgramModel import SimpleNgramModel
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.common import Params

from CustomLM import *

import tqdm
import torch


from PrefixTrie import PrefixTrie

USE_PREFIX_MATCH = True
CONF_THRESHOLD = 0.4

print(f"Took {time.time() - import_start} to import", file=sys.stderr)

print('Torch cuda available:', torch.cuda.is_available())


class LMModel:
    def __init__(self, model_path, batch_size=2048):
        self.vocab = Vocabulary.from_files(os.path.join(model_path, 'vocabulary'))
        cuda_device = 1 if torch.cuda.is_available() else -1
        with open(os.path.join(model_path, 'config.json')) as config_file:
            config = Params(json.load(config_file))
        self.model = Model.load(config, model_path, cuda_device=cuda_device)
        self.model.only_last = True
        self.pred = MyPredictor(self.model, TextReader('.', truncate_last_in=False, include_labels=False))
        self.batch = []
        self.batch_size = batch_size
        # Maps index of sentence to string answer
        self.results = []

        self.bad_for_pred = [idx for token, idx in self.vocab.get_token_to_index_vocabulary('labels').items() if len(token) > 1]

        self.total_conf = 0
        self.total = 0

    def queue(self, idx, sentence):
        self.batch.append((idx, sentence))
        if len(self.batch) >= self.batch_size:
            self.flush()

    def flush(self):
        if len(self.batch) == 0:
            return
        idxs, sentences = zip(*self.batch)
        outputs = self.pred.predict_batch(sentences)
        for idx, output in zip(idxs, outputs):
            probs = output['probs']
            probs[self.bad_for_pred] = -1
            pairs = [(self.vocab.get_token_from_index(token_id, 'labels'), probs[token_id]) for token_id in probs.argpartition(-10)[-10:]]
            #pairs = [(self.vocab.get_token_from_index(token_id, 'labels'), prob) for token_id, prob in enumerate(output['probs'])]
            lower_pairs = {}
            for char, prob in pairs:
                # if len(char) > 1:
                #    continue
                assert len(char) == 1
                char = char.lower()[0]
                lower_pairs[char] = lower_pairs.get(char, 0) + prob
            pairs = list(lower_pairs.items())
            pairs.sort(key=lambda x: x[1], reverse=True)
            ans = ''.join([char for char, _ in pairs[:3]])
            self.results.append((idx, ans))
            self.total_conf += sum((prob for _, prob in pairs[:3]))
        self.total += len(self.batch)
        self.batch = []

    def get_avg_conf(self):
        return self.total_conf / (self.total + 1)


class Ensemble:
    def __init__(self, lm_model=None):
        self.lm_model = lm_model

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_pred(self, data):
        # PREFIX MATCHER
        unknown_indices = range(len(data))
        most_common = [' ', 'e', 'i']

        if USE_PREFIX_MATCH:
            prefix_matcher = PrefixTrie()
            preds, unknown_indices, _, most_common = prefix_matcher.run_pred(
                data)
            unknown_indices = set(unknown_indices)
        else:
            preds = [''] * len(data)

        for idx, inp in enumerate(tqdm.tqdm(data)):
            if idx in unknown_indices:
                self.lm_model.queue(idx, inp)
        self.lm_model.flush()

        lm_model_confidence = self.lm_model.get_avg_conf()
        if lm_model_confidence < CONF_THRESHOLD:
            # LSTM has low confidence, switch to ngram model
            print('LSTM is not confident,', lm_model_confidence, file=sys.stderr)
            ngram_pred = SimpleNgramModel.train_and_test(data, most_common)
            for idx in unknown_indices:
                preds[idx] = ngram_pred[idx]
        else:
            # LSTM is confident
            print('LSTM is confident,', lm_model_confidence, file=sys.stderr)
            for idx, ans in self.lm_model.results:
                preds[idx] = ans

        return preds

    @classmethod
    def load(cls, work_dir):
        return Ensemble(LMModel(work_dir))


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data',
                        default='example/input.txt')
    parser.add_argument(
        '--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        print('No training!')
        exit(1)

        print('Instatiating model')
        model = SimpleNgramModel()
        print('Loading training data')
        train_data = SimpleNgramModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        load_start = time.time()
        model = Ensemble.load(args.work_dir)
        print(f"Took {time.time() - load_start} to load model")
        print('Loading test data from {}'.format(args.test_data))
        test_data = Ensemble.load_test_data(args.test_data)
        print('Making predictions')
        pred_start = time.time()
        pred = model.run_pred(test_data)
        print(f"Took {time.time() - pred_start} to predict all")
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(
            len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
