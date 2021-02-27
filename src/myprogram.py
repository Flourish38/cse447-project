#!/usr/bin/env python
import os
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

class LMModel:
    def __init__(self, model_path, batch_size=64):
        self.vocab = Vocabulary.from_files(os.path.join(model_path, 'vocabulary'))
        cuda_device = 0 if torch.cuda.is_available() else -1
        with open(os.path.join(model_path, 'config.json')) as config_file:
            config = Params(json.load(config_file))
        self.model = Model.load(config, model_path, cuda_device = cuda_device)
        self.model.only_last = True
        self.pred = MyPredictor(self.model, TextReader('.', truncate_last_in=False, include_labels=False))
        self.batch = []
        self.batch_size = batch_size
        # Maps index of sentence to string answer
        self.results = []

        self.bad_for_pred = [idx for token, idx in self.vocab.get_token_to_index_vocabulary('labels').items() if len(token) > 1]
    
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
                #if len(char) > 1:
                #    continue
                assert len(char) == 1
                char = char.lower()[0]
                lower_pairs[char] = lower_pairs.get(char, 0) + prob
            pairs = list(lower_pairs.items())
            pairs.sort(key=lambda x: x[1], reverse=True)
            ans = ''.join([char for char, _ in pairs[:3]])
            self.results.append((idx, ans))
        self.batch = []

    

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, lm_model = None):
        self.lm_model = lm_model

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
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

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for idx, inp in enumerate(tqdm.tqdm(data)):
            # this model just predicts a random character each time
            #top_guesses = [random.choice(all_chars) for _ in range(3)]
            #preds.append(''.join(top_guesses))
            self.lm_model.queue(idx, inp)
            preds.append('')
        self.lm_model.flush()
        for idx, ans in self.lm_model.results:
            preds[idx] = ans
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        return MyModel(LMModel(work_dir))


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))