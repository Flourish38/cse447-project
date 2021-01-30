#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import transformers
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import tqdm

device = torch.device('cuda')
torch.autograd.set_grad_enabled(False)

def dfs_all(cur, result):
    for c in cur[1]:
        n = cur[1][c]
        if n[0] != -1:
            result.append(n[0])
        dfs_all(n, result)

def traverse(cur, text):
    for c in text:
        if c not in cur[1]:
            return None
        cur = cur[1][c]
    return cur

def combine_probs(parent, child, factor):
    for c in child:
        parent[c] = parent.get(c, 0) + child[c] * factor
    return parent

def pop_best(result):
    best = max(result.keys(), key=lambda x: result[x])
    del result[best]
    return best

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, model_name = 'bert-base-multilingual-cased'):
        if model_name is None:
            model_name = 'bert-base-multilingual-cased'
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()

        config, tokenizer, model = self.config, self.tokenizer, self.model
        vocab_trie = [-1, {}]
        seen_chars = set()
        for i, token in enumerate(tokenizer.convert_ids_to_tokens(list(range(config.vocab_size)))):
            if token.startswith('[') and token.endswith(']'):
                continue
            cur = vocab_trie
            seen_chars.update(token)
            for char in token:
                if char not in cur[1]:
                    cur[1][char] = [-1, {}]
                cur = cur[1][char]
            cur[0] = i
        self.vocab_trie = vocab_trie
        self.seen_chars = seen_chars

    def find_allowed_ids(self, head, target):
        cur = self.vocab_trie
        trail = target[len(head):]
        if len(head):
            cur = traverse(cur, '##')
        allowed = []
        for c in trail:
            if c not in cur[1]:
                return allowed, []
            cur = cur[1][c]
            if cur[0] != -1:
                allowed.append(cur[0])
        complete = []
        dfs_all(cur, complete)
        return complete + allowed, complete

    def foo(self, pretext, head, target, space = True):
        tokenizer, model = self.tokenizer, self.model
        vocab_trie = self.vocab_trie

        x = tokenizer(pretext + (' ' if space and len(head) else '') 
                        + head + ' [MASK]',
                    return_tensors='pt')
        x = x.to(device)
        y = model(**x)[0] # get logits
        y = y[0] # remove batch dimension
        y = y[-2] # get mask token
        allowed_ids, complete_ids = self.find_allowed_ids(head, target)
        allowed = torch.tensor(allowed_ids).to(device)
        actual_logits = y[allowed]
        probs = nn.Softmax()(actual_logits)
        
        results = {}
        complete_index = len(target) - len(head) + (2 if len(head) else 0)
        for i, (complete_id, complete_token) in enumerate(zip(complete_ids, tokenizer.convert_ids_to_tokens(complete_ids))):
            char = complete_token[complete_index]
            results[char] = results.get(char, 0) + probs[i]
        
        for i, (allowed_id, allowed_token) in enumerate(zip(allowed_ids, tokenizer.convert_ids_to_tokens(allowed_ids))):
            if i < len(complete_ids):
                # skip since it's already processed
                continue
            next_head = allowed_token
            if head:
                next_head = tokenizer.convert_tokens_to_string([head, allowed_token])
            child_results = self.foo(pretext, next_head, target, space)
            results = combine_probs(results, child_results, probs[i])
            
        return results

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
        for inp in tqdm.tqdm(data):
            # this model just predicts a random character each time
            inp = inp.strip()
            pretext, target = ' '.join(inp.split()[:-1]), inp.split()[-1]
            target = ''.join([c for c in target if c in self.seen_chars])
            result = self.foo(pretext, '', target)
            guesses = []
            for _ in range(3):
                guesses.append(pop_best(result))
            preds.append(''.join(guesses))
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
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


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
