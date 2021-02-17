#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SimpleNgramModel import SimpleNgramModel
from PrefixTrie import PrefixTrie

USE_PREFIX_MATCH = True

def ensemble_test(data, ngramModel):
    preds = []
    unknown_indices = range(len(data))

    if USE_PREFIX_MATCH:
        prefix_matcher = PrefixTrie()
        preds, unknown_indices, data = prefix_matcher.run_pred(data)

    ngram_preds = ngramModel.run_pred(data)
    assert len(ngram_preds) == len(data), 'Expected {} predictions but got {}'.format(len(data), len(ngram_preds))

    # TODO: add in other models here

    if USE_PREFIX_MATCH:
        # add in those predictions into preds
        for i in range(len(unknown_indices)):
            preds[unknown_indices[i]] = ngram_preds[i]
    else:
        preds = ngram_preds

    return preds

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
        model = SimpleNgramModel()
        print('Loading training data')
        train_data = SimpleNgramModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading models')
        model1 = SimpleNgramModel.load(args.work_dir)

        print('Loading test data from {}'.format(args.test_data))
        test_data = SimpleNgramModel.load_test_data(args.test_data)

        print('Making predictions')
        pred = ensemble_test(test_data, model1)

        print('Writing predictions to {}'.format(args.test_output))

        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model1.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
