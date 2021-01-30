import os
import glob
import pickle
from collections import Counter, defaultdict
import random
import nltk

class SimpleNgramModel:
    """
    Uses character-level ngrams. Super simple and bad and incomplete
    """

    @classmethod
    def load_training_data(cls):
        data = []
        for fname in glob.glob('../train/*'):
            with open(fname) as f:
                for line in f:
                    data.append(line[:-1].lower())
        return data

    @classmethod
    def load_test_data(cls, fname):
        with open(fname) as f:
            return [line[:-1].lower() for line in f]

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write(f'{p}\n')

    def run_train(self, data, work_dir):
        self.ngrams = [defaultdict(Counter) for _ in range(10)]
        for line in data:
            line = ' ' + line
            for n in range(len(self.ngrams)):
                line = ' ' + line
                for ng in nltk.ngrams(line, n + 2):
                    self.ngrams[n][ng[:-1]][ng[-1]] += 1

    def run_pred(self, data):
        preds = []
        for inp in data:
            top_guesses = set()

            inp = ' ' + inp.lower()
            for x in range(len(self.ngrams)):
                inp = ' ' + inp  # leftpad with spaces
                hist = tuple(inp[i] for i in range(len(inp) - x - 1, len(inp)))
                val = self.ngrams[x][hist].most_common(1)

                if len(val) and val[0][1] > 1:
                    top_guesses.add(val[0][0])

            # common English characters
            if len(top_guesses) < 3:
                top_guesses.add(' ')
            if len(top_guesses) < 3:
                top_guesses.add('e')
            if len(top_guesses) < 3:
                top_guesses.add('a')

            top_guesses = random.sample(list(top_guesses), 3)

            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'ngrams.pickle'), 'wb') as f:
            pickle.dump(self.ngrams, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, work_dir):
        model = SimpleNgramModel()
        with open(os.path.join(work_dir, 'ngrams.pickle'), 'rb') as f:
            ngrams = pickle.load(f)
            model.ngrams = ngrams
        return model
