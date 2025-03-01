#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob

import numpy as np


def iterate_lines_over_file(filename):
    with open(filename) as file:
        print(filename)
        for line in file:
            line = line.strip()
            if len(line) < 2:
                continue
            yield line


def iterate_lines(root):
    filenames = [root]
    if os.path.isdir(root):
        filenames = glob.glob(os.path.join(root, '**/*'), recursive=True)

    for filename in filenames:
        for val in iterate_lines_over_file(filename):
            yield val


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('dev_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--num', default=-1, type=int)

    args = parser.parse_args()
    random.seed(0)

    count = sum(1 for _ in iterate_lines(args.dev_dir))
    allowed = None
    if args.num > 0 and args.num < count:
        allowed = set(np.random.choice(count, args.num, replace=False))

    with open(os.path.join(args.out_dir, 'input.txt'), 'w+') as input_file, \
            open(os.path.join(args.out_dir, 'answer.txt'), 'w+') as ans_file:
        for i, line in enumerate(iterate_lines(args.dev_dir)):
            if allowed is not None and i not in allowed:
                continue
            keep = random.randint(2, len(line))
            line = line[:keep]
            input_file.write(line[:-1] + '\n')
            ans_file.write(line[-1] + '\n')
