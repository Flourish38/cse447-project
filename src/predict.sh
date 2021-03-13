#!/usr/bin/env bash
set -e
set -v

python src/myprogram.py test --work_dir work/model-final --test_data $1 --test_output $2

# uncomment this when running on nlpg02
#python src/myprogram.py test --work_dir /local1/d0/447-models/model-5-c --test_data $1 --test_output $2
#python src/myprogram.py test --work_dir ~/cs447/cse447-project/model-final --test_data $1 --test_output $2