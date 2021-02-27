#!/usr/bin/env bash
set -e
set -v
python src/myprogram.py test --work_dir /local1/d0/447-models/model-4-c --test_data $1 --test_output $2
