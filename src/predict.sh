#!/usr/bin/env bash
set -e
set -v
python src/myprogram.py test --work_dir /local1/d0/447-models/test-1 --test_data $1 --test_output $2
