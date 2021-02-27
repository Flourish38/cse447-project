#!/usr/bin/env bash
epoch_delta=2
model_dir=$1
iters=$2
config="$model_dir/config.json"
data_dir="/local1/d0/447-data"

source /homes/iws/d0/miniconda3/etc/profile.d/conda.sh
conda activate 447
for i in $(seq 1 $iters); do
    echo "Starting iter ${i} of $iters"
    scripts/ds.sh /local1/d0/447-data/train /local1/d0/447-data/train_new
    epochs=`grep num_epochs $config`
    if [[ $epochs =~ ([0-9]*), ]]; then
        epoch=${BASH_REMATCH[1]}
    fi
    sed -i "s/\"num_epochs\": [0-9]*,/\"num_epochs\": $(expr $epoch + 2),/" $config
    allennlp train $config -s $model_dir --recover
done
