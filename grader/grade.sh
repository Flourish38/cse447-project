#!/usr/bin/env bash
set -e
set -x

DATA=$1
OUT=${2:-output}
mkdir -p $OUT

docker build -t cse447-proj/demo -f Dockerfile .

# Debugging allennlp's docker image
# echo "Finished building!"
# docker run --rm --gpus all -v $PWD/.allennlp:/root/.allennlp cse447-proj/demo test-install
# echo "Finished test install!"

function run() {
  docker run --gpus all --rm \
    -v $PWD/src:/job/src \
    -v $PWD/work:/job/work \
    -v $DATA:/job/data \
    -v $PWD/$OUT:/job/output \
    cse447-proj/demo \
    bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
}

(time run) > $OUT/output 2>$OUT/runtime
python grader/grade.py $OUT/pred.txt $DATA/answer.txt > $OUT/success
