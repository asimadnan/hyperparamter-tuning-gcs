#!/bin/bash

if [ "$1" == "train" ]; then
    shift
    exec python train.py "$@"
elif [ "$1" == "predict" ]; then
    shift
    exec python predict.py "$@"
else
    echo "Usage: $0 {train|predict} [options]"
    exit 1
fi
