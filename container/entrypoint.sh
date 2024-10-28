#!/bin/bash
MODE=${1:-predict}

if [ "$MODE" == "train" ]; then
    shift
    exec python train.py "$@"
elif [ "$MODE" == "predict" ]; then
    shift
    exec python predict.py "$@"
else
    echo "Usage: $0 {train|predict} [options]"
    exit 1
fi
