#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/evaluation
export DATA_DIR=./data/eval

set_save_dir() {
    save_dir=$1/eval/$2
}

set_valid_dir() {
    save_dir=$1/valid/$2
}

export -f set_save_dir
export -f set_valid_dir
