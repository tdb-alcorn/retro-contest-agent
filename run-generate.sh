#!/usr/bin/env bash

for i in `seq 0 46`; do
    for j in `seq 0 10`; do
        python -m train.generate $i --save ../data/retro-contest-data --episodes 1 --save-suffix $j
    done
done
