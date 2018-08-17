#!/bin/bash

# run537classify.sh

# simple bash script to call python & run classifier in this folder
# takes one argument: path to an audio file
# produces .rttm and .frame_prob.mat files in same folder

# get the model if missing
if [ ! -f model.pt ]; then
  wget http://speechkitchen.org/model.pt
fi

python predict.py $1
