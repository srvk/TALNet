#!/bin/bash

# runTALNet.sh

# simple bash script to call python & run classifier in this folder
# takes one argument: path to an audio file
# produces .rttm and .frame_prob.mat files in same folder

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
BASEDIR=`dirname $SCRIPT`

DATA=$(readlink -f $1)
OUT=`dirname $DATA`

cd $BASEDIR

# get the model if missing
if [ ! -f model.pt ]; then
  wget http://speechkitchen.org/model.pt
fi

python predict.py $1

filename=$(basename "$1")
dirname=$(dirname "$1")
extension="${filename##*.}"
basename="${filename%.*}"
output=$dirname/$basename.rttm

#echo "basename is " $basename
#exit

mv $output /tmp
sort -V -k3 /tmp/$basename.rttm > $OUT/$basename.rttm
