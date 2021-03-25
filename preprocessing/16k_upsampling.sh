
#Converts a dir of 16kHz wav files (defined by $1) to 8kHz wav file (output in $2)

#!/bin/bash
mkdir -p $2
for f in "$1"/*
do
        sox $f -r 16000 -c 1   $2/$(basename $f)
        echo 'Converting: ' $(basename $f)
done
#echo "$1"/*

