#!/bin/bash

# Check if all required arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Error: Expected 3 arguments: ref, hyp, and out"
    exit 1
fi

# Assign the arguments to variables
ref=$1
hyp=$2
out=$3


# Run the align-text command
/alt/asr/yelkheir/kaldi/egs/gop_ls/s5/../../../src/bin/align-text ark:"$ref" ark:"$hyp" ark,t:- | \
/alt/asr/yelkheir/wav2vec_arabic/scoring/wer_per_utt_details.pl > "$out"
