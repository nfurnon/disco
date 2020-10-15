#!/bin/bash
# Full pipeline to create the MEETIT dataset [1]:
# 1. Download LibriSpeech corpus
# 2. Create the room configurations and corresponding RIRs; convovlve the speech signals in the room; save them.

# Example here is given for 10/3 files (train/test).
# This can be easily changed, but for a big number of files, advised is to parallelize the process.
#
# [1] Furnon N., Serizel R., Illina I., Essid Slim.
#     Distributed node-specific algorithm for speech separation in spatially distributed microphone arrays.
#     (submitted)

# 1.
cd pre_generation || exit
bash download_librispeech.sh

# 2.
echo "Convolve the signals"
cd ../gen_meetit || exit
n_train=10
rir_start_train=1
n_test=3
rir_start_test=11001
echo "  train"
echo "TODO (sorry)"
echo "  test"
echo "TODO (sorry)"
