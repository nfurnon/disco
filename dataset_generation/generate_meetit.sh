#!/bin/bash
# Full pipeline to create the MEETIT dataset [1]:
# 1. Download LibriSpeech corpus
# 2. Create the room configurations and corresponding RIRs; convovlve the speech signals in the room; save them.

# Example here is given for 10/3 files (train/test).
# This can be easily changed, but for a big number of files, advised is to parallelize the process.
#
# [1] Furnon N., Serizel R., Illina I., Essid Slim.
#     Distributed speech separation in spatially unconstrained microphone arrays 
# 	  https://hal.archives-ouvertes.fr/hal-02985794
#     (submitted)

# 1.
cd pre_generation || exit
bash download_librispeech.sh

# 2.
echo "Convolve the signals"
cd ../gen_meetit || exit
dir_out=../../dataset/meetit/example
n_train=10
rir_start_train=1
n_test=3
rir_start_test=11001
n_src=3
echo "  train"
python convolve_signals.py --dset train\
						   -- rirs ${rir_start_train} ${n_train}\
						   --n_src ${n_src}
						   --dir_out ${dir_out}
echo "  test"
python convolve_signals.py --dset test\
						   -- rirs ${rir_start_test} ${n_test}\
						   --n_src ${n_src}
						   --dir_out ${dir_out}
