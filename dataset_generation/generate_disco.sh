#!/bin/bash
# Full pipeline to create the DISCO dataset:
# 1. Download the speech and noise signals
# 2. Create the room configurations and corresponding RIRs; convovlve the speech and noise signals in the room
# 3. Mix the speech and noise at desired SNR.

# Example here is given for 10/3 files (train/test).
# This can be easily changed, but for a big number of files, advised it to parallelize the process.

# 1.
cd pre_generation || exit
bash download_librispeech.sh
bash download_noises_from_zenodo.sh

# 2.
echo "Convolve the signals"
cd ../generation || exit
n_train=10
rir_start_train=1
n_test=3
rir_start_test=11001
echo "  train"
python convolve_signals.py --dset train --scenario random --rirs ${rir_start_train} ${n_train} -d ../../dataset/disco/
echo "  test"
python convolve_signals.py --dset test --scenario random --rirs ${rir_start_test} ${n_test} -d ../../dataset/disco/

# 3.
echo "Mix the signals"
noise=ssn   # Also possible to replace by "it" or "fs"
echo "  train"
python mix_convolved_signals.py  -scenario random --rirs ${rir_start_train} ${n_train} -n ${noise}
echo "  test"
python mix_convolved_signals.py  -scenario random --rirs ${rir_start_test} ${n_test} -n ${noise}
