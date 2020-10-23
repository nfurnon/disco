#!/bin/bash
# Launch the speech enhancement script for one RIR. Can be easily modified to loop over several RIRs and/or be parallelized.

# Activate environment
ppython=/path/to/anaconda3/envs/torch13/bin/python
tango=../../disco_theque/speech_enhancement/tango.py
path_to_models=models
# Go to correct working directory
WORKDIR=./
cd $WORKDIR || exit

set -xv

# VARIABLES
scene=${1}      # meeting/living/random
noise=${2}      # it/fs/ssn
model_sc=${3}   # Name of a single-node model (4-char name randomly given in train.py)
model_mc=${4}   # Name of a multi-node model
k=${5}          # ID of RIR to process

vad1=crnn       # can be replaced by 'irm1', 'ivad'
vad2=crnn
sav_dir=${model_sc}_${model_mc}
zsigs=zs_hat    # zs_hat/zn_hat/'zs_hat zn_hat'
msc=${path_to_models}/${model_sc}_model.pt
mmc=${path_to_models}/${model_mc}_model.pt

${ppython} ${tango} -vt ${vad1} ${vad2} -sd ${sav_dir} --rir $k -scene ${scene}\
                    --noise ${noise} --zsigs ${zsigs} -m ${model} ${mod_mc}
