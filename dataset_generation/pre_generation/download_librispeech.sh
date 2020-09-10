#!/bin/bash
# Download LibriSpeech files from http://www.openslr.org/12/

data_dir=../../dataset/
mkdir -p ${data_dir}
declare -a sets=(test-clean.tar.gz train-clean-100.tar.gz train-clean-360.tar.gz)

for set in ${sets[@]}
do
    echo "Downloading "${set}
    wget http://www.openslr.org/resources/12/${set} -P ${data_dir}
    echo "Extracting "${set}
    tar -xf ${data_dir}${set} -C ${data_dir}
    rm ${data_dir}${set}
done

