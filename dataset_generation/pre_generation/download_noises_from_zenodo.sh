#!/bin/bash
# Download the zipfile of https://zenodo.org/record/4019030/files/disco_noises.zip, save it and uncompress it in ../../dataset/freesound/data

data_dir=../../dataset/freesound/
mkdir -p ${data_dir}

if [[ -d "${data_dir}data/" ]]; then
	echo "data has already been downloaded"
else
    wget https://zenodo.org/record/4019030/files/disco_noises.zip -P ${data_dir}
    unzip -q ${data_dir}disco_noises.zip -d ${data_dir}
    mv ${data_dir}disco_noises/ ${data_dir}data/
    rm ${data_dir}disco_noises.zip
fi
