# If the training is operated on a remote machine, a good practice is to first copy the data
# on this machine to quicker access to it.
# This script loads the files names, writes them into files and copies the corresponding samples
# on the local disk of the remote machine.

# Environment
ppython=/path/to/anaconda3/envs/torch13/bin/python
code_dir=../../disco_theque/dnn

set -v

# Parameters
scene=${1}
noise=${2}
weights=None
nfiles=11001
zsigs=${3}
zfile=${4}
n_eps=150

# Get files to copy
tmp_dir=/tmp/nfurnon/files_to_copy
mkdir -p ${tmp_dir}
$ppython ${code_dir}/data/lists_to_load.py --scene ${scene} --noise ${noise} --zsigs ${zsigs[@]}\
                                           --path_list ${tmp_dir} --n_files ${nfiles} --zfile ${zfile}
files=$(ls ${tmp_dir})
for file in ${files}
do
  sed -i -e 's!/disco/!/disco/./!g' ${tmp_dir}/${file}  # rsync relative part after 'train' only
done

# Copy data in /tmp
dir_data=/tmp/nfurnon/dataset/disco/
mkdir -p ${dir_data}
date
for file in ${files}
do
  rsync --files-from=${tmp_dir}/${file} / ${dir_data} &
done
wait
echo "data copied in /tmp"
date

# Rename the files to load
for file in ${files}
do
  sed -i -e 's!/home/!/tmp/!g' ${tmp_dir}/${file}       # replace /home/ by /tmp/ to load from local disk
done

# Train NN
$ppython -u ${code_dir}/engine/train.py --scene ${scene} --noise $noise --zsigs ${zsigs[@]} -n ${nfiles}\
                                        --zfile ${zfile} -f2l ${tmp_dir} -path ${dir_data} -w ${weights} -epo ${n_eps}

