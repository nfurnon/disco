"""
In order to copy them with bash's `rsync --files-from`, write the files that we need during the training into text
files.
"""
import os
import argparse
import numpy as np
from disco_theque.dnn.utils import get_input_lists


def load_input_lists(lists_folder):
    """Return a list of lists of files to load as expected in the train pipeline
    Args:
        lists_folder (str): folder of the files that gather all the files to load.
    Returns:
        a list of lists as expected in the [training sript](../engine/train.py)
    """
    files = os.listdir(lists_folder)
    files = sorted(files, key=lambda x: int(x.split('.txt')[0].split('_')[-1]))
    output_list = []
    for file in files:
        output_list.append(open(os.path.join(lists_folder, file), 'r').read().splitlines())

    return output_list


def write_input_lists(lists_to_write, output_dir='/tmp/files_to_copy'):
    """
    Write the files in `lists_to_write` into separate files to easy the copying process with rsync --files-from
    Args:
        lists_to_write (list): list of lists. One element in the list corresponds to one text file written in
                               `output_dir`. This text file gathers the element of the list element.
        output_dir (str): name of the directory where the lists files will be written ['/tmp/files_to_copy']
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, cat in enumerate(lists_to_write):
        with open(os.path.join(output_dir, '') + 'list_{}.txt'.format(str(i + 1)), 'w') as f:
            for line in cat:
                f.write(line)
                f.write('\n')


if __name__ == '__main__':
    np.random.seed(26)
    parser = argparse.ArgumentParser(description="Parameters relative to the training data")
    parser.add_argument("--scene",
                        help="Living or meeting room configuration ?",
                        type=str,
                        default=["living"])
    parser.add_argument("--noise",
                        choices=['ssn', 'it', 'fs', 'noit', 'all'],
                        default='ssn')
    parser.add_argument("--zsigs", "-zs",
                        type=str,
                        nargs='+',
                        default=['zs_hat'])
    parser.add_argument('--n_files', '-n',
                        help='Number of sequences to use for training',
                        type=int,
                        default=11001)
    parser.add_argument('--zfile', '-zf',
                        help='Folder where z are saved',
                        type=str,
                        default='oracle')
    parser.add_argument('--path_list', '-pl',
                        default=None)
    parser.add_argument('--path_data', '-path',
                        default=None)
    args = parser.parse_args()
    scene = args.scene
    noise = args.noise
    zsigs = args.zsigs
    zfile = args.zfile
    n_files = args.n_files
    path_list = None if args.path_list == 'None' else args.path_list
    path_data = None if args.path_data == "None" else args.path_data
    # Data
    if path_data is None:
        path_to_dataset = os.path.join('/home', 'nfurnon', 'dataset', 'disco', '')
    else:
        path_to_dataset = path_data
    training_ids = np.arange(1, n_files)
    # Lists
    zsigs = None if zsigs[0] == "None" else zsigs
    lists_to_load = get_input_lists(path_to_dataset, training_ids,
                                    scenes=scene, noise_to_get=noise, z_sigs=zsigs, z_file=zfile)

    # Write files to lists
    write_input_lists(lists_to_load, output_dir=path_list)
