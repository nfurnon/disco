"""Checks and clean Freesound csv info files.

Assumptions:
    * The audio files listed in a csv file are in the same directory as the csv
      file
    * The audio files have the corresponding ID as their names.

"""

import argparse
import glob
import os.path as osp

import pandas as pd

import utils


def clean_audio_info(arguments=None):
    """Main program.

    Args:
        arguments (list[str]): Arguments. Pass None to parse command line
            arguments (Default: ``None``)
    """
    args = parse_args(arguments)
    pd_write_opts = {'header': True, 'index': False, 'sep': '\t'}
    logger = utils.set_up_log(level=1)

    for csv in glob.iglob(osp.join(args.dir, '**', '*.csv'), recursive=True):
        logger.info(f'Processing: {csv}')
        missing = get_missing(csv, label='id', sep=pd_write_opts['sep'])
        if missing:
            logger.warning(f'Following files have no info: {missing}')
        clean_info(csv, label='id', **pd_write_opts)


def parse_args(arguments):
    """Parses program arguments.

    Args:
        arguments (list[str]): Arguments to parse. Pass ``None`` to parse command line arguments.

    Returns:
        argparse.ArgumentParser: Parsed arguments

    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir',
                        help='Freesound directory to check',
                        type=str)
    args = parser.parse_args(args=arguments)
    return args


def get_missing(csv_path, label='id', **kwargs):
    """Gets names of files on disk not in csv.

    Args:
        csv_path (str): Path to csv file
        label (str): Label of column with filenames (without extension)
            (Default: ``'id'``)
        kwargs (dict): Arguments given to :func:`pd.read_csv`

    Returns:
        list: Missing files sorted in ascending order

    """
    dirname = osp.dirname(csv_path)
    audio_set = get_not_csv_filenames(dirname)

    infos = pd.read_csv(csv_path, **kwargs)
    missing = list(audio_set - set(infos[label].to_list()))

    missing.sort()
    return missing


def get_not_csv_filenames(dirname):
    """Gets all non csv files in `dirname`.

    Args:
        dirname (str): Directory name

    Returns:
        set: Filenames without extensions

    """
    audio_set = set()
    for audio in glob.iglob(osp.join(dirname, '*')):
        name, ext = osp.splitext(audio)
        if ext != '.csv':
            audio_set.add(osp.basename(name))
    return audio_set


def clean_info(csv_path, label='id', **kwargs):
    """Removes rows in `csv_path` without corresponding file on disk.

    Args:
        csv_path (str): Path to csv file
        label (str): Label of column containing filename (without extension)
            (Default: ``'id'``)
        kwargs (dict): Arguments given to :func:`pd.DataFrame.to_csv`

    """
    dirname = osp.dirname(csv_path)
    audio_set = get_not_csv_filenames(dirname)

    sep = kwargs.get('sep', ',')
    infos = pd.read_csv(csv_path, sep=sep)
    common = set(infos[label].to_list()).intersection(audio_set)
    to_keep = infos[label].map(lambda x: x in common)
    infos[to_keep].to_csv(csv_path, **kwargs)


if __name__ == '__main__':
    clean_audio_info()
