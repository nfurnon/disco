"""Module to download data from freesound.

To get an access token, please follow the instructions at
https://freesound.org/docs/api/authentication.html#oauth2-authentication

"""

import argparse
from collections import namedtuple
import functools
import logging
from multiprocessing import Pool
import os
import sys
import time

from contextlib import closing
from tqdm import tqdm
import freesound
import numpy as np
import pandas as pd
import yaml

def main(arguments=None):
    """Command line program.

    Args:
        arguments (list[str]): Program arguments. Pass ``None`` to parse command line (Default: ``None``)
    """
    args = parse_args(arguments)
    config = Config.from_yaml(args.config)
    inquirer = FreesoundInquirer(args.token)
    set_up_log(level=1)

    if args.num_jobs > 1:
        func_exec = functools.partial(parallel_exec, num_proc=args.num_jobs)
    else:
        func_exec = serial_exec

    if config.id_files:
        requested_data = config.id_files
        get_files = inquirer.id_file_to_files
    else:
        requested_data = config.queries
        get_files = inquirer.query_to_files

    for dir_name, requests in requested_data.items():
        output_dir = os.path.join(args.save_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        for request in requests:
            files = get_files(request, config.fields_to_save, min_duration=args.min_duration)
            csv_path = os.path.join(output_dir, f'freesound_domestic_noises_{dir_name}.csv')
            write_info(files, csv_path, sep='\t', header=True, index=False)
            func_exec(functools.partial(limited_download, output_dir=output_dir), files)


def parse_args(arguments):
    """Parses program arguments.

    Args:
        arguments (list[str]): Arguments to parse. Pass ``None`` to parse command line arguments.

    Returns:
        argparse.ArgumentParser: Parsed arguments

    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_jobs', '-nj',
                        help='Number of parallel download. Pass 0 or 1 for serial downloads',
                        type=int,
                        default=0)
    parser.add_argument('--save_dir', '-sd',
                        help='Directory to save in',
                        type=str,
                        default='tmp')
    parser.add_argument('--min_duration', '-md',
                        help='Minimal duration of the files to download',
                        type=float,
                        default=5.5)
    parser.add_argument('token', action='store',
                        help='Freesound token',
                        type=str)
    parser.add_argument('config', action='store',
                        help='Configuration file in yaml format',
                        type=str)
    args = parser.parse_args(args=arguments)
    return args


class Config(namedtuple("Config", "queries, id_files, fields_to_save")):
    """Freesound download configuration.

    Attributes:
        queries (dict[str, list[str]]): Dirname/queries pairs
        id_file (dict[str, list[str]]): Dirname/id_file pairs
        fields_to_save (list[str]): List of fields to save

    Args:
        queries (dict[str, Union[str, list[str]]]): Dirname/queries pairs
        id_file (dict[str, Union[str, list[str]]]): Dirname/id_file pairs
        fields_to_save (list[str]): List of fields to save
    """

    def __new__(cls, queries=None, id_files=None, fields_to_save=()):
        self = super().__new__(cls, queries, id_files, fields_to_save)
        self._format_inputs()
        return self

    def _format_inputs(self):
        for member in [self.queries, self.id_files]:
            if not member:
                continue
            for key, value in member.items():
                if isinstance(value, str):
                    member[key] = [value]

    @classmethod
    def from_yaml(cls, config_file):
        """Instantiates from yaml file.

        Args:
            config_file (str): Yaml formatted configuration file

        Returns:
            Config: Freesound config
        """
        with open(config_file) as config:
            content = yaml.safe_load(config)
            return cls(**content)


class FreesoundInquirer(freesound.FreesoundClient):
    """Inquirer to Freesound API.

    Adds functionalities to :class:`freesound.FreesoundClient` for easier use

    """
    def __init__(self, token, authentication_method='oauth'):
        """Instantiates class.

        Args:
            token (str): Freesound access token
            authentication_method (str): Authentication method to choose from {'oauth', 'token'} (Default: 'oauth')
        """
        super().__init__()
        self.set_token(token, auth_type=authentication_method)

    def query_to_files(self, query, fields_to_save, min_duration=5.5):
        """Gets Freesound objects corresponding to `query`.

        Args:
            query (str): Query
            fields_to_save (list[str]): List of fields to save
            min_duration (float, optional): Minimum file duration in s (Default: 5.5)

        Returns:
            list: List of relevant files
        """
        list_results = []
        page = 1
        while True:
            results = self.text_search(query=query,
                                       filter='duration:[{} TO *]'.format(min_duration),
                                       sort="score",
                                       fields=",".join(fields_to_save),
                                       page_size=150,  # 150 is the maximum
                                       page=page)
            list_results += list(results)
            dict_results = results.as_dict()
            if dict_results["next"] is None:
                break
            page += 1

        return list_results


    def id_file_to_files(self, id_file, fields_to_save, min_duration=5.5):
        raise RuntimeError('Not implemented yet')


def set_up_log(logfile='', level=0):
    """Sets up master logger.

    Args:
        logfile (str): Log file. Pass empty string to write to std.err (Default: '')
        level (int): Verbosity level (Default: 0)
            * 0: warnings only
            * 1: info and warnings
            * otherwise: debug, info and warnings
    """
    log_format = '[%(levelname)s] %(asctime)s %(funcName)s: %(message)s'
    time_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, time_format)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = [handler]
    if level == 0:
        logger.setLevel(logging.WARNING)
    elif level == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    return logger


def parallel_exec(func, iterable, num_proc):
    """Executes `func` over `num_proc` processes.

    Args:
        func (callable): Pickable function
        iterable (iterable): Values given to `func`
        num_proc (int): Number of processes

    Returns:
        list: Outputs of func for each element in `iterable`
    """
    with Pool(processes=num_proc) as proc:
        return list(tqdm(proc.imap_unordered(func, iterable), total=len(iterable)))


def serial_exec(func, iterable):
    """Executes `func` over iterable.

    Args:
        func (callable): Function
        iterable (iterable): Values given to `func`
    """
    return [func(val) for val in tqdm(iterable)]


def write_info(files, file_path, **kwargs):
    """Dumps Freesound data info into `file_path`.

    Args:
        files (list): Freesound files retrieved via the Freesound API
        file_path (str): File where file info is dumped
        kwargs (dict): Optional arguments given to pandas `to_csv` method
    """
    sound_info = [file.json_dict for file in files]
    if sound_info:
        df = pd.DataFrame(sound_info)
        df.to_csv(file_path, **kwargs)


def limit_exec(function, max_per_minute=50):
    """Limits number of executions of `function` per minute.

    When `function` has executed `max_per_minute` times within a minute, it goes to sleep until the remaining time to a
    minute as ellapsed.

    Args:
        function (callable): Function
        max_per_minute (int, optional): Maximum number of execution per minute (Default: 50)
    """
    @functools.wraps(function)
    def time_limited_function(*args, **kwargs):
        if time_limited_function.num_exec == 0:
            time_limited_function.start = time.time()
        res = function(*args, **kwargs)
        time_limited_function.num_exec += 1

        if time_limited_function.num_exec == max_per_minute:
            end = time.time()
            ellapsed = end - time_limited_function.start
            time_to_sleep = 60 - ellapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            time_limited_function.num_exec = 0
        return res

    time_limited_function.num_exec = 0
    return time_limited_function


@limit_exec
def limited_download(file, output_dir):
    """Downloads file with a limited number of execution per minute.

    Args:
        file:
        output_dir:
    """
    return download(file, output_dir)


def download(file, output_dir):
    logger = logging.getLogger(__name__)
    filename = "{}.{}".format(file.id, file.type)
    if not os.path.exists(os.path.join(output_dir, filename)):
        try:
            logger.info(f'\t\tDownloading: {file.name}')
            file.retrieve(output_dir, name=filename)
        except freesound.FreesoundException:
            logger.warning(f'Error while downloading {filename}')
    else:
        logger.info(f'\t {file.name} already downloaded')


def get_queries(client, **params):
    """Gets text results of a query.

    Args:
      client (freesound.FreesoundClient): Freesound client
      **params (dict): args given to freesound.FreesoundClient().text_search() (query, filter, ...)

    Returns:
      list, int: A list of results, count

    """
    list_results = []
    page = 1
    next_page = True
    while next_page:
        results = client.text_search(**params, page_size=150, page=page)  # 150 is the maximum
        dict_results = results.as_dict()
        list_results += [l for l in results]
        if dict_results["next"] is None:
            next_page = False
        page += 1

    count = dict_results["count"]

    return list_results, count


def download_fs_file(result_dir, sound, cnt=1):
    filename = "{}.{}".format(sound.id, sound.type)
    result_dict = sound.json_dict
    os.makedirs(result_dir, exist_ok=True)
    if not os.path.exists(os.path.join(result_dir, filename)):
        try:
            print("\t\tDownloading:", sound.name)
            sound.retrieve(result_dir, name=filename)
            cnt += 1
        except freesound.FreesoundException:
            print('Error while downloading ' + filename)
            time.sleep(60)
    else:
        print("\t" + sound.name + " already downloaded")

    return result_dict, cnt


def parallel_download(out_dir, fs_sound, n_jobs=6, chunk_size=5):
    """Parallelizes downloading of freesound files.

    Args:
      out_dir (str): Output directory where data is saved
      fs_sound (dict): Freesound object
      n_jobs (int):  Number of parallel download streams (Default: 6)
      chunk_size (int): (Default: 5)

    Returns:
        list: Information about the downloaded files

    """
    infos = []
    with closing(Pool(n_jobs)) as p:
        download_file_alias = functools.partial(download_fs_file, out_dir)
        for val, _ in tqdm(p.imap_unordered(download_file_alias, fs_sound, chunk_size),
                           total=len(fs_sound)):
            infos.append(val)

    return infos


def parallel_infos(out_dir, fs_sound, infos, n_jobs=6, chunk_size=5):
    """Parallelize downloading of freesound files

    Args:
      out_dir: 
      fs_sound: 
      infos: 
      n_jobs:  (Default value = 6)
      chunk_size:  (Default value = 5)

    Returns:

    """
    with closing(Pool(n_jobs)) as p:
        download_file_alias = functools.partial(download_fs_file, out_dir)
        for val, _ in tqdm(p.imap_unordered(download_file_alias, fs_sound, chunk_size),
                           total=len(fs_sound)):
            infos.append(val)

    return infos


def serial_download(list_of_files, saving_dir):
    infos = []
    cnt = 1
    for file in list_of_files:
        f_dict, cnt = download_fs_file(saving_dir, file, cnt)
        infos.append(f_dict)

        if cnt == 50:
            time.sleep(60)
            print("sleep")
            cnt = 1

    return infos


def get_str_ids(file):
    df = pd.read_csv(file, dtype='str')
    categories = list(df.columns.values[1:])
    values = [[] for _ in range(len(categories))]
    for i in range(len(categories)):
        values[i] = df[categories[i]].dropna().values

    return values, categories


def download_from_file_list(files_list, result_dir, dwnl_type, n_jobs=5, chunk_size=6):
    """Download from a file list obtained from freesound request.

    Information about downloaded files are stored in a file named
    `freesound_domestic_noises_{category}.csv`. `category` is the basename of
    `result_dir`.

    Args:
      files_list (list): List of freesound sounds to download
      result_dir (str): Directory where sound files will be stored
      dwnl_type (str): Download type. To choose from ``serial`` and ``parallel``
      n_jobs:  Number of parallel download (Default: 5)
      chunk_size:  (Default: 6)
    """
    category = os.path.split(result_dir)[-1]
    if dwnl_type == 'serial':
        infos = serial_download(files_list, result_dir)
    else:
        infos = parallel_download(result_dir, files_list, n_jobs=n_jobs, chunk_size=chunk_size)

    df = pd.DataFrame(infos)
    df.to_csv(os.path.join(result_dir, "freesound_domestic_noises_" + category + ".csv"),
              sep="\t", header=True, index=False)


def download_by_query(fs_client, out_dir, dwnl_type, fields_to_save,
                      min_duration=5.5, n_jobs=5, chunk_size=6):
    """Downloads the files by sending specific queries to freesound.

    Args:
      fs_client (freesound.FreesoundClient): Freesound client
      out_dir (str): Output directory
      dwnl_type (str): Download type. To choose from ``serial`` and ``parallel``
      fields_to_save (list): Fields to save 
      min_duration (float): Minimum file duration in seconds (Default: 5.5)
      n_jobs: Number of parallel download (Default: 5)
      chunk_size (int):  (Default: 6)

    """
    for query, dir_name in zip(queries, dir_names):
        files_list, cl = get_queries(fs_client,
                                     query=query,
                                     filter='duration:[{} TO *]'.format(min_duration),
                                     sort="score",
                                     fields=",".join(fields_to_save))
        result_dir = os.path.join(out_dir, dir_name)
        download_from_file_list(files_list, result_dir, dwnl_type,
                                n_jobs=n_jobs, chunk_size=chunk_size)


def download_by_id(fs_client, ids_file, out_dir, dwnl_type, fields_to_save,
                   min_duration=5.5, n_jobs=5, chunk_size=6):
    """Downloads the files by sending an ID query to freesound.

    Args:
      fs_client (freesound.FreesoundClient): Freesound client
      ids_file (str): Files of ids
      out_dir (str): Output directory
      dwnl_type (str): Download type. To choose from ``serial`` and ``parallel``
      fields_to_save (list): Fields to save
      min_duration (float): Minimum file duration in seconds (Default: 5.5)
      n_jobs: Number of parallel downloads (Default: 5)
      chunk_size: (Default: 6)

    """
    ids_per_cat, cats = get_str_ids(ids_file)
    for ids_str, dir_name in zip(ids_per_cat, cats):
        for i in range(int(np.ceil(len(ids_str) / 200))):
            ids_str_ = ids_str[i * 200:(i + 1) * 200]
            files_list, cl = get_queries(fs_client,
                                         query="",
                                         filter='duration:[{} TO *] id:('.format(min_duration)
                                                + ' OR '.join(ids_str_) + ')',
                                         sort="score",
                                         fields=",".join(fields_to_save))
            # Download files
            result_dir = os.path.join(out_dir, dir_name)
            download_from_file_list(files_list, result_dir, dwnl_type,
                                n_jobs=n_jobs, chunk_size=chunk_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_type', '-dwnl',
                        help='Serial or parallelized way to download files ?',
                        choices=['serial', 'parallel'],
                        type=str,
                        default='serial')
    parser.add_argument('--query_by', '-q',
                        help='Which way should the freesound query be done ?',
                        choices=['query', 'id'],
                        type=str,
                        default='query')
    parser.add_argument('--file_of_ids', '-f',
                        help='.csv file gathering all the IDs to download, if --query_by="id"',
                        default=None)
    parser.add_argument('--save_dir', '-sd',
                        help='Directory to save in',
                        type=str,
                        default='tmp')
    parser.add_argument('--min_duration', '-md',
                        help='Minimal duration of the files to download',
                        type=float,
                        default=5.5)
    args = parser.parse_args()

    dwnl = args.download_type
    query_by = args.query_by
    file_of_ids = args.file_of_ids
    min_duration = args.min_duration
    out_dir = args.save_dir

    # Instantiate freesound client
    # To get token to *download*, follow https://freesound.org/docs/api/authentication.html#oauth2-authentication
    token = ""
    n_jobs, chunk_size = 6, 5
    client = freesound.FreesoundClient()
    client.set_token(token, "oauth")
    cnt = 1

    fields_to_save = ["id", "name", "type", "license", "username", "duration"]
    # Search files
    if query_by == 'query':
        download_by_query(client, out_dir, dwnl, fields_to_save,
                          min_duration=min_duration, n_jobs=n_jobs, chunk_size=chunk_size)
    else:
        download_by_id(client, file_of_ids, out_dir, dwnl, fields_to_save,
                       min_duration=min_duration, n_jobs=n_jobs, chunk_size=chunk_size)



