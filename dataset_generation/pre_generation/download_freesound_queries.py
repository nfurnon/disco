"""Module to download data from freesound.

To get an access token, please follow the instructions at
https://freesound.org/docs/api/authentication.html#oauth2-authentication

Queries are found by trial and error on the freesound website.

Configuration file example:

.. code-block:: yaml

    # id_file will be used instead of queries if not empty
    id_file: dataset_generation/pre_generation/ids_per_category.csv  # may be left empty

    queries:
        baby: "baby cry"
        fan: ["fan vent", "air conditioning"]

    fields_to_save: ['id']

    min_duration: 3  # If missing in config, a value of 5.5 (s) is assumed

"""

import argparse
from collections import namedtuple
import functools
import logging
from multiprocessing import Pool
import os
import os.path as osp
import sys
import time

from tqdm import tqdm
import freesound
import numpy as np
import pandas as pd
import yaml

import utils


def main(arguments=None):
    """Command line program.

    Args:
        arguments (list[str], optional): Program arguments. Pass ``None`` to parse command line (Default: ``None``)
    """
    args = parse_args(arguments)
    config = Config.from_yaml(args.config)
    logger = utils.set_up_log(level=1)

    if args.num_jobs > 1:
        func_exec = functools.partial(parallel_exec, num_proc=args.num_jobs)
    else:
        func_exec = serial_exec

    inquirer = FreesoundInquirer(args.token)
    if config.id_file:
        requested_data = extract_category_ids(config.id_file)
        get_files = inquirer.ids_to_files
    else:
        requested_data = config.queries
        get_files = inquirer.queries_to_files

    for dir_name, requests in requested_data.items():
        logger.info(f'Downloading category {dir_name}')
        output_dir = os.path.join(args.save_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'freesound_domestic_noises_{dir_name}.csv')
        downloader = functools.partial(limited_download, output_dir=output_dir)
        for files in get_files(requests, config.fields_to_save, min_duration=config.min_duration):
            new_info = [file.json_dict for file in files]
            update_csv(new_info, csv_path, sort_label='id', sep='\t', header=True, index=False)
            files = list(filter(lambda x: not(osp.exists(osp.join(output_dir, f'{x.id}.{x.type}'))), files))
            filenames = [f'{file.id}.{file.type}' for file in files]
            func_exec(downloader, list(zip(files, filenames)))


def parse_args(arguments):
    """Parses program arguments.

    Args:
        arguments (list[str]): Arguments to parse. Pass ``None`` to parse command line arguments.

    Returns:
        argparse.ArgumentParser: Parsed arguments

    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num_jobs', '-nj',
                        help='Number of parallel download. Pass 0 or 1 for serial downloads (Default: 0)',
                        type=int,
                        default=0)
    parser.add_argument('--save_dir', '-sd',
                        help='Directory to save in (Default: \'tmp\')',
                        type=str,
                        default='tmp')
    parser.add_argument('token', action='store',
                        help='Freesound token',
                        type=str)
    parser.add_argument('config', action='store',
                        help='Configuration file in yaml format',
                        type=str)
    args = parser.parse_args(args=arguments)
    return args


class Config(namedtuple("Config", "queries, id_file, fields_to_save, min_duration")):
    """Freesound download configuration.

    Attributes:
        queries (dict[str, list[str]]): Dirname/queries pairs
        id_file (dict[str, list[str]]): Dirname/id_file pairs
        fields_to_save (list[str]): List of fields to save
        min_duration (float): Minimum file duration in seconds

    """

    def __new__(cls, queries=None, id_file=None, fields_to_save=(), min_duration=5.5):
        """Creates new instance

        Args:
            queries (dict[str, Union[str, list[str]]], optional): Dirname/queries pairs (Default: ``None``)
            id_file (dict[str, Union[str, list[str]]], optional): Dirname/id_file pairs (Default: ``None``)
            fields_to_save (list[str], optional): List of fields to save (Default: ())
            min_duration (float, optional) Minimum file duration in seconds (Default: 5.5)
        """
        self = super().__new__(cls, queries, id_file, fields_to_save, min_duration)
        if not bool(queries) and not bool(id_file):
            raise ValueError('At least one of "queries" and "id_file" must be non-empty')
        self._format_inputs()
        return self

    def _format_inputs(self):
        for key, value in self.queries.items():
            if isinstance(value, str):
                self.queries[key] = [value]

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
            authentication_method (str, optional): Authentication method to choose from {'oauth', 'token'}
                (Default: 'oauth')
        """
        super().__init__()
        self.set_token(token, auth_type=authentication_method)

    def queries_to_files(self, queries, fields_to_save, min_duration=5.5):
        """Gets Freesound objects corresponding to `queries`.

        Args:
            queries (str): Queries
            fields_to_save (list[str]): List of fields to save
            min_duration (float, optional): Minimum file duration in s (Default: 5.5)

        Yields:
            list: List of relevant files
        """
        for query in queries:
            page = 1
            while True:
                results = self.text_search(query=query,
                                           filter='duration:[{} TO *]'.format(min_duration),
                                           sort="score",
                                           fields=",".join(fields_to_save),
                                           page_size=150,  # 150 is the maximum
                                           page=page)
                dict_results = results.as_dict()
                if dict_results["next"] is None:
                    break
                page += 1
                yield results

    def ids_to_files(self, ids, fields_to_save, min_duration=5.5):
        """Gets files from list of IDs.

        Args:
            ids (list[str]): List of IDs.
            fields_to_save (list[str]): Fields to save
            min_duration (float, optional): Minimum file duration in s (Default: 5.5)
        """
        # packs of 200 necessary to work with Freesound API (as query encoded in URL)
        for i in range(int(np.ceil(len(ids) / 200))):
            ids_str_ = ids[i * 200:(i + 1) * 200]
            results = self.text_search(query='',
                                       filter=f'duration:[{min_duration} TO *] id:(' \
                                              f'{" OR ".join(ids_str_)})',
                                       sort='score',
                                       fields=','.join(fields_to_save))
            yield results


def extract_category_ids(id_file):
    """Extracts ids per category given in `id_file`.

    Args:
        file (str): CSV file of labelled audio files

    Returns:
        dict[str, list[str]]: Category -> list of ids
    """
    df = pd.read_csv(id_file, dtype='str')
    df = df.iloc[:, 1:]  # skip index column
    df.dropna(inplace=True)
    return df.to_dict(orient='list')


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
        return list(proc.starmap(func, iterable))


def serial_exec(func, iterable):
    """Executes `func` over iterable.

    Args:
        func (callable): Function
        iterable (iterable): Nested array of arguments given to `func`
    """
    return [func(*val) for val in tqdm(iterable)]


def update_csv(data, file_path, sort_label='', **kwargs):
    """Updates csv `file_path` with `data`.

    Duplicate information are dropped before writing to `file_path`.

    Args:
        data (dict[str, list]): New data
        file_path (str): File where file info is dumped
        sort_label (str): Label used to sort the data before dumping. Use ``''``
            to not sort (Default: ``''``)
        kwargs (dict): Optional arguments given to pandas `to_csv` method
    """
    try:
        # Cannot use kwargs as read_csv and to_csv have different interfaces
        sep = kwargs.get('sep', ',')
        previous_data = pd.read_csv(file_path, sep=sep)
    except FileNotFoundError:
        previous_data = pd.DataFrame()
    previous_data = previous_data.append(pd.DataFrame(data))
    previous_data.drop_duplicates(inplace=True)
    if sort_label:
        # use mergesort to always have n*log(n) complexity
        previous_data.sort_values(sort_label, inplace=True, kind='mergesort')
    previous_data.to_csv(file_path, **kwargs)


def limit_exec(function=None, *, max_per_minute=50):
    """Limits number of executions of `function` per minute.

    When `function` has executed `max_per_minute` times within a minute, it goes to sleep until the time remaining to a
    minute has ellapsed.

    Args:
        function (callable, optional): Function (Default: ``None``)
        max_per_minute (int, optional): Maximum number of execution per minute (Default: 50)
    """
    def arg_wrapper(func):
        @functools.wraps(func)
        def time_limited_function(*args, **kwargs):
            if time_limited_function.num_exec == 0:
                time_limited_function.start = time.time()
            res = func(*args, **kwargs)
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

    if function is None:
        return arg_wrapper
    return arg_wrapper(function)


@limit_exec
def limited_download(file, filename, output_dir):
    """Downloads file with a limited number of execution per minute.

    Args:
        file: Freesound file object
        filename (str): Name of file on disk
        output_dir (str): Output directory where file will be stored
    """
    logger = logging.getLogger(__name__)
    logger.info(f'\t\tDownloading: {file.name}')
    try:
        file.retrieve(output_dir, name=filename)
    except freesound.FreesoundException:
        logger.warning(f'Error while downloading {filename}')


if __name__ == '__main__':
    main()
