"""Module to download data from freesound.

To get an access token, please follow the instructions at
https://freesound.org/docs/api/authentication.html#oauth2-authentication

"""
from tqdm import tqdm
import argparse
from contextlib import closing
from multiprocessing import Pool
import functools
import numpy as np
import os
import time
import pandas as pd
import freesound

# Global variables -- Categories of sounds to download for --query_by='query' and their respective folders
queries = ["washing machine-beat", "vacuum cleaner-off -on", "mixer,blender kitchen",
           "fan vent", "air conditioner", "dishwasher -door", "baby cry", "fireplace,chimney - wind",
           "rain window", "printer -laser-warming-startup", "water sink"]
dir_names = ["washing_machine", "vacuum_cleaner", "blender", "fan", "fan", "dishwasher", "baby", "fireplace",
             "rain", "printer", "water"]


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



