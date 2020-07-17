## DISCO -- DIStributed semi-COnstrained microphone arrays
This repository gathers scripts to:
 * [download files](./dataset_generation/pre_generation) from [freesound](freesound.org/)
 * [Simulate microphone arrays and their recordings](./dataset_generation/generation) using [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics)


### Install

#### In an anaconda environment

To install the content of this package inside an anaconda environment, make sure
that:

1. the environment has been created (see [official conda
   documentation][conda_env] for more info)
2. the environment is currently active

Then follow the steps outlined in the next subsection.

#### Auto-install

Type `make install` from the command line.

__Note__: The installation of `soundfile` may fail as it requires `libsndfile`
to be installed on the machine. To make it succeed, make sure `libsndfile` is
installed before running `make install`.

### To generate the DISCO dataset:
__1. Download freesound files__

Example:
```bash
python dataset_generation/pre_generation/download_freesound_queries.py \
        --num_jobs 4 \
        --save_dir ../../../dataset/freesound/data/train \
        <freesound_token> \
        config.yaml
```

For more information about the program, type

```bash
python dataset_generation/pre_generation/download_freesound_queries.py --help
```

__2. Simulate RIRs and convolve the signals__.
The argument `--rir_nb` makes it possible to parallelize the process.

Example:
```
python dataset_generation/generation/generate_disco.py --dset train \
                                                       --scenario meeting \
                                                       --rir_id 1 \
                                                       --rir_nb 10 \
                                                       --dir_out tmp
```

### Nota bene
You will need the [LibriSpeech](http://www.openslr.org/12/) corpus, accessible from working directory and following the
structure:

```
 ──path_to_librispeech/
    ├──test-clean
    │   └──speaker_te1
    │       ├──chapter1
    │       │   ├──sentence1
    │       │   ├──sentence2
    │       │   │
    │       │   └──sentenceS
    │       │
    │       └──chapterC
    └──train-clean-360
        └──speaker_tr1
            ├──chapter1
            │   ├──sentence1
            │   ├──sentence2
            │   │
            │   └──sentenceS
            │
            └──chapterC
```

[conda_env]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

### Tests & Documentation

#### Tests

Tests provide examples of how functions and classes are to be used in addition
to ensuring that modifications made to the source code are backward compatible.
The test files are all under the directory named `tests`.

To run the tests, simply type `pytest` from the command line.

To generate a coverage report, type `make coverage` from the command line. This
will generate a directory named `coverage_html_report`, which contains the
coverage report.

#### Documentation

To generate the documentation, type `make doc` from the command line.

#### Shortcut

In case one wishes to generate a coverage report as well as the documentation,
one simply needs to type `make` from the command line. Under the hood, the
commands `make coverage` and `make doc` will be run.
