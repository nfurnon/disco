## DISCO -- DIStributed semi-COnstrained microphone arrays
This repository gathers scripts to:
 * [download files](./dataset_generation/pre_generation) from [freesound](freesound.org/)
 * [Simulate microphone arrays and their recordings](./dataset_generation/generation) using [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics)

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
### │To generate the Disco dataset:
1. Download freesound files:
```
python dataset_generation/pre_generation/download_freesound_queries.py \
        --download_type parallel --query_by id --file_of_ids dataset_generation/pre_generation/ids_per_category.csv \
        --save_dir trial --min_duration 5.5
```
2. Simulate RIRs and convolve the signals:

The argument `--rir_nb` makes it possible to parallelize the process.
```
python dataset_generation/generation/generate_disco.py --dset train --scenario meeting --rir_id 3 --rir_nb 1 \
                                                       --dir_out tmp
```
