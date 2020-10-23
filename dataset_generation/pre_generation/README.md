### pre_generation
This folder gathers scripts to download the files needed to generate the dataset of [our paper].

We also provide a [script](download_freesound_queries.py) to download files from [Freesound](freesound.org), with their [API](https://freesound.org/help/developers/).

#### Download LibriMix
```bash
bash download_librispeech.sh
```


#### Download then noise files from Zenodo
```bash
bash download_noises_from_zenodo.sh
```
  
#### Download some sounds from Freesound
We provide a script to download [freesound](freesound.org/) files either from queries or from IDs. An [example file](config.yaml) is given to show how to use the [main script](download_freesound_queries.py).

To get an access token, follow [these instructions](https://freesound.org/docs/api/authentication.html#oauth2-authentication). We will assume that our token is AbcD12eF.

__Example:__ Serial download

```python
python download_freesound_queries AbcD12eF config.yaml
```


