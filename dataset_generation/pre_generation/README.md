### pre_generation
Scripts to download [freesound](freesound.org/) files either from queries or from IDs. An [example file](config.yaml) is given to show how to use the [main script](download_freesound_queries.py)

To get an access token, follow [these instructions](https://freesound.org/docs/api/authentication.html#oauth2-authentication). We will assume that our token is AbcD12eF
__Example 1:__ Serial download

```python
python download_freesound_queries AbcD12eF config.yaml
```


