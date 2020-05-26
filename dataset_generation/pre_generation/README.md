### pre_generation
This folders contains:
 * [download_freesound_queries.py](./download_freesound_queries.py): A python script to download sounds 
    from [freesound](https://freesound.org/) either by requesting specific queries or by requesting a list
    of ids.
    
    __Example__:
    
    ```
    python download_freesound_queries.py --download_type serial --query_by query --save_dir tmp --min_duration 5.5
   ```
 * [ids_per_category.csv](./ids_per_category.csv): An example of (pre-selected) list of freesounds IDs
    and their corresponding category/tag/sound type.
    
    __Example__:
    
    ```
   python download_freesound_queries.py --download_type parallel --query_by id  --file_of_ids ids_per_category.csv \
                                          --save_dir tmp --min_duration 5.5
   ```
