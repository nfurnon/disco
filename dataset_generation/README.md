### Generate the DISCO dataset
The DISCO dataset is described in our submitted paper: [DNN-based mask estimation for distributed speech enhancement in spatially unconstrained microphone arrays](https://hal.archives-ouvertes.fr/hal-02985867). It simulates three kinds of everyday life spatial configurations:
 * A random room where microphones and sources can be placed anywhere in a room;
 * A living room where most of the microphones are placed close to the walls;
 * A meeting room where two sources are around a table and four recording devices on the table.

To generate the dataset:
```bash
bash generate_disco.sh
```

If you use this code, please cite the following:
```
@article{furnon2021dnn,
  title={DNN-based mask estimation for distributed speech enhancement in spatially unconstrained microphone arrays},
  author={Furnon, Nicolas and Serizel, Romain and Essid, Slim and Illina, Irina},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={2310--2323},
  year={2021},
  publisher={IEEE}
}
``` 

### Generate the Meetit dataset
The Meetit (MEETing with Interferent Talkers) dataset is described in our other submitted paper: [Distributed speech separation in spatially unconstrained microphone arrays](https://hal.archives-ouvertes.fr/hal-02985794). It simulates typical meeting configurations, where several sources are talking around a table, with their recording devices placed on the same table.
```bash
bash generate_meetit.sh
```

If you use this code, please cite the following:
```
@inproceedings{furnon2021distributed,
  title={Distributed speech separation in spatially unconstrained microphone arrays},
  author={Furnon, Nicolas and Serizel, Romain and Illina, Irina and Essid, Slim},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4490--4494},
  year={2021},
  organization={IEEE}
}
``` 
