## DISCO -- DIStributed semi-COnstrained microphone arrays
This repository gathers scripts to:
 * [download files](./dataset_generation/pre_generation) from [freesound](freesound.org/)
 * Simulate microphone arrays and their recordings using [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics)


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

### To generate datasets:
Please refer to the [dedicated section](dataset_generation)


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

#### References
[1] [DNN-based mask estimation for distributed speech enhancement in spatially unconstrained microphone arrays]()
```BibTex
@inproceedings{Furnon2020SE,
    title={DNN-based mask estimation for distributed speech enhancement in spatially unconstrained microphone arrays},
    author={Furnon, Nicolas and Serizel, Romain and Illina, Irina and Essid, Slim},
    year={2020},
    booktitle={submitted to TASLP},
}
```

[2] [Distributed speech separation in spatially unconstrained microphone arrays](https://hal.archives-ouvertes.fr/hal-02985794)
```BibTex
@inproceedings{Furnon2020SE,
    title={Distributed speech separation in spatially unconstrained microphone arrays},
    author={Furnon, Nicolas and Serizel, Romain and Illina, Irina and Essid, Slim},
    year={2020},
    booktitle={submitted to ICASSP},
}
```
