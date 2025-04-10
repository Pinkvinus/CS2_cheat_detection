# Dataset creation

This folder contains all scripts regarding the dataset creation.

- `shell.nix` : Contains the requirements for running scripts contained in this folder including it's subfolders.
- `dem2datafile.py` : The "main" file responsible for creation of the dataset.
- `/Data` : The folder containing the final dataset.
    - Note, the `dem2datafile` script requires the following folders that are not pushed to this repository: 
        - `Data/no_cheater_present` : The directory for data without a cheater present (Note that this is not validated)
        - `Data/with_cheater_present` : The directory for data with a cheater present (this has been validated through manual review)
        - `Demo_data` : The directory for the raw data
- `/utils` : Contains all helper functions used in `dem2datafile.py`