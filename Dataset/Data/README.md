# Counter Strike 2 Cheat Detection Dataset

## Overview

The **CS2CD (Counter-Strike 2 Cheat Detection)** dataset is an anonymised dataset comprised of Counter-Strike 2(CS2) gameplay at a variety of skill-levels with cheater annotations. This dataset contains NUMBER CS2 matches with no cheater present, and NUMBER matches CS2 matches with at least one cheater present.

## Dataset structure

The dataset is partitioned into data with at least one cheater present, and data with no cheaters present. 

<div style="border:1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #fff3cd; color: #000;">
  <p><strong style="color: #000">⚠️ Warning: </strong>Data containing no cheaters has not been verified</p>
  <p>Only files, containing at least one VAC(Valve Anti-cheat)-banned player, have been manually labelled and verified. Hence, <strong style="color: #000">cheaters may be present in the data without cheaters</strong>.</p>
  <p>When examining a subset containing 50 data points (demos) with no VAC-banned players, it was discovered that in 97.2% of players in these matches were not presenting any cheater-like behaviour. When examining a subset of 50 data points (demos) in the set of matches with with at least one VAC-banned player, it was discovered that the label precission of the "not cheater" label was 55.6%. This is possibly due to CS2 using <a href="https://help.steampowered.com/en/faqs/view/00EF-D679-C76A-C185" target="_blank" style="color: #007bff; text-decoration: underline;">trust factor match making</a>.</p>

  <p>Hence, it was decided, that resources were best spent with labeling data containing at least one VAC-banned player.</p>

  <p>For more information regarding the data collection see <a href="https://help.steampowered.com/en/faqs/view/00EF-D679-C76A-C185" target="_blank" style="color: #007bff; text-decoration: underline;">Counter-Strike 2 Game data collection with cheat labelling</a> by Mille Mei Zhen Loo & Gert Lužkov.</p>
</div>


### Root folder

- `no_cheater_present` : Folder containing data where no cheaters are present.
- `with_cheater_present` : Folder containing data with at least one cheater present.
- `README.md`: This documentation file

### Data files

Each data point(counter strike match) is captured in 2 files: 

| Filetype | Sorting |Data Description |
|----------|---------| -------------|
| `.csv`   | Ticks   | The data is contained as a series of events, also known as ticks. Each tick has 10 rows containing data on the 10 players. |
| `.json`  | Events  | The data is stored by the event type. Each occurrence of an event consequently stores the tick, in which the event occurred. Note, that this file also contains general game information, such as the cheater labeling, map, and server settings. |

## Loading dataset

The following piece of code loads a single data point in the dataset. The resulting types are the same as if they were a demo parsed by demoparser2.

```python
import pandas as pd
import json

filepath = "Data/no_cheater_present/0"

# Loading csv tick data as a pd.DataFrame
match_0_ticks = pd.read_csv(filepath_or_buffer=filepath+".csv.gz", compression="gzip")

# Loading json event data a list of tuples (str, pd.Dataframe)
def json_2_eventlist(filepath:str) -> list[tuple[str, pd.DataFrame]]: 
    with open(filepath, "r") as f:
        json_data = json.load(f)

    data = []       

    for key, value in json_data.items():
        if isinstance(value, list):
            df = pd.DataFrame(value)
            data.append((key, df))

    return data

match_0_events = json_2_eventlist(filepath=filepath+".json")
```


## Data source

The data is scraped from the website [csstats.gg](https://csstats.gg/) using the `ALL MATCHES` page as an entry point for scraping. This resulted in NUMBER `.dem` files. 

## Data processing

Due to `.dem` files containing sensitive information regarding the users. the data required anonymisation before publishing. This meant extracting the data from the `.dem` files and censoring sensitive data.

In order to extract the data from these files the python library demoparser2 was used[[github](https://github.com/LaihoE/demoparser)][[pypi](https://pypi.org/project/demoparser2/)]. The demoparser parses events and ticks as two separate data types: 

- events : `list[tuple[str, pd.DataFrame]]` with the string describing the event type.
- tick : `pd.DataFrame`

Loading of the data as recommended in the section "[Loading dataset](#loading-dataset)" returns these types as well.

### Data anonymisation

The following is the complete list of **data removed** from the dataset:

- `crosshair_code`
- `player_name`
- `player_steamid`
- `music_kit_id`
- `leader_honors`
- `teacher_honors`
- `friendly_honors`
- `agent_skin`
- `user_id`
- `active_weapon_skin`
- `custom_name`
- `orig_owner_xuid_low`
- `orig_owner_xuid_high`
- `fall_back_paint_kit`
- `fall_back_seed`
- `fall_back_wear`
- `fall_back_stat_track`
- `weapon_float`
- `weapon_paint_seed`
- `weapon_stickers`
- `xuid`
- `networkid`
- `PlayerID`
- `address`
- `name`
- `user_name`
- `victim_name`
- `attacker_name`
- `assister_name`
- `chat_message`

The following data is the complete list of **altered data** in the dataset:

- `steamid`
- `user_steamid`
- `attacker_steamid`
- `victim_steamid`
- `active_weapon_original_owner`
- `assister_steamid`
- `approximate_spotted_by`

Data added from scraping process:
- `map`
- `avg_rank`
- `server`
- `match_making_type`
- `cheater`

## Usage notes

- The dataset is formated in UTF-8 encoding.
- Researchers should **cite this dataset appropriately** in publications
- In the case that all players from a single team quits the match, a single bot is spawned to fill the empty team. This may result in kills where no steamid is present. This is due to the bot not having a steamid.

## Applications

CS2CD is well suited for the following tasks

- Cheat detection
- Player performance prediction
- Match outcome prediction
- Player behaviour clustering
- Weapon effectiveness analysis
- Strategy analysis

## Acknowledgements

A big heartfelt thanks to [Paolo Burelli](http://paoloburelli.com/) for supervising the project.