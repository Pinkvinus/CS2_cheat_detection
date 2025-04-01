# Counter Strike 2 Cheat Detection Dataset


## Overview

The **CS2CD (Counter-Strike 2 Cheat Detection)** dataset is an anonymised dataset comprised of Counter-Strike 2(CS2) gameplay at a variety of skill-levels with cheater annotations. This dataset contains NUMBER CS2 matches with no cheater present, and NUMBER matches CS2 matches with at least one cheater present.

## Dataset structure

The dataset is partitioned into data with at least one cheater present, and data with no cheaters present. 

> [!Warning]
> Only files, containing at least one VAC(Valve Anti-cheat)-banned player, have been manually labelled and verifyed. Hence, **cheaters may be present in the data without cheaters**.
> When examining a subset of NUMBER data points in the set of matches with no VAC-banned players, it was discovered that in NUMBER% of players in these matches were not presenting any cheater-like behaviour.
> When examining a subset of NUMBER data points in the set of matches with with at least one VAC-banned players, it was discovered that in NUMBER% of players in these matches were not presenting any cheater-like behaviour. This is possibly due to CS2 using [trust factor match making](https://help.steampowered.com/en/faqs/view/00EF-D679-C76A-C185).
> Hence, it was decided, that resources were best spent with labelling data containing at least one VAC-banned player.

### Root folder

- `no_cheater_present` : Folder containing data where no cheaters are present.
- `with_cheater_present` : Folder containing data with at least one cheater present.
- `README.md`: This documentation file

### Data files

Each data point(counter strike match) is captured in 2 files: 

| Filetype | Sorting |Data Description |
|----------|---------|-------------|
| `.json`  | Events  | The data is stored by the event type. Each occurence of an event consequently stores the tick, in which the event occured. Note, that this file also contains general game information, such as the cheater labelling, map, and server settings. |
| `.csv`   | Ticks   | The data is contained as a seried of events, also known as ticks. Each tick has 10 rows containing data on the 10 players. |

## Data source

The data is scraped from the website [csstats.gg](https://csstats.gg/) using the `ALL MATCHES` page as an entry point for scraping.

## Data processing






### Data anonymisation

Some data was deemed possibly sensitive and omitable


The following is the complete list of data, not included in the dataset:

- crosshair_code
- player_name
- player_steamid
- music_kit_id
- leader_honors
- teacher_honors
- friendly_honors
- agent_skin
- user_id
- active_weapon_skin
- custom_name
- orig_owner_xuid_low
- orig_owner_xuid_high
- fall_back_paint_kit
- fall_back_seed
- fall_back_wear
- fall_back_stat_track
- weapon_float
- weapon_paint_seed
- weapon_stickers
- xuid
- networkid
- PlayerID
- address

The following data is the complete list of data, altered, in the dataset:

- name
- user_name
- names
- steamid
- user_steamid
- attacker_name
- attacker_steamid
- victim_name
- victim_steamid
- active_weapon_original_owner
- assister_name
- assister_steamid
- approximate_spotted_by


Data added from scraping process:
- Map
- Average rank
- Server
- Matchmaking type
- Cheater label