# Counter Strike 2 Cheat Detection Dataset

This is an anonymised, labeled dataset of Counter-Strike 2 gameplay.

## Data source

This data is scraped from the website csstats.gg.

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