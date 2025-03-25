from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas
import json
import time

import pandas.core
import pandas.core.frame
from demo_parser_fields import ALL_FIELDS

START_BALANCE = 800
MAX_HEALTH = 100.0
NUM_PLAYERS = 10

SENSITIVE_DATA_REMOVAL = ["crosshair_code",
                          "player_name",
                          "player_steamid",
                          "music_kit_id",
                          "leader_honors",
                          "teacher_honors",
                          "friendly_honors",
                          "agent_skin",
                          "user_id",
                          "active_weapon_skin",
                          "custom_name",
                          "orig_owner_xuid_low",
                          "orig_owner_xuid_high",
                          "fall_back_paint_kit",
                          "fall_back_seed",
                          "fall_back_wear",
                          "fall_back_stat_track",
                          "weapon_float",
                          "weapon_paint_seed",
                          "weapon_stickers",
                          "xuid",
                          "networkid",
                          "PlayerID",
                          "address"
                          ]

SENSITIVE_DATA_REPLACE = ["name",
                          "user_name",
                          "names",
                          "steamid",
                          "user_steamid",
                          "attacker_name",
                          "attacker_steamid",
                          "victim_name",
                          "victim_steamid",
                          "active_weapon_original_owner",
                          "assister_name",
                          "assister_steamid",
                          #"approximate_spotted_by"
                          ] # chat messages and cvar names

parser = -1

def get_fields():
    with open("fields.txt", "r") as f:
        all_fields = [line.strip() for line in f.readlines()]

    return all_fields

def df_to_readable_csv(df:pandas.core.frame.DataFrame, filename:str): 
    """ takes a pandas data frame and a output file name and aligns the columns in the resulting csv such that it is readable"""

    max_lengths = {col: max(df[col].astype(str).apply(len).max(), len(col)) for col in df.columns}

    # Apply str.ljust() to ensure each column has the same width for both headers and values
    df_aligned = df.apply(lambda col: col.astype(str).apply(lambda x: x.ljust(max_lengths[col.name])))

    # Writing the formatted data to a CSV file, with commas separating columns and no quotes
    with open(filename, 'w') as f:
        # Write the header (with commas and aligned)
        f.write(','.join([col.ljust(max_lengths[col]) for col in df.columns]) + '\n')
        
        # Write the data rows (with commas and aligned)
        for index, row in df_aligned.iterrows():
            f.write(','.join([str(row[col]).ljust(max_lengths[col]) for col in df.columns]) + '\n')

def binary_search_sorted_dataframe(df:pandas.core.frame.DataFrame, col, val)-> int:
    """
        Uses binary search to find the first occurance of val in a sorted column(col) of a dataframe(df)
    """
    def binary_search_helper(df:pandas.core.frame.DataFrame , low:int, high:int, col, val)-> int:
        if high >= low:
            mid = low + (high - low) // 2

            if df[col].iloc[mid] == val:
                # Check if this is the first occurrence
                if mid == 0 or df[col].iloc[mid-1] != val:
                    return mid
                else:
                    # Look to the left
                    return binary_search_helper(df, low, mid-1, col, val)
                
            elif df[col].iloc[mid] > val:
                # Look to the left
                return binary_search_helper(df, low, mid-1, col, val)
            else:
                # Look to the right
                return binary_search_helper(df, mid+1, high, col, val)
        else:
            return -1  # If no match is found
        
    return binary_search_helper(df, 0, len(df.index), col, val)

def remove_warmup_rounds(df: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    """removes the ticks corresponding to a warmup round for a given data frame"""

    index = binary_search_sorted_dataframe(df,'is_warmup_period', False)
    return df.loc[index:].reset_index(drop=True)

def get_tick(df:pandas.core.frame.DataFrame, index:int) -> pandas.core.frame.DataFrame:
    """Returns a single tick of the index from the dataframe"""

    i = binary_search_sorted_dataframe(df, 'tick', index)

    return df.loc[i:i+NUM_PLAYERS-1].reset_index(drop=True)

def get_ticks(df:pandas.core.frame.DataFrame, start:int, end:int) -> pandas.core.frame.DataFrame:
    """Returns a dataframe of ticks of the index from the dataframe"""

    s = binary_search_sorted_dataframe(df, 'tick', start)
    t = binary_search_sorted_dataframe(df, 'tick', end)

    return df.loc[s:t+NUM_PLAYERS-1].reset_index(drop=True)

def events_2_json(filename:str):
    """
        
    """
    events = parser.list_game_events()
    event_data = {}

    for e in events:
        df = parser.parse_event(e)

        if "user_steamid" in df.columns:
            df = df.drop(columns=["user_steamid"])

        dict = df.to_dict(orient="records")
        event_data.update({e:dict})

    with open(filename, "w") as f:
        json.dump(event_data, f, indent=4)

    return event_data

def event_list_2_json(list:list, filename:str):
    
    event_data = {}

    for (s, df) in list:
        dict = df.to_dict(orient="records")
        event_data.update({s:dict})

    with open(filename, "w") as f:
        json.dump(event_data, f, indent=4)

def get_all_events():
    events = parser.list_game_events()
    list = parser.parse_events(events)

    return list

def sensitive_data_events(l:list):
    """
        Takes a list of tuples. This should represent the data in all events
    """

    altered_list = []

    for (s, df) in l:
        # remember to remove chat messages
        df = remove_sensitive_data_df(df)
        df = replace_sensitive_data_df(df)

        altered_list.append((s, df))

    return altered_list

def get_player_nameid_dict():
    player_info = parser.parse_player_info()

    # creates a dict from the steamid and names to Player_x
    steamid_to_player = {steamid: f'Player_{i+1}' for i, steamid in enumerate(player_info['steamid'].unique())}
    name_to_player = {name: steamid_to_player[steamid] for steamid, name in zip(player_info['steamid'], player_info['name'])}
    player_mapping = {**steamid_to_player, **name_to_player}
    player_mapping = {str(key): value for key, value in player_mapping.items()}

    return player_mapping

def replace_sensitive_data_df(df:pandas.core.frame.DataFrame):
    player_mapping = get_player_nameid_dict()

    df_anonymised = df.copy()

    for d in SENSITIVE_DATA_REPLACE:
        if d in df_anonymised.columns:
            df_anonymised[d] = df_anonymised[d].astype(str).map(player_mapping).fillna("")

    return df_anonymised

def remove_sensitive_data_df(df:pandas.core.frame.DataFrame):
    df_anonymised = df.copy()

    df_anonymised = df_anonymised.drop(columns=SENSITIVE_DATA_REMOVAL, errors='ignore') # 'ignore' ignores the errors raised by non-existing columns

    return df_anonymised
