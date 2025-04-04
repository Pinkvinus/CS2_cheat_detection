# This is the main file for the parsing of the data

from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas as pd
from utils.demo_parser_fields import ALL_FIELDS
import utils.dem_utils as demu
import os
import sys
import ast

INPUT_PATH = "./Demo_data/Demos"
OUTPUT_PATH = "./Data"
CHEATER_PATH = OUTPUT_PATH + "/with_cheater_present"
NO_CHEATER_PATH = OUTPUT_PATH + "/no_cheater_present"

SCRAPE_PATH = "./Demo_data/cs_scrape_2025-04-03.csv"

counter_no_cheater = 0
counter_cheater = 0

scrape_df = pd.read_csv(SCRAPE_PATH)
df = scrape_df.dropna(subset=["demo_file_name"])
df = df.drop(columns=["team1_string", "team2_string", "match_id"])
df["cheater_names_str"] = df["cheater_names_str"].apply(ast.literal_eval)

for index, row in df.iterrows():

    # Looks up cheater information
    cheaters = row["cheater_names_str"]
    contains_cheaters = len(cheaters) > 0

    cheater_filepath = CHEATER_PATH + "/" + str(counter_cheater)
    no_cheater_filepath = NO_CHEATER_PATH + "/" + str(counter_no_cheater)

    # check if this file has already been parsed
    if contains_cheaters and os.path.isfile(cheater_filepath + ".json") and os.path.isfile(cheater_filepath + ".csv.gz"):
        print(f"File '{counter_cheater}' already exists (with cheater)")
        counter_cheater = counter_cheater + 1
        continue
    elif not contains_cheaters and os.path.isfile(no_cheater_filepath + ".json") and os.path.isfile(no_cheater_filepath + ".csv.gz"):
        print(f"File '{counter_no_cheater}' already exists")
        counter_no_cheater = counter_no_cheater + 1
        continue

    # Updating parser with correct information
    demu.parser = DemoParser(INPUT_PATH + "/" + row["demo_file_name"])

    # loading the informations
    print("Parsing file")
    tick_df = demu.parser.parse_ticks(demu.ALL_FIELDS)
    events_list = demu.get_all_events()

    if contains_cheaters:

        # Create a cheater dataframe
        cheater_df = pd.DataFrame(columns=["name"])
        
        for c in cheaters:
            new_row = pd.DataFrame({"name":[c]})
            cheater_df = pd.concat([cheater_df, new_row], ignore_index=True)
        
        # add cheater dataframe to events_list
        events_list.append(("cheaters", cheater_df))

    # Add other info from csstats to events
    map_df = pd.DataFrame({"map":[row["map"]], "server":[row["server"]],"avg_rank":[row["avg_rank"]],"match_making_type":[row["type"]]})

    events_list.append(("CSstats_info", map_df))
    


    print("Handling Ticks =========")
    print("Replacing sensitive data")
    tick_df = demu.replace_sensitive_data_df(tick_df)
    print("Removing sensitive data")
    tick_df = demu.remove_sensitive_data_df(tick_df)

    print("Handling Events ========")
    events_list = demu.sensitive_data_events(events_list)

    path = ""

    if len(cheaters) != 0:
        path = CHEATER_PATH + "/" + str(counter_cheater)
        counter_cheater = counter_cheater + 1
    else:
        path = NO_CHEATER_PATH + "/" + str(counter_no_cheater)
        counter_no_cheater = counter_no_cheater + 1

    print("Writting data to csv file")
    tick_df.to_csv(path_or_buf=path + ".csv.gz", compression='gzip')

    print("Writting data to json file")
    demu.event_list_2_json(events_list, path + ".json")