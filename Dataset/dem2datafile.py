# This is the main file for the parsing of the data

from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas as pd
from utils.demo_parser_fields import ALL_FIELDS
import utils.dem_utils as demu
import os
import sys
import ast

INPUT_PATH = "./Demos"
OUTPUT_PATH = "./Data"
CHEATER_PATH = OUTPUT_PATH + "/with_cheater_present"
NO_CHEATER_PATH = OUTPUT_PATH + "/no_cheater_present"

SCRAPE_PATH = "cs_scrape_test.csv" # TODO: CHANGE BACK

counter = 0

scrape_df = pd.read_csv(SCRAPE_PATH)
df = scrape_df.dropna(subset=["demo_file_name"])
df = df.drop(columns=["team1_string", "team2_string", "match_id"])
df["cheater_names_str"] = df["cheater_names_str"].apply(ast.literal_eval)

for index, row in df.iterrows():

    demu.parser = DemoParser(INPUT_PATH + "/" + row["demo_file_name"])

    # loading the informations
    print("Parsing file")
    tick_df = demu.parser.parse_ticks(demu.ALL_FIELDS)
    events_list = demu.get_all_events()

    # Add the cheater information to events_list
    cheaters = row["cheater_names_str"]
    
    if len(cheaters) != 0:
        # Create a cheater dataframe
        cheater_df = pd.DataFrame(columns=["cheater"])
        
        for c in cheaters:
            new_row = pd.DataFrame({"cheater":[c]})
            cheater_df = pd.concat([cheater_df, new_row], ignore_index=True)
        
        # add cheater dataframe to events_list
        events_list.append(("cheaters", cheater_df))

    map_df = pd.DataFrame({"map":[row["map"]], "server":[row["server"]],"avg_rank":[row["avg_rank"]],"match_making_type":[row["type"]]})

    events_list.append(("CSstats_info", map_df))
    


    print("Handling Ticks =========")
    print("Replacing sensitive data")
    tick_df = demu.replace_sensitive_data_df(tick_df)
    print("Removing sensitive data")
    tick_df = demu.remove_sensitive_data_df(tick_df)

    print("Handling Events ========")
    events_list = demu.sensitive_data_events(events_list)


    path = OUTPUT_PATH + "/" + str(counter)
    print("Writting data to csv file")
    tick_df.to_csv(path_or_buf=path + ".csv.gz", compression='gzip')

    print("Writting data to json file")
    demu.event_list_2_json(events_list, path + ".json")

    counter = counter + 1