# This is the main file for the parsing of the data

import pandas as pd
from utils.demo_parser_fields import ALL_FIELDS
import os
import ast
from CSDemoConverter import CSDemoConverter
from pathlib import Path
import gc

INPUT_PATH = Path("./Demo_data/Demos")
OUTPUT_PATH = "../CS2CD.Counter-Strike_2_Cheat_Detection"
CHEATER_PATH = OUTPUT_PATH + "/with_cheater_present"
NO_CHEATER_PATH = OUTPUT_PATH + "/no_cheater_present"

SCRAPE_PATH = "./Demo_data/cs_scrape_2025-04-03.csv"
TICK_FILETYPE = ".parquet"

counter_no_cheater = 0
counter_cheater = 0

scrape_df = pd.read_csv(SCRAPE_PATH)
df = scrape_df.dropna(subset=["demo_file_name"])
df = df.drop(columns=["team1_string", "team2_string", "match_id"])
df["cheater_names_str"] = df["cheater_names_str"].apply(ast.literal_eval)

skip = 5
num_files = len(df)

def skip_str(num:int):
    for _ in range(num):
        print("\033[F\033[K", end="")

for index, row in df.iterrows():
    print(f'{counter_cheater + counter_no_cheater}/{num_files}')
    print(f'Cheater:     {counter_cheater}')
    print(f'Not cheater: {counter_no_cheater}')
    print(row['demo_file_name'])

    # Looks up cheater information
    cheaters = row["cheater_names_str"]
    contains_cheaters = len(cheaters) > 0

    cheater_filepath = CHEATER_PATH + "/" + str(counter_cheater)
    no_cheater_filepath = NO_CHEATER_PATH + "/" + str(counter_no_cheater)


    # check if this file has already been parsed
    if contains_cheaters and os.path.isfile(cheater_filepath + ".json") and os.path.isfile(cheater_filepath + TICK_FILETYPE):
        print(f"\rFile '{counter_cheater}' already exists (with cheater)")
        counter_cheater = counter_cheater + 1
        skip_str(skip)
        continue
    elif not contains_cheaters and os.path.isfile(no_cheater_filepath + ".json") and os.path.isfile(no_cheater_filepath + TICK_FILETYPE):
        print(f"\rFile '{counter_no_cheater}' already exists")
        counter_no_cheater = counter_no_cheater + 1
        skip_str(skip)
        continue

    matchfile = row['demo_file_name']
    filepath = INPUT_PATH.joinpath(matchfile)

    converter = CSDemoConverter(filepath, OUTPUT_PATH)

    if not contains_cheaters:
        cheaters = None

    tick_df, events_list = converter.convert_file(cheaters=cheaters, csstats_info=row)

    name = ''
    
    if contains_cheaters:
        name = counter_cheater
        counter_cheater = counter_cheater + 1
    else:
        name = counter_no_cheater
        counter_no_cheater = counter_no_cheater + 1
    
    converter.save(tick_df, events_list, str(name))

    skip_str(11)

    gc.collect()

    


    