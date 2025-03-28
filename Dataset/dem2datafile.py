from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas
import json
from demo_parser_fields import ALL_FIELDS
import dem_utils as demu
import os

INPUT_PATH = "./Demos"
OUTPUT_PATH = "./Data"

counter = 0


for demo in os.listdir(INPUT_PATH):

    demu.parser = DemoParser(INPUT_PATH + "/" + demo)
    path = OUTPUT_PATH + "/" + str(counter)

    # loading the information
    print("Parsing file")
    tick_df = demu.parser.parse_ticks(demu.ALL_FIELDS)
    events_list = demu.get_all_events()

    print("Handling Ticks =========")
    print("Replacing sensitive data")
    tick_df = demu.replace_sensitive_data_df(tick_df)
    print("Removing sensitive data")
    tick_df = demu.remove_sensitive_data_df(tick_df)

    print("Handling Events ========")
    events_list = demu.sensitive_data_events(events_list)

    print("Writting data to csv file")
    tick_df.to_csv(path_or_buf=path + ".csv", compression='gzip')

    print("Writting data to json file")
    demu.event_list_2_json(events_list, path + ".json")