from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas as pd
from utils.demo_parser_fields import ALL_FIELDS
import utils.dem_utils as demu
import os
import ast

class CSDemoConverter():

    def __init__(self, input_path, output_path, scrape_path):
        self.INPUT_PATH = input_path #"./Demo_data/Demos"
        self.OUTPUT_PATH = output_path#"./Data"
        self.CHEATER_PATH = self.OUTPUT_PATH + "/with_cheater_present"
        self.NO_CHEATER_PATH = self.OUTPUT_PATH + "/no_cheater_present"

        self.SCRAPE_PATH = scrape_path#"./Demo_data/cs_scrape_2025-04-03.csv"
        self.TICK_FILETYPE = ".parquet"
        self.EVENT_FILETYPE = ".json"

        self.counter_no_cheater = 0
        self.counter_cheater = 0
    
    def convert_file(self, in_filepath, out_filepath, cheaters:list=None, csstats_info:pd.Series=None):
        """
            Converts a single demofile
        """

        # Updating parser with correct information
        demu.parser = DemoParser(in_filepath)

        # loading the informations
        print("Parsing file")
        tick_df = demu.parser.parse_ticks(demu.ALL_FIELDS)
        events_list = demu.get_all_events()
        demu.update_player_mapping()

        if cheaters is not None:

            # Create a cheater dataframe
            cheater_df = pd.DataFrame(columns=["name"])
            
            for c in cheaters:
                new_row = pd.DataFrame({"name":[c]})
                cheater_df = pd.concat([cheater_df, new_row], ignore_index=True)
            
            # add cheater dataframe to events_list
            events_list.append(("cheaters", cheater_df))

        if csstats_info is not None:
            # Add other info from csstats to events
            map_df = pd.DataFrame({"map":[csstats_info["map"]], 
                                   "server":[csstats_info["server"]],
                                   "avg_rank":[csstats_info["avg_rank"]],
                                   "match_making_type":[csstats_info["type"]]})

            events_list.append(("CSstats_info", map_df))
        
        print("Handling Ticks =========")
        print("Removing sensitive data")
        tick_df = demu.remove_sensitive_data_df(tick_df)
        print("Replacing sensitive data")
        tick_df = demu.replace_sensitive_data_df(tick_df)

        print("Handling Events ========")
        events_list = demu.sensitive_data_events(events_list)


        print("Writting data to csv file")
        tick_df.to_parquet(path=out_filepath + self.TICK_FILETYPE, index=False)
        #tick_df.to_csv(path_or_buf=path + ".csv")

        print("Writting data to json file")
        demu.event_list_2_json(events_list, out_filepath + self.EVENT_FILETYPE)





conv = CSDemoConverter('','','',)

conv.convert_file('./test/postest1.dem','./test_out/postest1')
