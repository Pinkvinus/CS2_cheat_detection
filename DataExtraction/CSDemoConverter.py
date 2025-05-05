from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas as pd
from utils.demo_parser_fields import ALL_FIELDS
from utils.sensitive_data_fields import SENSITIVE_DATA_REMOVAL, SENSITIVE_DATA_REPLACE
from utils.EventList import EventList
from pathlib import Path

class CSDemoConverter():

    def __init__(self, filepath, output_path):
        self.filepath =filepath
        self.OUTPUT_PATH = Path(output_path)#"./Data"
        self.CHEATER_PATH = self.OUTPUT_PATH.joinpath("with_cheater_present")
        self.NO_CHEATER_PATH = self.OUTPUT_PATH.joinpath("no_cheater_present")

        self.NO_CHEATER_PATH.mkdir(parents=True, exist_ok=True)
        self.CHEATER_PATH.mkdir(parents=True, exist_ok=True)

        self.TICK_FILETYPE = ".parquet"
        self.EVENT_FILETYPE = ".json"

        self.parser = DemoParser(str(filepath))
        self.player_mapping = self.get_player_mapping()
    
    def convert_file(self, anon:bool=True, cheaters:list=None, csstats_info:pd.Series=None):
        """
            Converts a single demofile
        """

        tick_df, events_list = self.parse_file()

        if cheaters is not None:
            self.add_cheaters(events_list, cheaters)

        if csstats_info is not None:
            self.add_csstats_info(events_list, csstats_info)

        if anon == True:
            tick_df, events_list = self.anonymise(tick_df, events_list)

        return tick_df, events_list
    
    def parse_file(self):

        # loading the informations
        print("Parsing file")
        tick_df = self.parser.parse_ticks(ALL_FIELDS)
        events_list = EventList(self.get_all_events())

        return tick_df, events_list

    def add_cheaters(self, events_list:EventList, cheaters:list):
        # Create a cheater dataframe
        cheater_df = pd.DataFrame(columns=["steamid"])

        player_info = self.parser.parse_player_info()
        
        for _, row in player_info.iterrows():
            player_name = row["name"]
            player_steamid = row["steamid"]

            # Check if the name is in the cheaters list
            for name in cheaters:
                if name in player_name:
                    #print(f"Cheater found: {row['name']} with steamid {row['steamid']}")
                    # Add the cheater to the cheater dataframe
                    new_row = pd.DataFrame({"steamid": [player_steamid]})
                    cheater_df = pd.concat([cheater_df, new_row], ignore_index=True)
                    break
                
        # add cheater dataframe to events_list
        events_list.append(("cheaters", cheater_df))
    
    def add_csstats_info(self, events_list:EventList, csstats_info:pd.Series):
        # Add other info from csstats to events
        map_df = pd.DataFrame({"map":[csstats_info["map"]], 
                                "server":[csstats_info["server"]],
                                "avg_rank":[csstats_info["avg_rank"]],
                                "match_making_type":[csstats_info["type"]]})

        events_list.append(("CSstats_info", map_df))

    def anonymise(self, tick_df:pd.DataFrame, events_list:EventList):
        print("Handling Ticks =========")
        print("Removing sensitive data")
        tick_df = self._remove_sensitive_data_df(tick_df)
        print("Replacing sensitive data")
        tick_df = self._replace_sensitive_data_df(tick_df)

        print("Handling Events ========")
        events_list = EventList(self._sensitive_data_events(events_list))

        return tick_df, events_list
    
    def save(self, tick_df:pd.DataFrame, events_list:EventList, filename:Path):

        path = self.NO_CHEATER_PATH
        if "cheaters" in events_list:
            path = self.CHEATER_PATH

        print("Writting tick data to file")
        tick_df.to_parquet(path=path.joinpath(filename + self.TICK_FILETYPE), index=False)

        print("Writting event data to file")
        events_list.write_json(path.joinpath(filename + self.EVENT_FILETYPE))

    def get_all_events(self):
        events = self.parser.list_game_events()
        list = self.parser.parse_events(events)

        return list

    def get_player_mapping(self):

        player_mapping = {}

        player_info = self.parser.parse_player_info()

        # creates a dict from the steamid and names to Player_x
        steamid_to_player = {steamid: f'Player_{i+1}' for i, steamid in enumerate(player_info['steamid'].unique())}
        name_to_player = {name: steamid_to_player[steamid] for steamid, name in zip(player_info['steamid'], player_info['name'])}
        pm = {**steamid_to_player, **name_to_player}
        player_mapping.update({str(key): value for key, value in pm.items()})

        return player_mapping

    def _sensitive_data_events(self, l:EventList):
        """
            Takes a list of tuples. This should represent the data in all events
        """

        altered_list = []

        for (s, df) in l:
            if s == "chat_message": # Removes chat messages send in game from the data.
                continue

            if s == "server_cvar": # An outlier that has a "redactible" column with no sensitive information
                altered_list.append((s, df))
                continue

            df = self._remove_sensitive_data_df(df)
            df = self._replace_sensitive_data_df(df)

            altered_list.append((s, df))

        return altered_list

    def _replace_sensitive_data_df(self, df:pd.DataFrame):
        df_anonymised = df.copy()

        for d in SENSITIVE_DATA_REPLACE:

            if d in df_anonymised.columns:
                # Check if the column contains any lists
                if df_anonymised[d].apply(lambda x: isinstance(x, list)).any():
                    df_anonymised[d] = df_anonymised[d].apply(lambda lst: [self.player_mapping.get(str(id)) for id in lst])
                    continue

                df_anonymised[d] = df_anonymised[d].astype(str).map(self.player_mapping).fillna("")

        return df_anonymised

    def _remove_sensitive_data_df(self, df:pd.DataFrame):
        df_anonymised = df.copy()
        df_anonymised = df_anonymised.drop(columns=SENSITIVE_DATA_REMOVAL, errors='ignore') # 'ignore' ignores the errors raised by non-existing columns

        return df_anonymised
