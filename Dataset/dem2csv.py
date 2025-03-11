from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas

parser = DemoParser("/home/pinkvinus/Documents/itu/CS2_cheat_detection/Dataset/match730_003715375984984195154_2128110453_185.dem")
event_df = parser.parse_event("player_death", player=["X", "Y"], other=["total_rounds_played"])
ticks_df = parser.parse_ticks(["X", "Y", "Z", "health", "score", "mvps", "is_alive", "balance", "Inventory", "FORWARD", "team_name", "team_clan_name", "custom_name"])



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

df_to_readable_csv(ticks_df, "fuck dig selv.csv")
