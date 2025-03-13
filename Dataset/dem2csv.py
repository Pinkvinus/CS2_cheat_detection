from demoparser2 import DemoParser #https://github.com/LaihoE/demoparser
import pandas
import time

START_BALANCE = 800
MAX_HEALTH = 100.0
NUM_PLAYERS = 10

parser = DemoParser("/home/pinkvinus/Documents/itu/CS2_cheat_detection/Dataset/match730_003715375984984195154_2128110453_185.dem")
#event_df = parser.parse_event("player_death", player=["X", "Y"], other=["total_rounds_played"])
ticks_df = parser.parse_ticks(["X", "Y", "Z", "health", "score", "mvps", "is_alive", "balance", "Inventory", "FORWARD", "team_name", "team_clan_name", "custom_name", "total_rounds_played","is_warmup_period", "is_freeze_period"])

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

    index = binary_search_sorted_dataframe(ticks_df,'is_warmup_period', False)
    return df.loc[index:].reset_index(drop=True)

def get_tick(df:pandas.core.frame.DataFrame, index:int) -> pandas.core.frame.DataFrame:
    """Returns a single tick of the index from the dataframe"""
    i = binary_search_sorted_dataframe(df, 'tick', index)
    print()
    return df.loc[index:index+NUM_PLAYERS-1].reset_index(drop=True)


#df_to_readable_csv(ticks_df, "new_test.csv")
    


ticks = get_tick(ticks_df, 50)

print(ticks)


no_warm = remove_warmup_rounds(ticks_df)
df_to_readable_csv(no_warm, "no_warm_new_test.csv")
