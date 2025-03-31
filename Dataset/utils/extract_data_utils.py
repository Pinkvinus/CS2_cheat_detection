import pandas

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

def get_tick(df:pandas.core.frame.DataFrame, index:int) -> pandas.core.frame.DataFrame:
    """Returns a single tick of the index from the dataframe"""

    i = binary_search_sorted_dataframe(df, 'tick', index)

    return df.loc[i:i+NUM_PLAYERS-1].reset_index(drop=True)

def get_ticks(df:pandas.core.frame.DataFrame, start:int, end:int) -> pandas.core.frame.DataFrame:
    """Returns a dataframe of ticks of the index from the dataframe"""

    s = binary_search_sorted_dataframe(df, 'tick', start)
    t = binary_search_sorted_dataframe(df, 'tick', end)

    return df.loc[s:t+NUM_PLAYERS-1].reset_index(drop=True)

def remove_warmup_rounds(df: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    """removes the ticks corresponding to a warmup round for a given data frame"""

    index = binary_search_sorted_dataframe(df,'is_warmup_period', False)
    return df.loc[index:].reset_index(drop=True)
