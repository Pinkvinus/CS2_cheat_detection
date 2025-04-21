import pandas as pd
import json

def json_2_event_list(filepath:str) -> list[tuple[str, pd.DataFrame]]:
    """
        This function loads a .json file into a list of events.
    """

    with open(filepath, "r") as f:
        json_data = json.load(f)

    data = []       

    for key, value in json_data.items():
        if isinstance(value, list):
            df = pd.DataFrame(value)
            data.append((key, df))

    return data

def event_list_2_json(list:list, filename:str):
    """
        This function saves a list of events as a .json file.
    """
    
    event_data = {}

    for (s, df) in list:
        dict = df.to_dict(orient="records")
        event_data.update({s:dict})

    with open(filename, "w") as f:
        json.dump(event_data, f, indent=4)
