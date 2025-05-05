import pandas as pd
import json

class EventList():

    def __init__(self, data=None):
        self._data = []
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("Initial data must be a list of (str, pd.DataFrame) tuples")
            for item in data:
                self._validate_item(item)
            self._data.extend(data)
    
    def append(self, item:tuple[str, pd.DataFrame]):
        self._validate_item(item)
        self._data.append(item)
    
    def _validate_item(self, item):
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], pd.DataFrame)):
            raise TypeError("Each item must be a tuple of (str, pandas.DataFrame)")

    def read_json(self, filepath):
        """
            This function loads a .json file into an eventlist.
        """

        with open(filepath, "r") as f:
            json_data = json.load(f)

        data = []

        for key, value in json_data.items():
            if isinstance(value, list):
                df = pd.DataFrame(value)
                data.append((key, df))

        self._data = data

    def write_json(self, filename: str):
        """
            This function saves a list of events as a .json file.
        """
        
        event_data = {}

        for (s, df) in self._data:
            dict = df.to_dict(orient="records")
            event_data.update({s:dict})

        with open(filename, "w") as f:
            json.dump(event_data, f, indent=4)
            
    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._validate_item(value)
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return repr(self._data)
    
    def __contains__(self, key):
        if isinstance(key, str):
            return any(name == key for name, _ in self._data)
        return False
