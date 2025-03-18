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

    
    demu.events_2_json(path  + ".json")