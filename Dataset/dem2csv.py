from demoparser2 import DemoParser

parser = DemoParser("/home/pinkvinus/Documents/itu/CS2_cheat_detection/Dataset/match730_003716241345732411515_0544541936_187.dem")
event_df = parser.parse_event("player_death", player=["X", "Y"], other=["total_rounds_played"])
ticks_df = parser.parse_ticks(["X", "Y"])

