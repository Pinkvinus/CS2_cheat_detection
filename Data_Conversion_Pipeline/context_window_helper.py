import pandas as pd

class MatchDataProcessor:
    def __init__(self, match_ticks: pd.DataFrame):
        self.match_ticks = match_ticks

    def get_player_kills(events, player_name, player_death_idx):
        player_deaths = events[player_death_idx][1]
        return player_deaths[player_deaths["attacker_name"] == player_name]

    def get_context_window_ticks(player_deaths, ticks_before_kill, tick_after_kill):
        start_ticks = []
        end_ticks = []
        for tick in player_deaths["tick"]:
            start_ticks.append(tick - ticks_before_kill)
            end_ticks.append(tick + tick_after_kill)
        return (start_ticks, end_ticks)
    
    def get_attacker_XYZ(start_tick, end_tick, match_ticks, player):
        X = match_ticks[(match_ticks["name"] == player) & (match_ticks["tick"] <= end_tick) & (match_ticks["tick"] > start_tick)]["X"].tolist()
        Y = match_ticks[(match_ticks["name"] == player) & (match_ticks["tick"] <= end_tick) & (match_ticks["tick"] > start_tick)]["Y"].tolist()
        Z = match_ticks[(match_ticks["name"] == player) & (match_ticks["tick"] <= end_tick) & (match_ticks["tick"] > start_tick)]["Z"].tolist()
        return (X, Y, Z)
    
    def get_value_from_match_data(value, start_tick, end_tick, match_ticks, player):
        value = match_ticks[(match_ticks["name"] == player) & (match_ticks["tick"] <= end_tick) & (match_ticks["tick"] > start_tick)][value].tolist()
