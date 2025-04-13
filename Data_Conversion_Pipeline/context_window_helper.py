import pandas as pd

class MatchDataProcessor:
    def __init__(self, match_ticks: pd.DataFrame):
        self.match_ticks = match_ticks

    def get_player_kills(self, events, player_name, player_death_idx):
        player_deaths = events[player_death_idx][1]
        return player_deaths[player_deaths["attacker_name"] == player_name]
    
    def get_victim_names(self, events, player_name, player_death_idx):
        victims = events[player_death_idx][1]
        return player_deaths[player_deaths["attacker_name"] == player_name]

    def get_context_window_ticks(self, player_deaths, ticks_before_kill, tick_after_kill):
        start_ticks = []
        end_ticks = []
        for tick in player_deaths["tick"]:
            start_ticks.append(tick - ticks_before_kill)
            end_ticks.append(tick + tick_after_kill)
        return (start_ticks, end_ticks)

    def get_value_from_match_data(self, key, start_tick, end_tick, player):
        values = self.match_ticks[(self.match_ticks["name"] == player) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)][key].tolist()
        return values
    
    def get_pitch_yaw_deltas(self, key, start_tick, end_tick, player):
        delta_values = []
        values = self.match_ticks[(self.match_ticks["name"] == player) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)][key].tolist()
        values.insert(0, values[0])
        context_window_size = end_tick - start_tick
        if key == "pitch":
            for i in range(1, context_window_size + 1):
                delta_values.append(self.pitch_delta(values[i-1], values[i]))
        elif key == "yaw":
            for i in range(1, context_window_size + 1):
                delta_values.append(self.yaw_delta(values[i-1], values[i]))

        return delta_values
    
    def get_pitch_yaw_head_deltas(self, key, start_tick, end_tick, attacker, victim):
        delta_values = []
        context_window_size = end_tick - start_tick
        values = self.match_ticks[(self.match_ticks["name"] == attacker) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)][key].tolist()
        attacker_X_values = self.match_ticks[(self.match_ticks["name"] == attacker) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)]["X"].tolist()
        attacker_Y_values = self.match_ticks[(self.match_ticks["name"] == attacker) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)]["Y"].tolist()
        attacker_Z_values = self.match_ticks[(self.match_ticks["name"] == attacker) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)]["Z"].tolist()
        victim_X_values = self.match_ticks[(self.match_ticks["name"] == victim) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)]["X"].tolist()
        victom_Y_values = self.match_ticks[(self.match_ticks["name"] == victim) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)]["Y"].tolist()
        victom_Z_values = self.match_ticks[(self.match_ticks["name"] == victim) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)]["Z"].tolist()
        if key == "pitch":
            ...
        elif key == "yaw":
            ...
    
    def pitch_delta(self, a1, a2):
        """Returns the shortest difference between two angles in degrees (-180 to 180)"""
        delta = (float(a2) - float(a1) + 180) % 360 - 180
        return delta
    
    def yaw_delta(self, a1, a2):
        """Returns the shortest difference between two yaw angles in degrees (-90 to 90)"""
        delta = (float(a2) - float(a1) + 90) % 180 - 90
        return delta

