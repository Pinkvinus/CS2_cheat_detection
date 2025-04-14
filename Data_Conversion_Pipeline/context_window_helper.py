import pandas as pd
import numpy as np

class MatchDataProcessor:
    def __init__(self, match_ticks: pd.DataFrame):
        self.match_ticks = match_ticks

    def get_tick_values_multiple(self, start_tick, end_tick, player, columns):
        filtered = self.match_ticks[
            (self.match_ticks["name"] == player) &
            (self.match_ticks["tick"] > start_tick) &
            (self.match_ticks["tick"] <= end_tick)
        ]
        return {col: filtered[col].tolist() for col in columns}
    
    def get_tick_values(self, start_tick, end_tick, player, key):
        values = self.match_ticks[(self.match_ticks["name"] == player) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)][key].tolist()
        return values

    def get_player_kills(self, events, player_name, player_death_idx):
        player_deaths = events[player_death_idx][1]
        return player_deaths[player_deaths["attacker_name"] == player_name]
    
    def get_victim_names(self, events, player_name, player_death_idx):
        victims = events[player_death_idx][1]
        return victims[victims["attacker_name"] == player_name]

    def get_context_window_ticks(self, player_deaths, ticks_before_kill, tick_after_kill):
        start_ticks = []
        end_ticks = []
        for tick in player_deaths["tick"]:
            start_ticks.append(tick - ticks_before_kill)
            end_ticks.append(tick + tick_after_kill)
        return (start_ticks, end_ticks)
    
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
    
    def _calculate_pitch_yaw(self, dx, dy, dz):
        # Pitch: vertical angle (looking up/down)
        pitch = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
        # Yaw: horizontal angle (looking around)
        yaw = np.degrees(np.arctan2(dy, dx))
        return pitch, yaw
    
    def get_pitch_yaw_head_deltas(self, start_tick, end_tick, attacker, victim):
        pitch_deltas = []
        yaw_deltas = []

        attacker_data = self.get_tick_values_multiple(start_tick, end_tick, attacker, ["X", "Y", "Z", "pitch", "yaw"])
        victim_data = self.get_tick_values_multiple(start_tick, end_tick, victim, ["X", "Y", "Z"])

        # Make sure lengths match
        n = min(len(attacker_data["pitch"]), len(victim_data["X"]), len(attacker_data["yaw"]))
        assert n == end_tick-start_tick, "Lenghts of data do not match"

        for i in range(n):
            dx = victim_data["X"][i] - attacker_data["X"][i]
            dy = victim_data["Y"][i] - attacker_data["Y"][i]
            dz = victim_data["Z"][i] - attacker_data["Z"][i]

            expected_pitch, expected_yaw = self._calculate_pitch_yaw(dx, dy, dz)

            actual_pitch = attacker_data["pitch"][i]
            actual_yaw = attacker_data["yaw"][i]

            pitch_delta = self.pitch_delta(actual_pitch, expected_pitch)
            yaw_delta = self.yaw_delta(actual_yaw, expected_yaw)

            pitch_deltas.append(pitch_delta)
            yaw_deltas.append(yaw_delta)

        return pitch_deltas, yaw_deltas
    
    def yaw_delta(self, a1, a2):
        """Returns the shortest difference between two angles in degrees (-180 to 180)"""
        delta = (float(a2) - float(a1) + 180) % 360 - 180
        return delta
    
    def pitch_delta(self, a1, a2):
        """Returns the shortest difference between two yaw angles in degrees (-90 to 90)"""
        delta = (float(a2) - float(a1) + 90) % 180 - 90
        return delta

