import pandas as pd
import numpy as np

class MatchDataProcessor:
    def __init__(self, match_ticks: pd.DataFrame):
        self.match_ticks = match_ticks

    def get_all_values_for_player(self, player, key):
        return self.match_ticks[self.match_ticks["name"] == player][key].tolist()

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
    
    def get_pitch_yaw_deltas(self, key, start_tick, end_tick, context_window_size, player):
        delta_values = []
        values = self.get_tick_values(start_tick, end_tick, player, key)
        values.insert(0, values[0])

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
    
    def get_pitch_yaw_head_deltas(self, start_tick, end_tick, context_window_size, attacker, victim):
        pitch_deltas = []
        yaw_deltas = []

        attacker_data = self.get_tick_values_multiple(start_tick, end_tick, attacker, ["X", "Y", "Z", "pitch", "yaw"])
        victim_data = self.get_tick_values_multiple(start_tick, end_tick, victim, ["X", "Y", "Z"])

        for i in range(context_window_size):
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
    
    def get_is_player_flashed(self, start_tick, end_tick, player):
        vals = self.get_tick_values(start_tick, end_tick, player, "flash_duration")
        vals = self.variable_flashbang_decay(vals)
        return vals
    
    def variable_flashbang_decay(self, values):
        """
            This function is meant to format the demo file flashbang data. In a demo, only the duration of the flash
            is given and the value is the length of the flash in seconds. This function creates a linear decay from 1 to 0
            over that time period to sort of simulate the flashbang decay.
        """
        output = [0] * len(values)
        i = 0
        n = len(values)

        while i < n:
            if values[i] > 0:
                # Start of a non-zero streak
                start = i
                current_val = values[i]

                # Find how long the streak goes
                while i < n and values[i] == current_val:
                    i += 1
                end = i
                decay_length = end - start

                # Apply linear decay from 1 to 0 over decay_length
                for j in range(decay_length):
                    output[start + j] = 1 - (j / decay_length)
            else:
                i += 1  # move to next value if zero

        return output


