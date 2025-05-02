import pandas as pd
import numpy as np
from IPython.display import display
import math

class MatchDataProcessor:
    def __init__(self, match_ticks: pd.DataFrame, match_events: pd.DataFrame, context_window_size):
        self.match_ticks = match_ticks
        self.match_events = match_events
        self.context_window_size = context_window_size

    def normalize(self, value, min_val, max_val):
        normalized = (value - min_val) / (max_val - min_val)
        return normalized

    def get_all_values_for_player(self, player, key):
        return self.match_ticks[self.match_ticks["steamid"] == player][key].tolist()

    def get_tick_values_multiple(self, start_tick, end_tick, player, columns):
        filtered = self.match_ticks[
            (self.match_ticks["steamid"] == player) &
            (self.match_ticks["tick"] > start_tick) &
            (self.match_ticks["tick"] <= end_tick)
        ]
        return {col: filtered[col].tolist() for col in columns}
    
    def get_tick_values(self, start_tick, end_tick, player, key):
        values = self.match_ticks[(self.match_ticks["steamid"] == player) & 
                                 (self.match_ticks["tick"] <= end_tick) & 
                                 (self.match_ticks["tick"] > start_tick)][key].tolist()
        return values

    def get_player_kills(self, player_name, player_death_idx):
        player_deaths = self.match_events[player_death_idx][1]
        return player_deaths[player_deaths["attacker_steamid"] == player_name]
    
    def get_player_team(self, start_tick, end_tick, player):
        vals = self.get_tick_values(start_tick, end_tick, player, "team_name")
        for val in vals:
            if val is not None:
                return val

    def get_context_window_ticks(self, ticks_before_kill, tick_after_kill, player, player_death_idx):
        player_deaths = self.get_player_kills(player, player_death_idx)
        start_ticks = []
        end_ticks = []
        all_ticks = self.get_all_values_for_player(player, "tick")
        for tick in player_deaths["tick"]:
            idx = all_ticks.index(tick)
            start_ticks.append(all_ticks[idx - ticks_before_kill])
            end_ticks.append(all_ticks[idx + tick_after_kill])
        return (start_ticks, end_ticks)

    # (X_min, X_max, Y_min, Y_max, Z_min, Z_max)
    map_min_max = {
        "de_dust2": (-2300, 1800, -1200, 3150, -130, 400),
        "de_mirage": (-2700, 1500, -2700, 1000, -400, 150),
        "de_inferno": (-1800, 2700, -800, 3600, -100, 500),
        "de_train": (-2200, 1800, -1800, 1800, -400, 200),
        "de_nuke": (-3000, 3500, -2500, 1000, -750, 100),
        "de_ancient": (-2310, 1400, -2600, 1800, -120, 400),
        "de_vertigo": (-2700, 100, -1600, 1200, 11400, 12100),
        "de_anubis": (-2000, 1810, -1810, 3200, -150, 200),
        "cs_office": (-1800, 2400, -2200, 1300, -350, 10),
        "de_overpass": (-4000, 20, -3500, 1700, 0, 800),
        "de_basalt": (-2100, 2000, -1700, 2350, -100, 400),
        "de_edin": (500, 3700, -350, 4300, 300, 750),
        "cs_italy": (-1550, 1100, -2200, 2650, -200, 300),
        "de_thera": (600, 4300, -2600, 2200, -170, 300),
        "de_mills": (-4300, 0, -5560, -300, -100, 300)
    }
    
    def get_player_coordinates(self, start_tick, end_tick, player, map_name):
        coords = self.get_tick_values_multiple(start_tick, end_tick, player, ["X", "Y", "Z"])
        
        if map_name not in self.map_min_max:
            raise ValueError(f"Unknown map: {map_name}")

        min_x, max_x, min_y, max_y, min_z, max_z = self.map_min_max[map_name]

        X_normalized = [self.normalize(x, min_x, max_x) for x in coords["X"]]
        Y_normalized = [self.normalize(y, min_y, max_y) for y in coords["Y"]]
        Z_normalized = [self.normalize(z, min_z, max_z) for z in coords["Z"]]
        X_normalized = np.clip(X_normalized, 0, 1)
        Y_normalized = np.clip(Y_normalized, 0, 1)
        Z_normalized = np.clip(Z_normalized, 0, 1)

        return (X_normalized, Y_normalized, Z_normalized)

    def get_player_velocity(self, start_tick, end_tick, player):
        vels = self.get_tick_values(start_tick, end_tick, player, "velocity")
        vels = [250 if x > 500 else x for x in vels]
        vel_normalized = [self.normalize(x, 0, 250) for x in vels]
        return vel_normalized
    
    def normalize_pitch(self, p):
        return [self.normalize(z, -90, 90) for z in p]
    
    def normalize_yaw(self, y):
        return [self.normalize(z, -180, 180) for z in y]
    
    def get_player_pitch(self, start_tick, end_tick, player):
        p = self.get_tick_values(start_tick, end_tick, player, "pitch")
        p_norm = self.normalize_pitch(p)
        return p_norm
    
    def get_player_yaw(self, start_tick, end_tick, player):
        y = self.get_tick_values(start_tick, end_tick, player, "yaw")
        y_norm = self.normalize_yaw(y)
        return y_norm

    def get_pitch_yaw_deltas(self, key, start_tick, end_tick, player):
        values = self.get_tick_values(start_tick, end_tick, player, key)
        values.insert(0, values[0])

        values = np.array(values)
        deltas = np.diff(values)

        if key == "pitch":
            deltas = np.clip(np.abs(deltas) / 45, 0, 1)
        elif key == "yaw":
            deltas = np.clip(np.abs(deltas) / 90, 0, 1)

        return deltas.tolist()
    
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

        pitch_deltas_normalized = [np.clip(abs(val), 0, 45) / 45 for val in pitch_deltas]
        yaw_deltas_normalized = [np.clip(abs(val), 0, 90) / 90 for val in yaw_deltas]

        return pitch_deltas_normalized, yaw_deltas_normalized
    
    def yaw_delta(self, a1, a2):
        """Returns the shortest difference between two angles in degrees (-180 to 180)"""
        delta = (float(a2) - float(a1) + 180) % 360 - 180
        return delta
    
    def pitch_delta(self, a1, a2):
        """Returns the shortest difference between two angles in degrees (-90 to 90)"""
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
                # start of a non-zero streak
                start = i
                current_val = values[i]

                # find how long the streak goes
                while i < n and values[i] == current_val:
                    i += 1
                end = i
                decay_length = end - start

                # apply linear decay from 1 to 0 over decay_length
                for j in range(decay_length):
                    output[start + j] = 1 - (j / decay_length)
            else:
                i += 1  # move to next value if zero

        return output
    
    def get_attacker_shots(self, start_tick, end_tick, player, weapon_fire_idx):
        shots = self.match_events[weapon_fire_idx][1]
        player_shots = shots[(shots["user_steamid"] == player) & 
                             (shots["tick"] >= start_tick) & 
                             (shots["tick"] < end_tick)]["tick"].tolist()
        all_ticks = self.get_tick_values(start_tick, end_tick, player, "tick")
        data = np.zeros(self.context_window_size)
        
        for idx, tick in enumerate(all_ticks):
            if tick in player_shots:
                data[idx] = 1
        return data
    
    def get_attacker_kill_data(self, start_tick, end_tick, player, player_death_idx):
        all_ticks = self.get_tick_values(start_tick, end_tick, player, "tick")
        kills = self.get_player_kills(player, player_death_idx)
        kills = kills[(kills["tick"] >= start_tick) & (kills["tick"] < end_tick)][["tick", "thrusmoke", "penetrated", "attackerinair", "headshot"]]
        kills_tick = kills["tick"].tolist()
        data_kill = np.zeros(self.context_window_size)
        data_headshot = np.zeros(self.context_window_size)
        data_thrusmoke = np.zeros(self.context_window_size)
        data_wallbang = np.zeros(self.context_window_size)
        data_inair = np.zeros(self.context_window_size)
        for i, tick in enumerate(all_ticks):
            if tick in kills_tick:
                data_kill[i] = 1
                if kills[kills["tick"] == tick]["headshot"].tolist()[0]:
                    data_headshot[i] = 1
                if kills[kills["tick"] == tick]["thrusmoke"].tolist()[0]:
                    data_thrusmoke[i] = 1
                if kills[kills["tick"] == tick]["penetrated"].tolist()[0] > 0:
                    data_wallbang[i] = 1
                if kills[kills["tick"] == tick]["attackerinair"].tolist()[0]:
                    data_inair[i] = 1
        return data_kill, data_headshot, data_thrusmoke, data_wallbang, data_inair
        
    weapon_knife = {"Zeus x27", "knife_t", "knife", "Bayonet", "Bowie Knife", "Butterfly Knife", "Classic Knife", "Falchion Knife", "Flip Knife", "Gut Knife", "Huntsman Knife", "Karambit", "Kukri Knife", "M9 Bayonet", "Navaja Knife", "Nomad Knife", "Paracord Knife", "Shadow Daggers", "Skeleton Knife", "Stiletto Knife", "Survival Knife", "Talon Knife", "Ursus Knife"}
    
    weapon_auto_rifle = {"AK-47", "M4A1-S", "Galil AR", "SG 553", "M4A4", "AUG", "FAMAS", "M249", "Negev"}
    
    weapon_semi_rifle = {"G3SG1", "SSG 08", "AWP", "SCAR-20"}

    weapon_dead = {None}

    weapon_pistols = {"CZ75-Auto", "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18", "P2000", "P250", "R8 Revolver", "Tec-9", "USP-S"}

    weapon_grenade = {"C4 Explosive", "Decoy Grenade", "Flashbang", "High Explosive Grenade", "Incendiary Grenade", "Molotov", "Smoke Grenade"}

    weapon_smg = {"MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45"}

    weapon_shotgun = {"MAG-7", "Nova", "Sawed-Off", "XM1014"}

    def get_attacker_weapon(self, start_tick, end_tick, player):
        used_weapons = self.get_tick_values(start_tick, end_tick, player, "active_weapon_name")
        
        weapon_knife_data = []
        weapon_auto_rifle_data = []
        weapon_semi_rifle_data = []
        weapon_pistols_data = []
        weapon_grenade_data = []
        weapon_smg_data = []
        weapon_shotgun_data = []

        for weapon in used_weapons:

            if weapon is None:
                weapon_knife_data.append(0)
                weapon_auto_rifle_data.append(0)
                weapon_semi_rifle_data.append(0)
                weapon_pistols_data.append(0)
                weapon_grenade_data.append(0)
                weapon_smg_data.append(0)
                weapon_shotgun_data.append(0)
                continue

            if weapon in self.weapon_knife:
                weapon_knife_data.append(1)
                weapon_auto_rifle_data.append(0)
                weapon_semi_rifle_data.append(0)
                weapon_pistols_data.append(0)
                weapon_grenade_data.append(0)
                weapon_smg_data.append(0)
                weapon_shotgun_data.append(0)
                continue
         
            elif weapon in self.weapon_auto_rifle:
                weapon_knife_data.append(0)
                weapon_auto_rifle_data.append(1)
                weapon_semi_rifle_data.append(0)
                weapon_pistols_data.append(0)
                weapon_grenade_data.append(0)
                weapon_smg_data.append(0)
                weapon_shotgun_data.append(0)
                continue
    
            elif weapon in self.weapon_semi_rifle:
                weapon_knife_data.append(0)
                weapon_auto_rifle_data.append(0)
                weapon_semi_rifle_data.append(1)
                weapon_pistols_data.append(0)
                weapon_grenade_data.append(0)
                weapon_smg_data.append(0)
                weapon_shotgun_data.append(0)
                continue
    
            elif weapon in self.weapon_pistols:
                weapon_knife_data.append(0)
                weapon_auto_rifle_data.append(0)
                weapon_semi_rifle_data.append(0)
                weapon_pistols_data.append(1)
                weapon_grenade_data.append(0)
                weapon_smg_data.append(0)
                weapon_shotgun_data.append(0)
                continue
    
            elif weapon in self.weapon_grenade:
                weapon_knife_data.append(0)
                weapon_auto_rifle_data.append(0)
                weapon_semi_rifle_data.append(0)
                weapon_pistols_data.append(0)
                weapon_grenade_data.append(1)
                weapon_smg_data.append(0)
                weapon_shotgun_data.append(0)
                continue
    
            elif weapon in self.weapon_smg:
                weapon_knife_data.append(0)
                weapon_auto_rifle_data.append(0)
                weapon_semi_rifle_data.append(0)
                weapon_pistols_data.append(0)
                weapon_grenade_data.append(0)
                weapon_smg_data.append(1)
                weapon_shotgun_data.append(0)
                continue
    
            elif weapon in self.weapon_shotgun:
                weapon_knife_data.append(0)
                weapon_auto_rifle_data.append(0)
                weapon_semi_rifle_data.append(0)
                weapon_pistols_data.append(0)
                weapon_grenade_data.append(0)
                weapon_smg_data.append(0)
                weapon_shotgun_data.append(1)
                continue
        
        return (weapon_knife_data, 
                weapon_auto_rifle_data, 
                weapon_semi_rifle_data,
                weapon_pistols_data,
                weapon_grenade_data,
                weapon_smg_data,
                weapon_shotgun_data)
    
    def get_player_health(self, start_tick, end_tick, player):
        health = self.get_tick_values(start_tick, end_tick, player, "health")
        health_norm = [val / 100 for val in health]
        return health_norm
    
    def get_player_made_noise(self, start_tick, end_tick, player, weapon_fire_idx):
        velocity = self.get_tick_values(start_tick, end_tick, player, "velocity")
        shots = self.get_attacker_shots(start_tick, end_tick, player, weapon_fire_idx)
        footsteps = []
        for v in velocity:
            if v >= 140:
                footsteps.append(1)
            else:
                footsteps.append(0)
        data = []
        for idx, v in enumerate(shots):
            val = 0
            if shots[idx] == 1:
                val += 0.5
            if footsteps[idx] == 1:
                val += 0.5
            data.append(val)
        return data
    
    # Not used, unsure how to normalize data
    # def get_players_distance(self, start_tick, end_tick, player1, player2):
    #     p1 = self.get_tick_values_multiple(start_tick, end_tick, player1, ["X", "Y", "Z"])
    #     p2 = self.get_tick_values_multiple(start_tick, end_tick, player2, ["X", "Y", "Z"])

    #     distances = []

    #     for x1, y1, z1, x2, y2, z2 in zip(p1["X"], p1["Y"], p1["Z"], p2["X"], p2["Y"], p2["Z"]):
    #         dx = x1 - x2
    #         dy = y1 - y2
    #         dz = z1 - z2
    #         distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    #         distances.append(distance)

    #     return distances
    
    def get_played_map(self, played_map):
        zeros = np.zeros(1024)
        ones = np.ones(1024)
    
        maps = [
            "de_dust2", "de_mirage", "de_inferno", "de_train", "de_nuke",
            "de_ancient", "de_vertigo", "de_anubis", "cs_office", "de_overpass",
            "de_basalt", "de_edin", "cs_italy", "de_thera", "de_mills"
        ]
    
        outputs = []
        for map_name in maps:
            if played_map == map_name:
                outputs.append(ones)
            else:
                outputs.append(zeros)
    
        return tuple(outputs)
