import pandas as pd
import json
import os
import context_window_helper as cwh
import matplotlib.pyplot as plt
import numpy as np

is_cheater_data = "cdata"
filepath = r"C:\Users\Gert\Desktop\parsed_data\with_cheater_present"
cheater_out_dir = r"C:\Users\Gert\Desktop\context_windows\cheater"
non_cheater_out_dir = r"C:\Users\Gert\Desktop\context_windows\not_cheater"
files_count = int(len(os.listdir(filepath)) / 2)
start_file_idx = 0
print(files_count)

def json_2_eventlist(filepath:str) -> list:
    with open(filepath, "r") as f:
        json_data = json.load(f)

    data = []

    for key, value in json_data.items():
        if isinstance(value, list):
            df = pd.DataFrame(value)
            data.append((key, df))

    return data

ticks_before_kill = 896
ticks_after_kill = 128
context_window_size = ticks_before_kill + ticks_after_kill

context_window_vals = ["attacker_X", "attacker_Y", "attacker_Z", "attacker_vel", "attacker_pitch", "attacker_yaw", "attacker_pitch_delta", 
                       "attacker_yaw_delta", "attacker_pitch_head_delta", "attacker_yaw_head_delta", "attacker_flashed", "attacker_shot", "attacker_kill", "is_kill_headshot", "is_kill_through_smoke", 
                       "is_kill_wallbang", "attacker_midair", "attacker_weapon_knife", "attacker_weapon_auto_rifle", "attacker_weapon_semi_rifle", "attacker_weapon_pistol",
                       "attacker_weapon_grenade", "attacker_weapon_smg", "attacker_weapon_shotgun",
                       "victim_X", "victim_Y", "victim_Z", "victim_health", "victim_noise", "map_dust2", "map_mirage", "map_inferno", "map_train",
                       "map_nuke", "map_ancient", "map_vertigo", "map_anubis", "map_office", "map_overpass", "map_basalt", "map_edin", "map_italy", "map_thera", "map_mills"]


for file_idx in range(start_file_idx, files_count):
    match_ticks = pd.read_parquet(path=f"{filepath}\{file_idx}.parquet")
    match_events = json_2_eventlist(filepath=f"{filepath}\{file_idx}.json")

    MDP = cwh.MatchDataProcessor(match_ticks, match_events, context_window_size)

    player_death_idx = -1
    weapon_fire_idx = -1
    played_map = match_events[-1][1]["map"].tolist()[0]

    for idx, event in enumerate(match_events):
        if event[0] == "player_death":
            player_death_idx = idx
        if event[0] == "weapon_fire":
            weapon_fire_idx = idx
    if player_death_idx == -1 or weapon_fire_idx == -1:
        raise Exception("not all events were found")

    all_players = match_ticks["steamid"].unique().tolist()
    for p in all_players:
        context_window = pd.DataFrame(columns=context_window_vals)
        attacker = p
        is_attacker_cheater = attacker in match_events[-2][1]["steamid"].tolist()
        player_deaths = MDP.get_player_kills(attacker, player_death_idx)
        start_ticks, end_ticks = MDP.get_context_window_ticks(ticks_before_kill, ticks_after_kill, attacker, player_death_idx)

        for i in range(len(start_ticks)):
            attacker_team = MDP.get_player_team(start_ticks[i], end_ticks[i], attacker)
            victim = player_deaths.iloc[i]["user_steamid"]
            victim_team = MDP.get_player_team(start_ticks[i], end_ticks[i], victim)

            # Skip grenade kills
            weapon_used = player_deaths.iloc[i]["weapon"]
            if weapon_used in MDP.weapon_grenade:
                continue

            # Skip teamkills
            if attacker_team == victim_team:
                continue

            # Check length of the context window and no ticks missing
            ticks = MDP.get_tick_values(start_ticks[i], end_ticks[i], attacker, "tick")
            if len(ticks) != context_window_size:
                gap = context_window_size - len(ticks)
                all_ticks = MDP.get_all_values_for_player(attacker, "tick")

                idx = all_ticks.index(start_ticks[i])
                end_ticks[i] = all_ticks[idx + 1024]

            (context_window["attacker_X"],
             context_window["attacker_Y"],
             context_window["attacker_Z"]) = MDP.get_player_coordinates(start_ticks[i], end_ticks[i], attacker, played_map)

            context_window["attacker_vel"] = MDP.get_player_velocity(start_ticks[i], end_ticks[i], attacker)
            context_window["attacker_pitch"] = MDP.get_player_pitch(start_ticks[i], end_ticks[i], attacker)
            context_window["attacker_yaw"] = MDP.get_player_yaw(start_ticks[i], end_ticks[i], attacker)
            context_window["attacker_pitch_delta"] = MDP.get_pitch_yaw_deltas("pitch", start_ticks[i], end_ticks[i], attacker)
            context_window["attacker_yaw_delta"] = MDP.get_pitch_yaw_deltas("yaw", start_ticks[i], end_ticks[i], attacker)

            context_window["attacker_pitch_head_delta"], context_window["attacker_yaw_head_delta"] = MDP.get_pitch_yaw_head_deltas(start_ticks[i], end_ticks[i], context_window_size, attacker, player_deaths.iloc[i]["user_steamid"])

            context_window["attacker_flashed"] = MDP.get_is_player_flashed(start_ticks[i], end_ticks[i], attacker)
            context_window["attacker_shot"] = MDP.get_attacker_shots(start_ticks[i], end_ticks[i], attacker, weapon_fire_idx)
            (context_window["attacker_kill"],
             context_window["is_kill_headshot"],
             context_window["is_kill_through_smoke"], 
             context_window["is_kill_wallbang"], 
             context_window["attacker_midair"]) = MDP.get_attacker_kill_data(start_ticks[i], end_ticks[i], attacker, player_death_idx)

            (context_window["attacker_weapon_knife"], 
             context_window["attacker_weapon_auto_rifle"], 
             context_window["attacker_weapon_semi_rifle"], 
             context_window["attacker_weapon_pistol"], 
             context_window["attacker_weapon_grenade"], 
             context_window["attacker_weapon_smg"], 
             context_window["attacker_weapon_shotgun"]) = MDP.get_attacker_weapon(start_ticks[i], end_ticks[i], attacker)

            (context_window["victim_X"],
             context_window["victim_Y"],
             context_window["victim_Z"]) = MDP.get_player_coordinates(start_ticks[i], end_ticks[i], victim, played_map)

            context_window["victim_health"] = MDP.get_player_health(start_ticks[i], end_ticks[i], victim)
            context_window["victim_noise"] = MDP.get_player_made_noise(start_ticks[i], end_ticks[i], attacker, weapon_fire_idx)

            (context_window["map_dust2"],
             context_window["map_mirage"],
             context_window["map_inferno"],
             context_window["map_train"],
             context_window["map_nuke"],
             context_window["map_ancient"],
             context_window["map_vertigo"],
             context_window["map_anubis"],
             context_window["map_office"],
             context_window["map_overpass"],
             context_window["map_basalt"],
             context_window["map_edin"],
             context_window["map_italy"],
             context_window["map_thera"],
             context_window["map_mills"]) = MDP.get_played_map(played_map)

            # Filename structure: c(cheater dataset)/n(not cheater dataset)_c(player cheater)/n(player not cheater)_fileidx_playername_killnr.parquet
            c_n = "cheater" if is_attacker_cheater else "notcheater"

            out_dir = cheater_out_dir if c_n == "cheater" else non_cheater_out_dir
            context_window = context_window.astype(np.float32)
            context_window.to_parquet(fr"{out_dir}\{is_cheater_data}-{c_n}-file_{file_idx}-{attacker}-kill_{i}.parquet", index=False)
            context_window.to_csv(fr"C:\Users\Gert\repos\CS2_cheat_detection\DataConversionPipeline\test_data\killtest.csv", index=True)

    print(f"file idx {file_idx} done")
