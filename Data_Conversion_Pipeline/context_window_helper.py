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
    X = match_ticks[(match_ticks["name"] == player) & (match_ticks["tick"] <= end_tick) & (match_ticks["tick"] > start_tick)]["X"]
    Y = match_ticks[(match_ticks["name"] == player) & (match_ticks["tick"] <= end_tick) & (match_ticks["tick"] > start_tick)]["Y"]
    Z = match_ticks[(match_ticks["name"] == player) & (match_ticks["tick"] <= end_tick) & (match_ticks["tick"] > start_tick)]["Z"]
    return (X, Y, Z)