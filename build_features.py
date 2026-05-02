# ============================================================
#  pro-football-ai / features/build_features.py
#  Feature Engineering — transforms raw data into model inputs
# ============================================================

import math
import numpy as np
from typing import List, Dict, Optional


def compute_form(matches: List[dict], team: str, window: int = 5) -> dict:
    """
    Compute recent form metrics for a team from match history.

    Args:
        matches: list of match dicts (sorted newest first)
        team: team name
        window: last N matches to consider

    Returns:
        dict with avg_gf, avg_ga, win_rate, draw_rate, points, form_str
    """
    relevant = []
    for m in matches:
        if m.get("home_team") == team or m.get("away_team") == team:
            relevant.append(m)
        if len(relevant) >= window:
            break

    if not relevant:
        return {"avg_gf": 1.2, "avg_ga": 1.2, "win_rate": 0.4,
                "draw_rate": 0.25, "points": 6.0, "form_str": "?????", "n": 0}

    gf_list, ga_list, pts_list, form_chars = [], [], [], []

    for m in relevant:
        is_home = m.get("home_team") == team
        gf = m.get("home_goals" if is_home else "away_goals", 0) or 0
        ga = m.get("away_goals" if is_home else "home_goals", 0) or 0
        gf_list.append(gf)
        ga_list.append(ga)

        if gf > ga:
            pts_list.append(3)
            form_chars.append("W")
        elif gf == ga:
            pts_list.append(1)
            form_chars.append("D")
        else:
            pts_list.append(0)
            form_chars.append("L")

    n = len(relevant)
    wins  = form_chars.count("W")
    draws = form_chars.count("D")

    return {
        "avg_gf":    round(sum(gf_list) / n, 3),
        "avg_ga":    round(sum(ga_list) / n, 3),
        "win_rate":  round(wins  / n, 3),
        "draw_rate": round(draws / n, 3),
        "points":    sum(pts_list),
        "form_str":  "".join(form_chars),
        "n":         n,
    }


def compute_h2h(matches: List[dict], home: str, away: str) -> dict:
    """Compute head-to-head statistics."""
    h2h = [m for m in matches
           if (m.get("home_team") == home and m.get("away_team") == away)
           or (m.get("home_team") == away and m.get("away_team") == home)]

    if not h2h:
        return {"home_avg_gf": 0, "home_avg_ga": 0, "n_matches": 0,
                "home_wins": 0, "draws": 0, "away_wins": 0}

    home_gf, home_ga, hw, d, aw = [], [], 0, 0, 0

    for m in h2h:
        if m.get("home_team") == home:
            gf = m.get("home_goals", 0) or 0
            ga = m.get("away_goals", 0) or 0
        else:
            gf = m.get("away_goals", 0) or 0
            ga = m.get("home_goals", 0) or 0
        home_gf.append(gf)
        home_ga.append(ga)
        if gf > ga:   hw += 1
        elif gf == ga: d += 1
        else:          aw += 1

    n = len(h2h)
    return {
        "home_avg_gf": round(sum(home_gf) / n, 3),
        "home_avg_ga": round(sum(home_ga) / n, 3),
        "n_matches":   n,
        "home_wins":   hw,
        "draws":       d,
        "away_wins":   aw,
        "home_win_pct": round(hw / n, 3),
    }


def attack_defense_strength(matches: List[dict], team: str,
                             league_avg_gf: float = 1.4,
                             league_avg_ga: float = 1.4,
                             window: int = 10) -> dict:
    """
    Compute Dixon-Coles-style attack/defense strength indices.
    """
    relevant = []
    for m in sorted(matches, key=lambda x: x.get("date", ""), reverse=True):
        if m.get("home_team") == team or m.get("away_team") == team:
            relevant.append(m)
        if len(relevant) >= window:
            break

    if not relevant:
        return {"attack": 1.0, "defense": 1.0, "home_attack": 1.0, "away_defense": 1.0}

    gf_all, ga_all = [], []
    gf_home, ga_home = [], []
    gf_away, ga_away = [], []

    for m in relevant:
        is_home = m.get("home_team") == team
        gf = m.get("home_goals" if is_home else "away_goals", 0) or 0
        ga = m.get("away_goals" if is_home else "home_goals", 0) or 0
        gf_all.append(gf); ga_all.append(ga)
        if is_home:
            gf_home.append(gf); ga_home.append(ga)
        else:
            gf_away.append(gf); ga_away.append(ga)

    n = len(relevant)
    avg_gf = sum(gf_all) / n if n > 0 else league_avg_gf
    avg_ga = sum(ga_all) / n if n > 0 else league_avg_ga

    # Strength = team average / league average
    attack_str  = avg_gf / max(league_avg_gf, 0.5)
    defense_str = avg_ga / max(league_avg_ga, 0.5)  # lower = better

    n_h = len(gf_home)
    n_a = len(gf_away)

    return {
        "attack":       round(attack_str, 3),
        "defense":      round(defense_str, 3),
        "home_attack":  round((sum(gf_home)/n_h / league_avg_gf) if n_h else attack_str, 3),
        "home_defense": round((sum(ga_home)/n_h / league_avg_ga) if n_h else defense_str, 3),
        "away_attack":  round((sum(gf_away)/n_a / league_avg_gf) if n_a else attack_str, 3),
        "away_defense": round((sum(ga_away)/n_a / league_avg_ga) if n_a else defense_str, 3),
        "avg_gf":       round(avg_gf, 3),
        "avg_ga":       round(avg_ga, 3),
        "n_matches":    n,
    }


def compute_xg(home_stats: dict, away_stats: dict,
               home_elo: float = 1500, away_elo: float = 1500,
               h2h: Optional[dict] = None,
               home_adv: float = 0.25) -> dict:
    """
    Compute expected goals using multi-signal approach:
    - Attack/defense strength indices (from stats or computed)
    - Elo difference adjustment
    - H2H blend
    - Home advantage
    """
    # League average (blend of both teams as approximation)
    lg_gf = max((home_stats.get("gf", 1.4) + away_stats.get("gf", 1.4)) / 2, 0.7)
    lg_ga = max((home_stats.get("ga", 1.4) + away_stats.get("ga", 1.4)) / 2, 0.7)
    lg    = (lg_gf + lg_ga) / 2

    # Attack/defense strength
    h_att = home_stats.get("hgf", home_stats.get("gf", 1.4)) / lg
    h_def = home_stats.get("hga", home_stats.get("ga", 1.4)) / lg
    a_att = away_stats.get("agf", away_stats.get("gf", 1.4)) / lg
    a_def = away_stats.get("aga", away_stats.get("ga", 1.4)) / lg

    # Base expected goals
    xg_h = h_att * a_def * lg * math.exp(home_adv)
    xg_a = a_att * h_def * lg

    # Elo adjustment (small correction based on quality difference)
    elo_diff   = home_elo - away_elo
    elo_factor = math.tanh(elo_diff / 800.0) * 0.15  # max ±15% adjustment
    xg_h *= (1.0 + elo_factor)
    xg_a *= (1.0 - elo_factor)

    # FTS correction (teams that fail to score often)
    xg_h *= max(1.0 - (home_stats.get("fts", 20) / 100) * 0.2, 0.7)
    xg_a *= max(1.0 - (away_stats.get("fts", 20) / 100) * 0.2, 0.7)

    # BTS signal: if both teams tend to score, raise the floor
    bts_mean = (home_stats.get("bts", 45) + away_stats.get("bts", 45)) / 200
    if bts_mean > 0.55:
        xg_h = max(xg_h, 0.9)
        xg_a = max(xg_a, 0.7)

    # H2H blend (20% weight)
    if h2h and h2h.get("n_matches", 0) >= 3:
        xg_h = xg_h * 0.80 + h2h["home_avg_gf"] * 0.20
        xg_a = xg_a * 0.80 + h2h["home_avg_ga"] * 0.20

    # First half xG (historical ~45% of FT goals in first half)
    fh_scale_h = home_stats.get("fhgf", home_stats.get("gf", 1.4) * 0.45) / max(home_stats.get("gf", 1.4), 0.1)
    fh_scale_a = away_stats.get("fhgf", away_stats.get("gf", 1.4) * 0.45) / max(away_stats.get("gf", 1.4), 0.1)
    xg_fh_h = xg_h * fh_scale_h * math.exp(home_adv * 0.5)
    xg_fh_a = xg_a * fh_scale_a

    return {
        "xg_home":    round(max(xg_h, 0.1),   3),
        "xg_away":    round(max(xg_a, 0.1),   3),
        "xg_fh_home": round(max(xg_fh_h, 0.05), 3),
        "xg_fh_away": round(max(xg_fh_a, 0.05), 3),
    }


def build_training_features(matches: List[dict], all_matches: List[dict]) -> tuple:
    """
    Build feature matrix + labels for XGBoost training.
    """
    X_rows, y_labels = [], []

    for m in matches:
        home = m.get("home_team")
        away = m.get("away_team")
        hg   = m.get("home_goals")
        ag   = m.get("away_goals")

        if hg is None or ag is None:
            continue

        # Get matches BEFORE this one
        match_date = m.get("date", "")
        prior = [mm for mm in all_matches
                 if mm.get("date", "") < match_date
                 and (mm.get("home_team") in (home, away)
                      or mm.get("away_team") in (home, away))]

        form_h = compute_form(prior, home)
        form_a = compute_form(prior, away)
        h2h    = compute_h2h(prior, home, away)
        str_h  = attack_defense_strength(prior, home)
        str_a  = attack_defense_strength(prior, away)
        xg     = compute_xg({"gf": str_h["avg_gf"], "ga": str_h["avg_ga"]},
                             {"gf": str_a["avg_gf"], "ga": str_a["avg_ga"]})

        feat = [
            xg["xg_home"] / (xg["xg_home"] + xg["xg_away"] + 1e-6),  # normalized xg share
            xg["xg_home"], xg["xg_away"], xg["xg_home"] - xg["xg_away"],
            form_h["avg_gf"], form_h["avg_ga"], form_h["points"],
            form_a["avg_gf"], form_a["avg_ga"], form_a["points"],
            str_h["attack"], str_h["defense"],
            str_a["attack"], str_a["defense"],
            h2h["n_matches"], h2h["home_avg_gf"], h2h["home_avg_ga"],
        ]
        X_rows.append(feat)

        # Label: 0=home, 1=draw, 2=away
        label = 0 if hg > ag else (1 if hg == ag else 2)
        y_labels.append(label)

    return np.array(X_rows, dtype=np.float32), np.array(y_labels, dtype=np.int32)
