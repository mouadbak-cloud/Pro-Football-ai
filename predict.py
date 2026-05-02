# ============================================================
#  pro-football-ai / pipeline/predict.py
#  Full Prediction Pipeline — orchestrates all 4 layers
# ============================================================

import math
from typing import Optional
import numpy as np

from models.elo          import EloRating
from models.dixon_coles  import DixonColes
from models.ml_calibration import MLCalibrator
from features.build_features import compute_form, compute_h2h, compute_xg
from data.db import (get_team_elo, get_team_matches, get_h2h,
                     get_latest_stats, save_prediction, update_elo)
from scraper.soccerstats_scraper import get_team_stats, search_teams


# Global singletons
_elo = EloRating()
_dc  = DixonColes()
_ml  = MLCalibrator()


def _ensure_stats(team_name: str, league: str, stats_id: str = None) -> dict:
    """
    Get team stats from DB cache or scrape fresh.
    """
    # Try DB first
    cached = get_latest_stats(team_name)
    if cached:
        return cached

    # Scrape fresh
    if not stats_id:
        teams = search_teams(league, team_name)
        if not teams or "error" in teams[0]:
            raise Exception(f"Team '{team_name}' not found in league '{league}'")
        stats_id = teams[0]["stats"]

    stats = get_team_stats(league, stats_id)
    # Save to DB
    from data.db import save_team_stats
    save_team_stats(stats)
    return stats


def predict_match(home_name: str, away_name: str, league: str,
                  home_stats_id: str = None, away_stats_id: str = None) -> dict:
    """
    Full 4-layer prediction pipeline.

    Layer 1: Data (fetch/cache stats)
    Layer 2: Feature Engineering (form, H2H, strength)
    Layer 3: Dixon-Coles Poisson Model
    Layer 4: XGBoost Calibration (if trained)
    """

    # ── LAYER 1: Data ──────────────────────────────────
    home_stats = _ensure_stats(home_name, league, home_stats_id)
    away_stats = _ensure_stats(away_name, league, away_stats_id)

    # Get match history for form
    home_matches = get_team_matches(home_name, limit=20)
    away_matches = get_team_matches(away_name, limit=20)
    h2h_matches  = get_h2h(home_name, away_name, limit=10)

    # ── LAYER 2: Feature Engineering ──────────────────
    all_matches = home_matches + [m for m in away_matches if m not in home_matches]

    home_form = compute_form(home_matches, home_name, window=5)
    away_form = compute_form(away_matches, away_name, window=5)
    h2h_stats = compute_h2h(h2h_matches,  home_name, away_name)

    home_elo = get_team_elo(home_name)
    away_elo = get_team_elo(away_name)
    elo_probs = _elo.win_probability(home_name, away_name)

    xg = compute_xg(
        home_stats, away_stats,
        home_elo=home_elo, away_elo=away_elo,
        h2h=h2h_stats
    )

    # ── LAYER 3: Dixon-Coles ───────────────────────────
    # Set params from stats (fast path — no MLE fitting needed)
    lg_avg_gf = (home_stats.get("gf", 1.4) + away_stats.get("gf", 1.4)) / 2
    lg_avg_ga = (home_stats.get("ga", 1.4) + away_stats.get("ga", 1.4)) / 2

    _dc.set_params_from_stats(home_name,
                               home_stats.get("hgf", home_stats.get("gf", 1.4)),
                               home_stats.get("hga", home_stats.get("ga", 1.4)),
                               lg_avg_gf, lg_avg_ga)
    _dc.set_params_from_stats(away_name,
                               away_stats.get("agf", away_stats.get("gf", 1.4)),
                               away_stats.get("aga", away_stats.get("ga", 1.4)),
                               lg_avg_gf, lg_avg_ga)

    # Override the DC lambdas with our computed xG (more accurate)
    _dc.mu       = 0.0
    _dc.home_adv = 0.0

    dc_pred = _dc.predict(home_name, away_name)

    # Override xG with our multi-signal computation
    dc_pred["xg_home"] = xg["xg_home"]
    dc_pred["xg_away"] = xg["xg_away"]

    # ── LAYER 4: XGBoost Calibration ──────────────────
    ml_input = {
        "dc_p_home":  dc_pred["p_home_win"],
        "dc_p_draw":  dc_pred["p_draw"],
        "dc_p_away":  dc_pred["p_away_win"],
        "xg_home":    xg["xg_home"],
        "xg_away":    xg["xg_away"],
        "elo_home":   home_elo,
        "elo_away":   away_elo,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "home_form":  home_form,
        "away_form":  away_form,
        "h2h":        h2h_stats,
    }
    calibrated = _ml.calibrate(ml_input)

    # Use calibrated probs if available, otherwise use DC
    if calibrated:
        final_p_home = calibrated["p_home_win"]
        final_p_draw = calibrated["p_draw"]
        final_p_away = calibrated["p_away_win"]
        model_used   = "dixon-coles + xgboost"
    else:
        final_p_home = dc_pred["p_home_win"]
        final_p_draw = dc_pred["p_draw"]
        final_p_away = dc_pred["p_away_win"]
        model_used   = "dixon-coles"

    # ── BUILD FIRST HALF PREDICTIONS ──────────────────
    fh_dc = DixonColes()
    fh_dc.set_params_from_stats(home_name,
                                 home_stats.get("fhgf", home_stats.get("gf", 1.4) * 0.45),
                                 home_stats.get("fhga", home_stats.get("ga", 1.4) * 0.45),
                                 lg_avg_gf * 0.45, lg_avg_ga * 0.45)
    fh_dc.set_params_from_stats(away_name,
                                 away_stats.get("fhgf", away_stats.get("gf", 1.4) * 0.45),
                                 away_stats.get("fhga", away_stats.get("ga", 1.4) * 0.45),
                                 lg_avg_gf * 0.45, lg_avg_ga * 0.45)
    fh_pred = fh_dc.predict(home_name, away_name)

    # ── ASSEMBLE FINAL RESULT ─────────────────────────
    result_label = ("HOME WIN" if final_p_home >= final_p_draw and final_p_home >= final_p_away
                    else "DRAW"  if final_p_draw  >= final_p_home and final_p_draw  >= final_p_away
                    else "AWAY WIN")

    max_p = max(final_p_home, final_p_draw, final_p_away)
    gap   = abs(max_p - 1/3)
    confidence = ("A ★★★" if gap >= 0.20 else
                  "B ★★☆" if gap >= 0.12 else
                  "C ★☆☆" if gap >= 0.06 else
                  "D ☆☆☆")

    prediction = {
        # Identity
        "home_team": home_stats.get("name", home_name),
        "away_team": away_stats.get("name", away_name),
        "league":    league,

        # Expected Goals
        "xg_home":    xg["xg_home"],
        "xg_away":    xg["xg_away"],
        "xg_fh_home": xg["xg_fh_home"],
        "xg_fh_away": xg["xg_fh_away"],

        # 1X2
        "p_home_win": round(final_p_home, 4),
        "p_draw":     round(final_p_draw, 4),
        "p_away_win": round(final_p_away, 4),
        "result":     result_label,
        "confidence": confidence,

        # Correct scores
        "top_scores":    dc_pred["top_scores"][:10],
        "ht_top_scores": fh_pred["top_scores"][:5],

        # Over/Under
        "over_under":     dc_pred["over_under"],

        # BTTS
        "btts":           dc_pred["btts"],

        # Asian Handicap
        "asian_handicap": dc_pred["asian_handicap"],

        # Exact totals
        "exact_totals":   dc_pred["exact_totals"],

        # Win to Nil
        "win_to_nil":     dc_pred["win_to_nil"],

        # First Half
        "fh_result": {
            "p_home": round(fh_pred["p_home_win"], 4),
            "p_draw": round(fh_pred["p_draw"],     4),
            "p_away": round(fh_pred["p_away_win"], 4),
        },
        "fh_over_under": fh_pred["over_under"],
        "fh_btts":       fh_pred["btts"],

        # Context
        "elo": {
            "home":     round(home_elo, 1),
            "away":     round(away_elo, 1),
            "diff":     round(home_elo - away_elo, 1),
            "elo_probs": elo_probs,
        },
        "form": {
            "home": home_form,
            "away": away_form,
        },
        "h2h":       h2h_stats,
        "home_stats": {k: v for k, v in home_stats.items() if k != "raw_json"},
        "away_stats": {k: v for k, v in away_stats.items() if k != "raw_json"},
        "model_used": model_used,
    }

    # Save to DB
    try:
        save_prediction({
            "home_team":       prediction["home_team"],
            "away_team":       prediction["away_team"],
            "league":          league,
            "xg_home":         xg["xg_home"],
            "xg_away":         xg["xg_away"],
            "p_home_win":      final_p_home,
            "p_draw":          final_p_draw,
            "p_away_win":      final_p_away,
            "predicted_score": f"{prediction['top_scores'][0][0]}-{prediction['top_scores'][0][1]}",
            "confidence":      confidence,
        })
    except Exception:
        pass

    return prediction
