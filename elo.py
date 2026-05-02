# ============================================================
#  pro-football-ai / models/elo.py
#  Dynamic Elo Rating System — football-specific variant
# ============================================================

import math
from typing import Dict, Optional
from config import ELO_K_FACTOR, ELO_DEFAULT_RATING, ELO_HOME_ADVANTAGE


class EloRating:
    """
    Football-specific Elo rating system.

    Differences from chess Elo:
    - Home advantage baked into expected score calculation
    - Goal difference used to scale K (margin of victory multiplier)
    - Draw is a valid outcome (0.5 for both teams)
    """

    def __init__(self, k: float = ELO_K_FACTOR, home_adv: float = ELO_HOME_ADVANTAGE):
        self.ratings: Dict[str, float] = {}
        self.k = k
        self.home_adv = home_adv  # Elo points added to home team's rating
        self.history: list = []

    def get(self, team: str) -> float:
        return self.ratings.get(team, ELO_DEFAULT_RATING)

    def set(self, team: str, rating: float):
        self.ratings[team] = rating

    def expected_score(self, team_a_rating: float, team_b_rating: float) -> float:
        """Standard Elo expected score for team A vs team B."""
        return 1.0 / (1.0 + 10 ** ((team_b_rating - team_a_rating) / 400.0))

    def _margin_multiplier(self, goal_diff: int, elo_diff: float) -> float:
        """
        Margin of Victory Multiplier (MoVM) — used in NFL Elo systems.
        Adapted for football: larger wins count more but with diminishing returns.
        """
        abs_diff = abs(goal_diff)
        if abs_diff == 0:
            return 1.0
        # Log-based multiplier (diminishing returns for large victories)
        raw = math.log(abs_diff + 1.0) * 2.2
        # Autocorrelation correction: big teams beating small teams by a lot
        # shouldn't get as much credit as equal teams
        correction = 1.0 / (elo_diff * 0.001 + 1.0) if elo_diff > 0 else 1.0
        return max(raw * correction, 1.0)

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               k_factor: Optional[float] = None) -> tuple:
        """
        Update Elo ratings after a match.

        Returns:
            (new_home_elo, new_away_elo, elo_change_home)
        """
        k = k_factor or self.k

        # Get current ratings
        home_r = self.get(home)
        away_r = self.get(away)

        # Apply home advantage to expected score
        home_r_adj = home_r + self.home_adv

        # Expected scores
        exp_home = self.expected_score(home_r_adj, away_r)
        exp_away = 1.0 - exp_home

        # Actual scores
        if home_goals > away_goals:
            act_home, act_away = 1.0, 0.0
        elif home_goals == away_goals:
            act_home, act_away = 0.5, 0.5
        else:
            act_home, act_away = 0.0, 1.0

        # Margin of victory multiplier
        goal_diff = home_goals - away_goals
        elo_diff  = abs(home_r_adj - away_r)
        movm      = self._margin_multiplier(goal_diff, elo_diff)

        # Elo updates
        delta_home = k * movm * (act_home - exp_home)
        delta_away = k * movm * (act_away - exp_away)

        new_home_r = home_r + delta_home
        new_away_r = away_r + delta_away

        self.ratings[home] = new_home_r
        self.ratings[away] = new_away_r

        self.history.append({
            "home": home, "away": away,
            "home_goals": home_goals, "away_goals": away_goals,
            "home_elo_before": home_r, "away_elo_before": away_r,
            "home_elo_after": new_home_r, "away_elo_after": new_away_r,
            "delta": delta_home
        })

        return new_home_r, new_away_r, delta_home

    def win_probability(self, home: str, away: str) -> dict:
        """
        Compute win/draw/loss probabilities from Elo ratings alone.
        Uses a logistic curve calibrated for football.
        """
        home_r = self.get(home) + self.home_adv
        away_r = self.get(away)

        elo_diff = home_r - away_r

        # Calibrated for football (draw probability peaks at ~0 diff)
        p_home_win = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

        # Draw probability model: gaussian centered at 0 elo diff
        draw_factor = 0.22  # empirical draw rate in football
        p_draw = draw_factor * math.exp(-((elo_diff / 400.0) ** 2) / 0.5)

        # Normalize
        total = p_home_win + p_draw + (1 - p_home_win)
        p_home_win_n = max(p_home_win - p_draw / 2, 0.05)
        p_away_win_n = max((1 - p_home_win) - p_draw / 2, 0.05)
        p_draw_n     = min(p_draw, 0.40)

        # Re-normalize to sum to 1
        s = p_home_win_n + p_draw_n + p_away_win_n
        return {
            "home_win": round(p_home_win_n / s, 4),
            "draw":     round(p_draw_n     / s, 4),
            "away_win": round(p_away_win_n / s, 4),
            "home_elo": round(self.get(home), 1),
            "away_elo": round(self.get(away), 1),
        }

    def top_teams(self, n: int = 20) -> list:
        """Return top N teams by Elo rating."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:n]

    def build_from_matches(self, matches: list):
        """
        Build Elo ratings from a list of match dicts.
        matches: [{"date", "home_team", "away_team", "home_goals", "away_goals"}, ...]
        Sorted by date ascending.
        """
        sorted_matches = sorted(matches, key=lambda m: m.get("date", ""))
        for m in sorted_matches:
            try:
                self.update(
                    m["home_team"], m["away_team"],
                    int(m["home_goals"]), int(m["away_goals"])
                )
            except (KeyError, TypeError, ValueError):
                continue
        return self
