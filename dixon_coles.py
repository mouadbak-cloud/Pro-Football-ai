# ============================================================
#  pro-football-ai / models/dixon_coles.py
#  Dixon-Coles Poisson Model — the statistical core
#
#  Reference: Dixon & Coles (1997) "Modelling Association
#  Football Scores and Inefficiencies in the Football
#  Betting Market", Applied Statistics, 46(2), 265-280.
# ============================================================

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import math

from config import DC_HOME_ADVANTAGE, DC_RHO, DC_MAX_GOALS, DC_TIME_DECAY


class DixonColes:
    """
    Full Dixon-Coles model with:
    - Maximum Likelihood Estimation of attack/defense parameters
    - Time-decay weighting (recent matches matter more)
    - Low-score correction (DC rho parameter)
    - All betting market predictions
    """

    def __init__(self):
        self.attack:  Dict[str, float] = {}
        self.defense: Dict[str, float] = {}
        self.home_adv: float = DC_HOME_ADVANTAGE
        self.rho:      float = DC_RHO
        self.mu:       float = 1.0   # global mean goals (intercept)
        self.fitted:   bool  = False
        self.teams:    list  = []

    # ── LOW-SCORE CORRECTION ─────────────────────────────

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        """
        Dixon-Coles tau correction for low-scoring matches.
        Corrects the over-dispersion in 0-0, 1-0, 0-1, 1-1 scorelines.
        """
        if x == 0 and y == 0:
            return 1.0 - lam * mu * rho
        elif x == 1 and y == 0:
            return 1.0 + mu * rho
        elif x == 0 and y == 1:
            return 1.0 + lam * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        else:
            return 1.0

    # ── LAMBDA CALCULATION ───────────────────────────────

    def _lambda(self, home: str, away: str) -> Tuple[float, float]:
        """Compute expected goals for home and away teams."""
        lam_h = math.exp(
            self.mu
            + self.attack.get(home,  0.0)
            + self.defense.get(away, 0.0)
            + self.home_adv
        )
        lam_a = math.exp(
            self.mu
            + self.attack.get(away,  0.0)
            + self.defense.get(home, 0.0)
        )
        return max(lam_h, 0.01), max(lam_a, 0.01)

    # ── SCORE MATRIX ─────────────────────────────────────

    def score_matrix(self, home: str, away: str, max_goals: int = DC_MAX_GOALS) -> np.ndarray:
        """
        Build probability matrix P[home_goals][away_goals].
        Shape: (max_goals+1, max_goals+1)
        """
        lam_h, lam_a = self._lambda(home, away)
        m = np.zeros((max_goals + 1, max_goals + 1))

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = (poisson.pmf(i, lam_h)
                     * poisson.pmf(j, lam_a)
                     * self._tau(i, j, lam_h, lam_a, self.rho))
                m[i][j] = max(p, 0.0)

        # Normalize
        total = m.sum()
        if total > 0:
            m /= total
        return m

    # ── CORE PREDICTIONS ─────────────────────────────────

    def predict(self, home: str, away: str) -> dict:
        """Full prediction for a match."""
        lam_h, lam_a = self._lambda(home, away)
        m = self.score_matrix(home, away)
        max_g = m.shape[0] - 1

        # 1x2
        p_home_win = float(np.sum(np.tril(m, -1)))
        p_draw     = float(np.trace(m))
        p_away_win = float(np.sum(np.triu(m, 1)))

        # Normalize
        total_1x2 = p_home_win + p_draw + p_away_win
        p_home_win /= total_1x2
        p_draw     /= total_1x2
        p_away_win /= total_1x2

        # Top correct scores
        scores = []
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                scores.append((i, j, float(m[i][j])))
        scores.sort(key=lambda x: x[2], reverse=True)

        # Over/Under lines — vectorised using cumsum on total goals
        tg_matrix = np.zeros(max_g * 2 + 2)
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                tg_matrix[i + j] += m[i][j]
        ou = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            thresh = int(line + 0.5)
            p_over = float(tg_matrix[thresh:].sum())
            ou[f"over_{line}"]  = p_over
            ou[f"under_{line}"] = 1.0 - p_over

        # BTTS
        btts_yes = float(m[1:, 1:].sum())
        btts_no  = 1.0 - btts_yes

        # Asian Handicap — vectorised
        ah = {}
        diff_matrix = np.zeros((max_g * 2 + 1,))  # index = (home-away) + max_g
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                diff_matrix[i - j + max_g] += m[i][j]
        for line in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            thresh = int(line) + max_g
            if line == int(line):  # integer line — push possible
                p_h = float(diff_matrix[thresh+1:].sum())
                p_a = float(diff_matrix[:thresh].sum())
            else:
                p_h = float(diff_matrix[thresh+1:].sum())
                p_a = float(diff_matrix[:thresh+1].sum())
            ah[str(line)] = {"home": p_h, "away": p_a, "push": max(1.0-p_h-p_a, 0.0)}

        # Exact totals
        exact = {}
        for n in range(7):
            exact[str(n)] = float(tg_matrix[n]) if n < len(tg_matrix) else 0.0
        exact["5+"] = float(tg_matrix[5:].sum())

        # Win to Nil
        win_nil_home = float(m[1:, 0].sum())
        win_nil_away = float(m[0, 1:].sum())

        # Confidence
        max_p = max(p_home_win, p_draw, p_away_win)
        gap   = abs(max_p - 1/3)
        if gap >= 0.20:   conf = "A"
        elif gap >= 0.12: conf = "B"
        elif gap >= 0.06: conf = "C"
        else:             conf = "D"

        return {
            "home_team":   home,
            "away_team":   away,
            "xg_home":     round(lam_h, 3),
            "xg_away":     round(lam_a, 3),
            "p_home_win":  round(p_home_win, 4),
            "p_draw":      round(p_draw,     4),
            "p_away_win":  round(p_away_win, 4),
            "result":      ("HOME WIN" if p_home_win >= p_draw and p_home_win >= p_away_win
                            else "DRAW" if p_draw >= p_home_win and p_draw >= p_away_win
                            else "AWAY WIN"),
            "confidence":  conf,
            "top_scores":  [(s[0], s[1], round(s[2], 4)) for s in scores[:10]],
            "over_under":  {k: round(v, 4) for k, v in ou.items()},
            "btts":        {"yes": round(btts_yes, 4), "no": round(btts_no, 4)},
            "asian_handicap": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ah.items()},
            "exact_totals": {k: round(v, 4) for k, v in exact.items()},
            "win_to_nil":  {"home": round(win_nil_home, 4), "away": round(win_nil_away, 4)},
            "model":       "dixon-coles-v2",
        }

    # ── MLE FITTING ──────────────────────────────────────

    def fit(self, matches: List[dict], reference_team: str = None) -> "DixonColes":
        """
        Fit attack/defense parameters via Maximum Likelihood Estimation.

        matches: list of dicts with keys:
            home_team, away_team, home_goals, away_goals, date (optional)
        """
        teams = sorted(set(
            [m["home_team"] for m in matches] +
            [m["away_team"] for m in matches]
        ))
        self.teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        if reference_team is None:
            reference_team = teams[0]
        ref_idx = idx[reference_team]

        # Time decay weights
        import datetime
        today = datetime.date.today()
        weights = []
        for m in matches:
            try:
                d = datetime.date.fromisoformat(str(m.get("date", today))[:10])
                days_ago = (today - d).days
                w = math.exp(-DC_TIME_DECAY * days_ago)
            except Exception:
                w = 1.0
            weights.append(w)

        def neg_log_likelihood(params):
            mu     = params[0]
            home   = params[1]
            rho    = params[2]
            attack = dict(zip(teams, params[3:3+n]))
            defense= dict(zip(teams, params[3+n:3+2*n]))

            ll = 0.0
            for i, m in enumerate(matches):
                ht = m["home_team"]
                at = m["away_team"]
                hg = int(m["home_goals"])
                ag = int(m["away_goals"])

                lam_h = max(math.exp(mu + attack[ht] + defense[at] + home), 1e-6)
                lam_a = max(math.exp(mu + attack[at] + defense[ht]       ), 1e-6)

                t = self._tau(hg, ag, lam_h, lam_a, rho)
                if t <= 0:
                    return 1e9

                log_p = (poisson.logpmf(hg, lam_h)
                       + poisson.logpmf(ag, lam_a)
                       + math.log(max(t, 1e-9)))
                ll += weights[i] * log_p

            return -ll

        # Initial parameters: [mu, home_adv, rho, *attack_n, *defense_n]
        x0 = [0.3, 0.25, 0.1] + [0.0] * n + [0.0] * n

        # Constraint: attack[reference] = 0 (identification)
        constraints = [
            {"type": "eq", "fun": lambda p: p[3 + ref_idx]},    # attack ref = 0
            {"type": "eq", "fun": lambda p: p[3 + n + ref_idx]}, # defense ref = 0
        ]

        bounds = (
            [(-1, 2),   # mu
             (0, 1),    # home_adv
             (-0.2, 0.2)] # rho
            + [(-2, 2)] * n   # attack
            + [(-2, 2)] * n   # defense
        )

        print(f"⚙️  Fitting Dixon-Coles on {len(matches)} matches, {n} teams...")
        result = minimize(
            neg_log_likelihood, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9}
        )

        params = result.x
        self.mu       = params[0]
        self.home_adv = params[1]
        self.rho      = params[2]
        self.attack   = dict(zip(teams, params[3:3+n]))
        self.defense  = dict(zip(teams, params[3+n:3+2*n]))
        self.fitted   = True

        print(f"✅ Dixon-Coles fitted. Home adv={self.home_adv:.3f}, rho={self.rho:.3f}")
        return self

    def set_params_from_stats(self, team: str, gf: float, ga: float,
                               league_avg_gf: float = 1.4, league_avg_ga: float = 1.4):
        """
        Set attack/defense directly from average stats (no MLE needed).
        Used when we have SoccerSTATS data but no raw match history.
        """
        self.attack[team]  = math.log(max(gf / league_avg_gf, 0.1))
        self.defense[team] = math.log(max(ga / league_avg_ga, 0.1)) * -1  # lower = better defense

    def strength_table(self) -> list:
        """Return all teams sorted by net strength (attack - defense)."""
        rows = []
        for t in self.teams:
            att = self.attack.get(t, 0)
            dfc = self.defense.get(t, 0)
            rows.append({"team": t, "attack": round(att, 3), "defense": round(dfc, 3), "net": round(att - dfc, 3)})
        return sorted(rows, key=lambda x: x["net"], reverse=True)
