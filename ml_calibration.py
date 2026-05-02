# ============================================================
#  pro-football-ai / models/ml_calibration.py
#  XGBoost Calibration Layer — corrects Poisson probabilities
#
#  Takes Dixon-Coles output + Elo features as input,
#  predicts actual match outcomes using gradient boosting.
# ============================================================

import numpy as np
import json
import os
from pathlib import Path
from typing import Optional

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")

from config import XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_SUBSAMPLE, XGB_MIN_MATCHES

MODEL_PATH = Path(__file__).parent.parent / "data" / "xgb_model.json"


class MLCalibrator:
    """
    Gradient Boosting calibration layer.

    Features used:
    - Dixon-Coles probabilities (p_home, p_draw, p_away)
    - Expected goals (xg_home, xg_away)
    - Elo ratings and difference
    - Recent form (last 5 goals for/against)
    - Head-to-head history
    - Home/Away split stats
    - League-specific features
    """

    def __init__(self):
        self.model = None
        self.feature_names = [
            # Dixon-Coles core
            "dc_p_home",       # DC home win probability
            "dc_p_draw",       # DC draw probability
            "dc_p_away",       # DC away win probability
            "xg_home",         # expected goals home
            "xg_away",         # expected goals away
            "xg_diff",         # xg_home - xg_away

            # Elo
            "elo_home",        # home team Elo
            "elo_away",        # away team Elo
            "elo_diff",        # elo_home - elo_away

            # Form (last 5)
            "home_form_gf",    # home team avg goals scored last 5
            "home_form_ga",    # home team avg goals conceded last 5
            "away_form_gf",
            "away_form_ga",
            "home_form_pts",   # home team points last 5 (W=3, D=1, L=0)
            "away_form_pts",

            # Stats from SoccerSTATS
            "home_gf_avg",     # season avg goals for
            "home_ga_avg",
            "away_gf_avg",
            "away_ga_avg",
            "home_cs_pct",     # clean sheet %
            "away_cs_pct",
            "home_fts_pct",    # failed to score %
            "away_fts_pct",
            "home_bts_pct",    # both teams scored %
            "away_bts_pct",

            # H2H
            "h2h_home_avg_gf",
            "h2h_home_avg_ga",
            "h2h_n_matches",

            # Home/Away specific
            "home_home_gf",    # home team's goals scored at home
            "home_home_ga",
            "away_away_gf",
            "away_away_ga",
        ]
        self.trained = False
        self._try_load()

    def _try_load(self):
        """Try to load a pre-trained model."""
        if XGB_AVAILABLE and MODEL_PATH.exists():
            try:
                self.model = XGBClassifier()
                self.model.load_model(str(MODEL_PATH))
                self.trained = True
                print(f"✅ XGBoost model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"⚠️  Could not load XGB model: {e}")

    def build_features(self, match_data: dict) -> np.ndarray:
        """
        Build feature vector from match prediction data.

        match_data keys:
            dc_p_home, dc_p_draw, dc_p_away, xg_home, xg_away,
            elo_home, elo_away, home_stats, away_stats, h2h
        """
        hs = match_data.get("home_stats", {})
        as_ = match_data.get("away_stats", {})
        h2h = match_data.get("h2h", {})
        form_h = match_data.get("home_form", {})
        form_a = match_data.get("away_form", {})

        xg_h = match_data.get("xg_home", 1.2)
        xg_a = match_data.get("xg_away", 1.0)

        features = [
            match_data.get("dc_p_home", 0.45),
            match_data.get("dc_p_draw", 0.28),
            match_data.get("dc_p_away", 0.27),
            xg_h,
            xg_a,
            xg_h - xg_a,

            match_data.get("elo_home", 1500),
            match_data.get("elo_away", 1500),
            match_data.get("elo_home", 1500) - match_data.get("elo_away", 1500),

            form_h.get("avg_gf", xg_h),
            form_h.get("avg_ga", xg_a),
            form_a.get("avg_gf", xg_a),
            form_a.get("avg_ga", xg_h),
            form_h.get("points", 6.0),
            form_a.get("points", 6.0),

            hs.get("gf", xg_h),
            hs.get("ga", xg_a),
            as_.get("gf", xg_a),
            as_.get("ga", xg_h),
            hs.get("cs",  25) / 100,
            as_.get("cs",  25) / 100,
            hs.get("fts", 20) / 100,
            as_.get("fts", 20) / 100,
            hs.get("bts", 45) / 100,
            as_.get("bts", 45) / 100,

            h2h.get("home_avg_gf", xg_h),
            h2h.get("home_avg_ga", xg_a),
            h2h.get("n_matches", 0),

            hs.get("hgf", hs.get("gf", xg_h)),
            hs.get("hga", hs.get("ga", xg_a)),
            as_.get("agf", as_.get("gf", xg_a)),
            as_.get("aga", as_.get("ga", xg_h)),
        ]

        return np.array(features, dtype=np.float32).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray) -> "MLCalibrator":
        """
        Train the calibrator.
        X: feature matrix (n_samples, n_features)
        y: outcome labels (0=home win, 1=draw, 2=away win)
        """
        if not XGB_AVAILABLE:
            print("⚠️  XGBoost not available. Skipping training.")
            return self

        if len(X) < XGB_MIN_MATCHES:
            print(f"⚠️  Only {len(X)} samples. Need at least {XGB_MIN_MATCHES} to train.")
            return self

        self.model = XGBClassifier(
            n_estimators   = XGB_N_ESTIMATORS,
            max_depth      = XGB_MAX_DEPTH,
            learning_rate  = XGB_LEARNING_RATE,
            subsample      = XGB_SUBSAMPLE,
            colsample_bytree = 0.8,
            eval_metric    = "mlogloss",
            use_label_encoder = False,
            random_state   = 42,
        )
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False)
        self.trained = True

        # Save model
        MODEL_PATH.parent.mkdir(exist_ok=True)
        self.model.save_model(str(MODEL_PATH))
        print(f"✅ XGBoost model trained and saved to {MODEL_PATH}")
        return self

    def calibrate(self, match_data: dict) -> Optional[dict]:
        """
        Return calibrated probabilities.
        Falls back to Dixon-Coles if model not trained.
        """
        if not self.trained or self.model is None:
            return None  # Caller will use DC probabilities directly

        try:
            X = self.build_features(match_data)
            proba = self.model.predict_proba(X)[0]
            # Classes: 0=home, 1=draw, 2=away (based on training)
            return {
                "p_home_win": round(float(proba[0]), 4),
                "p_draw":     round(float(proba[1]), 4),
                "p_away_win": round(float(proba[2]), 4),
                "calibrated": True,
            }
        except Exception as e:
            print(f"⚠️  Calibration failed: {e}")
            return None

    def feature_importance(self) -> Optional[list]:
        """Return feature importances if model is trained."""
        if not self.trained or self.model is None:
            return None
        imp = self.model.feature_importances_
        pairs = list(zip(self.feature_names, imp.tolist()))
        return sorted(pairs, key=lambda x: x[1], reverse=True)
