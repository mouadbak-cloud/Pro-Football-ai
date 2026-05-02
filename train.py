# ============================================================
#  pro-football-ai / pipeline/train.py
#  Training Pipeline — fits all models from historical data
# ============================================================

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import get_conn, init_db, update_elo
from models.elo import EloRating
from models.dixon_coles import DixonColes
from models.ml_calibration import MLCalibrator
from features.build_features import build_training_features


def load_all_matches() -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def train_elo(matches: list) -> EloRating:
    print(f"\n⚙️  Training Elo on {len(matches)} matches...")
    elo = EloRating()
    elo.build_from_matches(matches)

    # Save Elo ratings to DB
    for team, rating in elo.ratings.items():
        update_elo(team, rating)

    top = elo.top_teams(10)
    print("🏆 Top 10 teams by Elo:")
    for i, (team, rating) in enumerate(top, 1):
        print(f"   {i:2}. {team:<30} {rating:.1f}")

    return elo


def train_dixon_coles(matches: list) -> DixonColes:
    if len(matches) < 50:
        print(f"⚠️  Only {len(matches)} matches — Dixon-Coles MLE needs 50+. Using stats-based params.")
        return DixonColes()

    print(f"\n⚙️  Fitting Dixon-Coles MLE on {len(matches)} matches...")
    dc = DixonColes()
    dc.fit(matches)

    print("\n📊 Team Strength Table (top 10):")
    table = dc.strength_table()[:10]
    for row in table:
        print(f"   {row['team']:<30} att={row['attack']:+.3f}  def={row['defense']:+.3f}  net={row['net']:+.3f}")

    # Save model params
    model_path = Path(__file__).parent.parent / "data" / "dc_params.json"
    params = {
        "mu":       dc.mu,
        "home_adv": dc.home_adv,
        "rho":      dc.rho,
        "attack":   dc.attack,
        "defense":  dc.defense,
        "teams":    dc.teams,
        "fitted":   dc.fitted,
        "trained_at": datetime.now().isoformat(),
        "n_matches":  len(matches),
    }
    with open(model_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"✅ DC params saved to {model_path}")

    return dc


def train_xgboost(matches: list) -> MLCalibrator:
    if len(matches) < 50:
        print(f"⚠️  Only {len(matches)} matches — XGBoost needs 50+. Skipping.")
        return MLCalibrator()

    print(f"\n⚙️  Building XGBoost features from {len(matches)} matches...")
    X, y = build_training_features(matches, matches)

    if len(X) < 50:
        print(f"⚠️  Only {len(X)} valid feature rows. Skipping XGBoost training.")
        return MLCalibrator()

    print(f"   Feature matrix: {X.shape}")
    print(f"   Labels: {np.bincount(y)} (home/draw/away)")

    ml = MLCalibrator()
    ml.train(X, y)

    if ml.trained:
        imp = ml.feature_importance()
        if imp:
            print("\n📊 Top 10 XGBoost Feature Importances:")
            for feat, score in imp[:10]:
                bar = "█" * int(score * 50)
                print(f"   {feat:<30} {bar} {score:.4f}")

    return ml


def run_full_pipeline():
    """Run complete training pipeline."""
    print("=" * 60)
    print("  PRO FOOTBALL AI — TRAINING PIPELINE")
    print("=" * 60)

    # Init DB
    init_db()

    # Load matches
    matches = load_all_matches()
    print(f"\n📂 Loaded {len(matches)} matches from database")

    if not matches:
        print("\n⚠️  No matches in database yet!")
        print("   Run the scraper first to collect historical data:")
        print("   python scraper/soccerstats_scraper.py england")
        print("\n   Or insert test data:")
        print("   python pipeline/insert_test_data.py")
        return

    # Train all layers
    elo = train_elo(matches)
    dc  = train_dixon_coles(matches)
    ml  = train_xgboost(matches)

    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Elo:          {len(elo.ratings)} teams rated")
    print(f"  Dixon-Coles:  {'MLE fitted' if dc.fitted else 'Stats-based'}")
    print(f"  XGBoost:      {'Trained ✓' if ml.trained else 'Not enough data yet'}")
    print("\n  🚀 Start the API server:")
    print("     python api/app.py")


def insert_test_data():
    """Insert sample historical matches for testing."""
    from data.db import insert_match, upsert_team

    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Man City",
        "Man United", "Tottenham", "Newcastle", "Aston Villa",
        "West Ham", "Brighton", "Brentford", "Crystal Palace"
    ]

    for t in teams:
        upsert_team(t, "england")

    import random
    random.seed(42)

    matches = []
    for _ in range(200):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        hg   = random.choices([0,1,2,3,4], weights=[10,30,35,18,7])[0]
        ag   = random.choices([0,1,2,3,4], weights=[15,32,32,15,6])[0]
        ht_h = random.choices([0,1,2], weights=[40,45,15])[0]
        ht_a = random.choices([0,1,2], weights=[45,40,15])[0]

        year  = random.choice([2022, 2023, 2024])
        month = random.randint(1, 12)
        day   = random.randint(1, 28)
        date  = f"{year}-{month:02d}-{day:02d}"

        matches.append((date, home, away, hg, ag, ht_h, ht_a, "england", f"{year}/{year+1}"))

    for m in sorted(matches, key=lambda x: x[0]):
        insert_match(*m, source="test")

    print(f"✅ Inserted {len(matches)} test matches")


if __name__ == "__main__":
    if "--test-data" in sys.argv:
        insert_test_data()
    else:
        run_full_pipeline()
