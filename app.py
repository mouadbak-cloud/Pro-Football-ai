# ============================================================
#  pro-football-ai / api/app.py
#  Flask REST API — serves predictions to the frontend
# ============================================================

import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

from config import API_HOST, API_PORT, API_DEBUG, LEAGUES
from data.db import init_db, get_recent_predictions
from scraper.soccerstats_scraper import search_teams, get_team_stats
from pipeline.predict import predict_match

app = Flask(__name__, static_folder="../static")

# Initialize DB on startup
init_db()


# ── CORS ─────────────────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    static_dir = Path(__file__).parent.parent / "static"
    if path and (static_dir / path).exists():
        return send_from_directory(str(static_dir), path)
    return send_from_directory(str(static_dir), "index.html")


# ── HEALTH ───────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "2.0", "model": "dixon-coles-pro"})


# ── LEAGUES ──────────────────────────────────────────────

@app.route("/api/leagues")
def get_leagues():
    return jsonify({"leagues": [{"id": k, "name": v} for k, v in LEAGUES.items()]})


# ── TEAM SEARCH ──────────────────────────────────────────

@app.route("/api/search")
def search():
    league = request.args.get("league", "")
    query  = request.args.get("q",      "")

    if not league:
        return jsonify({"error": "league parameter required"}), 400

    try:
        teams = search_teams(league, query)
        if teams and "error" in teams[0]:
            return jsonify({"error": teams[0]["error"]}), 500
        return jsonify({"results": teams})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── TEAM STATS ───────────────────────────────────────────

@app.route("/api/stats")
def stats():
    league   = request.args.get("league",  "")
    stats_id = request.args.get("team_id", "")

    if not league or not stats_id:
        return jsonify({"error": "league and team_id required"}), 400

    try:
        data = get_team_stats(league, stats_id)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── PREDICT ──────────────────────────────────────────────

@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
    else:
        body = request.args

    home         = body.get("home",         "")
    away         = body.get("away",         "")
    league       = body.get("league",       "england")
    home_id      = body.get("home_id",      None)
    away_id      = body.get("away_id",      None)

    if not home or not away:
        return jsonify({"error": "home and away team names required"}), 400

    try:
        prediction = predict_match(home, away, league, home_id, away_id)
        return jsonify(prediction)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── RECENT PREDICTIONS ───────────────────────────────────

@app.route("/api/recent")
def recent():
    limit = int(request.args.get("limit", 10))
    preds = get_recent_predictions(limit)
    return jsonify({"predictions": preds})


# ── QUICK PREDICT (stats provided directly) ──────────────

@app.route("/api/quick-predict", methods=["POST"])
def quick_predict():
    """
    Predict without scraping — stats provided directly.
    Used when frontend already has stats from SoccerSTATS.
    """
    from models.dixon_coles import DixonColes
    from features.build_features import compute_xg

    body = request.get_json(silent=True) or {}
    home_stats = body.get("home_stats", {})
    away_stats = body.get("away_stats", {})

    if not home_stats or not away_stats:
        return jsonify({"error": "home_stats and away_stats required"}), 400

    try:
        xg = compute_xg(home_stats, away_stats)

        dc = DixonColes()
        lg = max((home_stats.get("gf", 1.4) + away_stats.get("gf", 1.4)) / 2, 0.7)
        dc.set_params_from_stats(home_stats["name"], home_stats.get("hgf", home_stats.get("gf", 1.4)),
                                  home_stats.get("hga", home_stats.get("ga", 1.4)), lg, lg)
        dc.set_params_from_stats(away_stats["name"], away_stats.get("agf", away_stats.get("gf", 1.4)),
                                  away_stats.get("aga", away_stats.get("ga", 1.4)), lg, lg)

        pred = dc.predict(home_stats["name"], away_stats["name"])
        pred["xg_home"] = xg["xg_home"]
        pred["xg_away"] = xg["xg_away"]

        return jsonify(pred)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Pro Football AI — starting server...")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
