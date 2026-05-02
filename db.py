# ============================================================
#  pro-football-ai / data/db.py
#  Database layer — SQLite with full schema
# ============================================================

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from config import DB_PATH


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    c = conn.cursor()

    # Teams
    c.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            league      TEXT,
            stats_slug  TEXT,
            elo         REAL DEFAULT 1500,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Matches (historical results)
    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,
            league       TEXT,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            home_goals   INTEGER,
            away_goals   INTEGER,
            ht_home      INTEGER DEFAULT 0,
            ht_away      INTEGER DEFAULT 0,
            season       TEXT,
            source       TEXT DEFAULT 'manual',
            created_at   TEXT DEFAULT (datetime('now'))
        )
    """)

    # Team stats snapshots (scraped from SoccerSTATS)
    c.execute("""
        CREATE TABLE IF NOT EXISTS team_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name   TEXT NOT NULL,
            league      TEXT,
            season      TEXT,
            gp          INTEGER,
            gf          REAL,
            ga          REAL,
            tg          REAL,
            fts_pct     REAL,
            cs_pct      REAL,
            bts_pct     REAL,
            o15_pct     REAL,
            o25_pct     REAL,
            o35_pct     REAL,
            home_gf     REAL,
            home_ga     REAL,
            away_gf     REAL,
            away_ga     REAL,
            fh_gf       REAL,
            fh_ga       REAL,
            fh_tg       REAL,
            fh_bts_pct  REAL,
            raw_json    TEXT,
            fetched_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Elo history
    c.execute("""
        CREATE TABLE IF NOT EXISTS elo_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id    INTEGER,
            team        TEXT NOT NULL,
            elo_before  REAL,
            elo_after   REAL,
            date        TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(id)
        )
    """)

    # Predictions log
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            home_team       TEXT,
            away_team       TEXT,
            league          TEXT,
            xg_home         REAL,
            xg_away         REAL,
            p_home_win      REAL,
            p_draw          REAL,
            p_away_win      REAL,
            predicted_score TEXT,
            confidence      TEXT,
            model_version   TEXT DEFAULT '2.0',
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")


# ── TEAMS ────────────────────────────────────────────────

def upsert_team(name: str, league: str = None, stats_slug: str = None, elo: float = 1500):
    conn = get_conn()
    conn.execute("""
        INSERT INTO teams (name, league, stats_slug, elo)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            league     = COALESCE(excluded.league, teams.league),
            stats_slug = COALESCE(excluded.stats_slug, teams.stats_slug),
            updated_at = datetime('now')
    """, (name, league, stats_slug, elo))
    conn.commit()
    conn.close()


def update_elo(team: str, elo: float):
    conn = get_conn()
    conn.execute("""
        UPDATE teams SET elo = ?, updated_at = datetime('now')
        WHERE name = ?
    """, (elo, team))
    conn.commit()
    conn.close()


def get_team_elo(team: str) -> float:
    conn = get_conn()
    row = conn.execute("SELECT elo FROM teams WHERE name = ?", (team,)).fetchone()
    conn.close()
    return row["elo"] if row else 1500.0


# ── MATCHES ──────────────────────────────────────────────

def insert_match(date, home, away, hg, ag, ht_h=0, ht_a=0, league=None, season=None, source="manual"):
    conn = get_conn()
    conn.execute("""
        INSERT INTO matches (date, league, home_team, away_team, home_goals, away_goals, ht_home, ht_away, season, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (date, league, home, away, hg, ag, ht_h, ht_a, season, source))
    conn.commit()
    conn.close()


def get_team_matches(team: str, limit: int = 30):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM matches
        WHERE home_team = ? OR away_team = ?
        ORDER BY date DESC
        LIMIT ?
    """, (team, team, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_h2h(home: str, away: str, limit: int = 10):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM matches
        WHERE (home_team = ? AND away_team = ?)
           OR (home_team = ? AND away_team = ?)
        ORDER BY date DESC
        LIMIT ?
    """, (home, away, away, home, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── TEAM STATS ───────────────────────────────────────────

def save_team_stats(stats: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO team_stats
        (team_name, league, season, gp, gf, ga, tg, fts_pct, cs_pct, bts_pct,
         o15_pct, o25_pct, o35_pct, home_gf, home_ga, away_gf, away_ga,
         fh_gf, fh_ga, fh_tg, fh_bts_pct, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        stats.get("name"), stats.get("league"), stats.get("season"),
        stats.get("gp"), stats.get("gf"), stats.get("ga"), stats.get("tg"),
        stats.get("fts"), stats.get("cs"), stats.get("bts"),
        stats.get("o15"), stats.get("o25"), stats.get("o35"),
        stats.get("hgf"), stats.get("hga"),
        stats.get("agf"), stats.get("aga"),
        stats.get("fhgf"), stats.get("fhga"), stats.get("fhtg"), stats.get("fhbts"),
        json.dumps(stats)
    ))
    conn.commit()
    conn.close()


def get_latest_stats(team_name: str):
    conn = get_conn()
    row = conn.execute("""
        SELECT * FROM team_stats
        WHERE team_name = ?
        ORDER BY fetched_at DESC
        LIMIT 1
    """, (team_name,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── PREDICTIONS ──────────────────────────────────────────

def save_prediction(pred: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO predictions
        (home_team, away_team, league, xg_home, xg_away,
         p_home_win, p_draw, p_away_win, predicted_score, confidence)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        pred.get("home_team"), pred.get("away_team"), pred.get("league"),
        pred.get("xg_home"), pred.get("xg_away"),
        pred.get("p_home_win"), pred.get("p_draw"), pred.get("p_away_win"),
        pred.get("predicted_score"), pred.get("confidence")
    ))
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 20):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
