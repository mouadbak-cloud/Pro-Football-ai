# ============================================================
#  pro-football-ai / config.py
#  Central configuration for the entire system
# ============================================================

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Database ─────────────────────────────────────────────
DB_PATH = BASE_DIR / "data" / "matches.db"

# ── Elo ──────────────────────────────────────────────────
ELO_K_FACTOR       = 20      # standard K for league matches
ELO_K_CUP         = 15      # lower K for cup games
ELO_DEFAULT_RATING = 1500
ELO_HOME_ADVANTAGE = 100     # home team gets +100 Elo points boost in expected calc

# ── Dixon-Coles ───────────────────────────────────────────
DC_HOME_ADVANTAGE  = 0.25    # log-scale home advantage parameter
DC_RHO             = 0.10    # low-score correction parameter
DC_MAX_GOALS       = 10      # max goals per team in score matrix
DC_TIME_DECAY      = 0.0065  # decay factor per day (older matches weigh less)

# ── Form Window ───────────────────────────────────────────
FORM_WINDOW        = 5       # last N matches for form calculation
LONG_FORM_WINDOW   = 10      # longer window for attack/defense strength

# ── XGBoost Calibration ───────────────────────────────────
XGB_N_ESTIMATORS   = 300
XGB_MAX_DEPTH      = 4
XGB_LEARNING_RATE  = 0.04
XGB_SUBSAMPLE      = 0.8
XGB_MIN_MATCHES    = 50      # min matches in DB before calibration runs

# ── Scraper ───────────────────────────────────────────────
SCRAPER_DELAY      = 2.0     # seconds between requests (be polite)
SCRAPER_TIMEOUT    = 15
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SOCCERSTATS_BASE   = "https://www.soccerstats.com"

LEAGUES = {
    "england":     "🏴 Premier League",
    "england2":    "🏴 Championship",
    "spain":       "🇪🇸 La Liga",
    "spain2":      "🇪🇸 La Liga 2",
    "germany":     "🇩🇪 Bundesliga",
    "germany2":    "🇩🇪 2. Bundesliga",
    "italy":       "🇮🇹 Serie A",
    "italy2":      "🇮🇹 Serie B",
    "france":      "🇫🇷 Ligue 1",
    "france2":     "🇫🇷 Ligue 2",
    "portugal":    "🇵🇹 Liga Portugal",
    "netherlands": "🇳🇱 Eredivisie",
    "belgium":     "🇧🇪 Pro League",
    "turkey":      "🇹🇷 Süper Lig",
    "scotland":    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Prem",
    "brazil":      "🇧🇷 Série A",
    "argentina":   "🇦🇷 Liga Profesional",
    "mexico":      "🇲🇽 Liga MX",
    "morocco":     "🇲🇦 Botola Pro",
    "egypt":       "🇪🇬 Egyptian Premier",
    "algeria":     "🇩🇿 Ligue Pro 1",
    "cleague":     "🏆 Champions League",
    "uefa":        "🏆 Europa League",
}

# ── Flask API ─────────────────────────────────────────────
API_HOST           = "0.0.0.0"
API_PORT           = 5000
API_DEBUG          = os.getenv("DEBUG", "false").lower() == "true"

# ── Telegram Bot (optional) ───────────────────────────────
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
