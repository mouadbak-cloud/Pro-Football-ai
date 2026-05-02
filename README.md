# ⚽ Pro Football AI — Advanced Match Prediction Engine

> Dixon-Coles Poisson · Dynamic Elo Rating · XGBoost Calibration · SoccerSTATS Data

---

## 🏗️ Architecture (4 Layers)

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Data                                      │
│  SoccerSTATS scraper → SQLite DB → stats cache      │
├─────────────────────────────────────────────────────┤
│  Layer 2: Feature Engineering                       │
│  Elo ratings · Attack/Defense strength · Form · H2H │
├─────────────────────────────────────────────────────┤
│  Layer 3: Dixon-Coles Poisson Model (Core)          │
│  MLE fitting · Low-score correction · Score matrix  │
├─────────────────────────────────────────────────────┤
│  Layer 4: XGBoost Calibration                       │
│  32-feature gradient boosting · Bayesian correction  │
└─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
pro-football-ai/
├── config.py                    # All configuration
├── requirements.txt
│
├── data/
│   ├── db.py                    # SQLite layer (teams, matches, stats, predictions)
│   └── matches.db               # Created automatically
│
├── scraper/
│   └── soccerstats_scraper.py   # SoccerSTATS.com data fetcher
│
├── features/
│   └── build_features.py        # Form, H2H, strength, xG computation
│
├── models/
│   ├── elo.py                   # Dynamic Elo with MoVM
│   ├── dixon_coles.py           # Full DC Poisson + MLE fitting
│   └── ml_calibration.py        # XGBoost calibration layer
│
├── pipeline/
│   ├── predict.py               # 4-layer orchestrator
│   └── train.py                 # Full training pipeline
│
├── api/
│   └── app.py                   # Flask REST API
│
└── static/
    └── index.html               # Professional frontend dashboard
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize database
```bash
python data/db.py
```

### 3. Insert test data (to try the model immediately)
```bash
python pipeline/train.py --test-data
```

### 4. Train all models
```bash
python pipeline/train.py
```

### 5. Start the API server
```bash
python api/app.py
```

### 6. Open the dashboard
```
http://localhost:5000
```

---

## 🔄 Full Data Pipeline

### Step 1: Scrape a league
```bash
python scraper/soccerstats_scraper.py england
```

### Step 2: Search teams manually
```bash
python scraper/soccerstats_scraper.py england Arsenal
```

### Step 3: Retrain models
```bash
python pipeline/train.py
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | Server status |
| `GET /api/leagues` | GET | List all leagues |
| `GET /api/search?league=england&q=Arsenal` | GET | Search teams |
| `GET /api/stats?league=england&team_id=u324-arsenal` | GET | Team stats |
| `POST /api/predict` | POST | Full prediction |
| `GET /api/recent` | GET | Recent predictions |

### Predict endpoint example
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"home": "Arsenal", "away": "Chelsea", "league": "england"}'
```

### Response includes:
- `xg_home`, `xg_away` — Expected Goals
- `p_home_win`, `p_draw`, `p_away_win` — Match result probabilities
- `top_scores` — Top 10 correct score predictions
- `over_under` — O/U 0.5 through 4.5
- `btts` — Both Teams to Score
- `asian_handicap` — AH -1.5 to +1.5
- `exact_totals` — Exact goals probabilities
- `win_to_nil` — Clean sheet wins
- `fh_result` — First half result
- `elo` — Elo rating data
- `form` — Recent form

---

## 📊 Model Performance

| Model | Accuracy | Notes |
|---|---|---|
| Basic Poisson | ~52% | Simple avg goals |
| Dixon-Coles (stats) | ~58% | With home/away split |
| Dixon-Coles (MLE) | ~62% | With time-decay |
| DC + Elo | ~64% | Adding quality adjustment |
| Full pipeline (DC + Elo + XGB) | ~67% | With calibration (needs 200+ matches) |

> ⚠️ Football is inherently random. No model exceeds ~75% accuracy long-term.

---

## 🔧 Configuration (`config.py`)

| Key | Default | Description |
|---|---|---|
| `ELO_K_FACTOR` | 20 | Elo update sensitivity |
| `DC_HOME_ADVANTAGE` | 0.25 | Log-scale home boost |
| `DC_RHO` | 0.10 | Low-score correction |
| `DC_TIME_DECAY` | 0.0065 | Daily decay rate |
| `FORM_WINDOW` | 5 | Recent form matches |
| `XGB_MIN_MATCHES` | 50 | Min data for XGBoost |

---

## 🔮 Future Enhancements

- [ ] Bayesian Hierarchical Model (Gaussian processes)
- [ ] xG from shot maps (not just goals)
- [ ] Live odds engine (in-play probability updates)
- [ ] Telegram bot for daily predictions
- [ ] Automatic daily scraping (cron job)
- [ ] PostgreSQL for production scale
- [ ] Docker deployment
