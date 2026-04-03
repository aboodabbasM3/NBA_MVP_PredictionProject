# NBA MVP Predictor

A full end-to-end machine learning pipeline that predicts NBA MVP voting share using historical player and team statistics — covering web scraping, data cleaning, model training, and custom evaluation.

---

## Motivation

I've followed the NBA for years and always wondered whether MVP voting could be predicted from pure statistics. This project is my attempt to answer that — building a system that learns from historical seasons and ranks MVP candidates the way a voter would, using real data and real machine learning rather than gut feeling.

---

## Tech Stack

**Data Collection**
- `BeautifulSoup` (html.parser) — static HTML parsing for MVP tables, team stats, and player stats
- `Selenium` — handles dynamic scraping where JavaScript rendering is required
- Raw HTML organized into three folders: `MVPs/`, `Teams/`, `PlayerStats/`

**Data Processing**
- `pandas` — DataFrames, merging, cleaning, null handling
- `io.StringIO` — adapter required for `pd.read_html` to accept raw HTML strings
- Custom franchise normalization dictionary — maps all historical team abbreviations and name variants to consistent identities (`CHA`, `CHH`, `CHO` → `Charlotte Hornets`, `NJN` → `Brooklyn Nets`, `WSB` → `Washington Bullets`, etc.)

**Modeling**
- `scikit-learn` — `Ridge Regression` and `RandomForestRegressor` (50 estimators)
- Random Forest outperformed Ridge Regression and was selected as the final model
- Rolling time-based backtesting — trained strictly on `Year < N`, tested on `Year == N` to prevent data leakage
- Custom `find_precision` metric — ranking-based evaluation measuring recovery of true top-5 MVP candidates
- Custom `add_ranks` function — computes `Rk`, `PredictedRk`, and `Difference` per player-season for error analysis

**Environment**
- Jupyter Notebook
- Git / GitHub

---

## Notebooks

| Notebook | What it does | Output |
|---|---|---|
| `NBA_Web_Scraping.ipynb` | Parses yearly HTML files across MVPs, Teams, and PlayerStats folders; strips duplicate header rows; converts tables via `pd.read_html(StringIO(...))` | 3 CSV files (mvps.csv, teams.csv, players.csv) |
| `NBA_Cleaning&Preprocessing.ipynb` | Drops nulls, normalizes franchise names, merges all three CSVs on `(Team, Year)` using a left join, validates post-merge integrity | 1 clean unified CSV |
| `NBA_MVP_ML_Final.ipynb` | Builds feature matrix, trains Ridge and Random Forest, runs rolling backtest, ranks predictions, evaluates with custom precision metric | Ranked predictions + precision scores |

---

## Workflow

```
Raw HTML files (MVPs/, Teams/, PlayerStats/)
     ↓
BeautifulSoup → strip bad header rows → pd.read_html(StringIO(...))
     ↓
3 CSV files: mvps.csv · teams.csv · players.csv
     ↓
Drop nulls → normalize franchise names → merge on (Team, Year)
     ↓
1 clean unified CSV
     ↓
Ridge Regression vs Random Forest → Random Forest wins
     ↓
Rolling backtest: train on Year < N, test on Year == N
     ↓
Predictions ranked → find_precision evaluates top-5 candidate recovery
```

---

## Evaluation

Standard regression metrics like RMSE don't reflect the actual goal. The real question is whether the model correctly identifies *who* the MVP candidates are in the right order. `find_precision` addresses this by iterating through predicted rankings, checking hits against the true top-5 by vote share, and averaging cumulative precision across all hits — aggregated into a mean precision score across all backtested seasons.

**The final model achieves ~73–75% precision**, meaning it correctly identifies roughly 3–4 out of the true top-5 MVP candidates in the right ranked order across historical seasons.

---
