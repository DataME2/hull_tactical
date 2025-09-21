# Hull Tactical — Market Prediction (Kaggle)

> **Goal:** Design and deploy a daily S&P 500 forecasting + position-sizing strategy that maximizes the competition’s **modified Sharpe** metric while respecting **risk** and **runtime** constraints.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" />
  <img src="https://img.shields.io/badge/lightgbm-optional-success" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" />
</p>

---

## 🎯 Competition Objectives
1. **Forecast next-day S&P 500 return** (one prediction per trading day) with **calibrated uncertainty** (mean + std or quantiles).
2. **Maximize the competition score** via **risk-aware position sizing** while keeping **portfolio volatility ≤ 1.2×** the S&P 500’s.
3. **Demonstrate out-of-sample robustness** using live-like **walk‑forward, purged time‑series CV** across bull/bear/sideways regimes.

## 🧮 Scoring & Core Constraints (summarized)
- **Score:** \\\((\\text{Portfolio Return} - \\text{S&P 500 Return}) / \\max(\\text{Portfolio Vol}, 1.2\\times\\text{S&P 500 Vol})\\\)
- **Exposure bounds:** **0%–200%** (long-only leverage allowed for the competition).
- **Volatility cap:** **Portfolio volatility ≤ 120%** of S&P 500 volatility.
- **Risk guardrail:** soft **max drawdown ≈ 20%** with automatic exposure throttling.
- **Turnover & costs:** model **5–10 bps** per trade to avoid alpha erosion.
- **Data hygiene:** **no look‑ahead**; respect `is_scored` windows and use **lagged** forward returns for testing.
- **Runtime:** packaged inference should complete within the evaluation time budget (≤ ~9h), be **deterministic** and **stateful**.

> ℹ️ Always consult the official Kaggle rules/overview for the exact, most current definitions and APIs.

---

## 🗂️ Repository Structure
```
.
├─ notebooks/
│  ├─ 00_data_audit.ipynb
│  ├─ 10_feature_engineering.ipynb
│  └─ 20_model_dev_walkforward.ipynb
├─ src/
│  ├─ data/
│  │  ├─ loader.py          # Calendar alignment, missing-data policy (FFILL/interp)
│  │  └─ leak_checks.py     # Strict lagging and feature availability at t
│  ├─ features/
│  │  ├─ momentum.py        # 1–60d returns, rolling means, RSI, etc.
│  │  ├─ volatility.py      # Realized vol, park vol, ATR-like stats
│  │  └─ macro_sentiment.py # Rates, spreads, macro cycles, news/sentiment
│  ├─ models/
│  │  ├─ baseline.py        # Ridge/ElasticNet or LightGBM baseline
│  │  └─ ensemble.py        # Optional: stacking / regime-switching
│  ├─ sizing/
│  │  ├─ vol_target.py      # Exposure ∝ target_vol / predicted_vol
│  │  └─ drawdown_brake.py  # Exposure throttle on MDD
│  ├─ evaluation/
│  │  ├─ scoring.py         # Competition metric + turnover/costs
│  │  └─ walkforward.py     # Purged walk-forward CV splits
│  ├─ runtime/
│  │  ├─ inference.py       # Deterministic, stateful prediction entrypoint
│  │  └─ state.py           # Save/restore model state across calls
│  └─ utils/
│     ├─ config.py
│     └─ logging.py
├─ configs/
│  ├─ baseline.yaml         # Features, model, CV, sizing knobs
│  └─ prod.yaml             # Frozen parameters for final runs
├─ tests/
│  └─ test_scoring.py
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## ⚡ Quickstart

### 1) Environment
```bash
# create env
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data
Download competition data from Kaggle and place under `./input/` (preserving original filenames).  
This repo assumes a layout like:
```
input/
├─ train.parquet
├─ test.parquet
├─ example_test_files/...
└─ supplemental/...
```

### 3) Baseline training (local)
```bash
python -m src.data.loader --check          # sanity checks + leakage tests
python -m src.models.baseline --config configs/baseline.yaml
python -m src.evaluation.walkforward --config configs/baseline.yaml
```

### 4) Inference (packaged, deterministic)
```bash
python -m src.runtime.inference --config configs/prod.yaml --save-state out/state/
```

---

## 🧠 Modeling Approach (MVP → Pro)
- **Signals:** trend (10–60d), short‑term reversal (1–5d), volatility clustering, macro & rates proxies, and sentiment features.
- **Learners:** start with **Ridge/ElasticNet** or **LightGBM**; optionally add **stacking** or **regime switching** (e.g., HMM/thresholds).
- **Uncertainty:** output mean + predictive std (or quantiles) to drive **vol-targeted sizing**.
- **Sizer:** `position_t = clip(signal_t * target_vol / predicted_vol_t, 0, 2)` with a **small Kelly fraction** & **drawdown brake**.
- **Costs:** include `bps_per_trade` and **turnover penalties** in objective and reports.

## 🧪 Validation
- **Walk‑forward, purged splits** with ~3‑month **gaps**; ≥2‑year warm‑up window.
- **Diagnostics:** rolling Sharpe, hit‑rate, MDD, turnover/cost impact, regime buckets.
- **Leakage protection:** enforce time‑correct lags and feature availability checks at time *t*.

---

## 📈 Reporting
- `reports/` folder stores:
  - Rolling metrics (Sharpe, hit rate, drawdown)
  - Exposure, turnover, and cost curves
  - Regime‑sliced performance tables
- `src/evaluation/scoring.py` mirrors the competition metric for apples‑to‑apples comparisons.

---

## 🔁 Reproducibility & Runtime
- Set a **global seed**; prefer deterministic algorithms and fixed thread counts.
- Package the inference path to run within the evaluation time budget (avoid slow external deps).
- Persist model **state** between calls (`src/runtime/state.py`).

---

## 🗺️ Roadmap
- [ ] Add macro & sentiment enrichments
- [ ] Calibrated quantile models
- [ ] Regime detector + adaptive ensembling
- [ ] Turnover‑aware hyperparameter search
- [ ] Experiment tracking (MLflow/W&B) + artifacts

---

## 🤝 Contributing
PRs are welcome! Please:
1. Open an issue describing the change.
2. Include tests for scoring/splits where relevant.
3. Keep configs reproducible and document new knobs.

---

## 📝 License
This project is licensed under the **MIT License** (see `LICENSE`).

## 🙏 Acknowledgments
- Kaggle — Hull Tactical Market Prediction
- Community discussions on time‑series CV, leakage prevention, and risk‑aware sizing.
