# QFS Trader

A set of Python/Streamlit intraday trading dashboards plus an opening-range (ORB)
strategy engine. The main app, **Intraday Analysis**, is a single-ticker intraday
dashboard styled after Robinhood Legend on a dark greyscale theme.

All market data comes from **Yahoo Finance** via `yfinance` — no account or API key
required.

---

## What's inside

| File | What it is |
|------|------------|
| `intraday_analysis_app.py` | **Main app** — intraday dashboard (candles, VWAP, MACD, RSI, EMA×RSI signal, volume, auto-refresh, stock info). |
| `run_intraday_analysis.command` | macOS one-click launcher for the main app (double-click in Finder). |
| `orb_long_app.py` | ORB opening-range sweep app — long-only. |
| `orb_long_short_app.py` | ORB opening-range sweep app — long & short. |
| `qfs_analyzer.py` | Strategy engine (liquidity-candle / ORB detection, chart, backtest). Shared by the apps and runnable as a CLI. |
| `.streamlit/config.toml` | Shared Streamlit theme (used by the ORB apps; the main app applies its own greyscale via CSS). |
| `requirements.txt` | Python dependencies. |

---

## Requirements

- **Python 3.10+** (developed/tested on 3.12)
- An internet connection (data is pulled live from Yahoo Finance)
- macOS, Linux, or Windows

---

## Setup

Clone or download the project, then create a virtual environment and install the
dependencies.

**macOS / Linux**
```bash
cd QFS_trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd QFS_trader
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> The apps expect the virtual environment to live in a folder named `.venv` in the
> project root (the macOS launcher relies on this).

---

## Running

### Main app — Intraday Analysis

With the virtual environment activated:
```bash
streamlit run intraday_analysis_app.py
```
Streamlit prints a local URL (usually <http://localhost:8501>) and opens it in your
browser. Type a ticker in the sidebar to change symbols. Press `Ctrl+C` in the
terminal to stop.

**macOS one-click:** instead of the command above, just **double-click
`run_intraday_analysis.command`** in Finder. It activates `.venv` and starts the app
for you. (First time only, if macOS blocks it, right-click → **Open** once.)

### ORB apps
```bash
streamlit run orb_long_app.py        # long-only
streamlit run orb_long_short_app.py  # long & short
```

### Strategy engine (CLI)
```bash
python qfs_analyzer.py MRVL                          # single-day analysis + chart
python qfs_analyzer.py --backtest MRVL --days 60     # historical backtest
python qfs_analyzer.py --backtest --tickers MU,NVDA,AAPL
python qfs_analyzer.py --help                        # all options
```
Single-day mode opens a matplotlib chart window.

---

## Main app features

**In short:** Intraday Analysis shows one stock's trading day as a single synced chart —
candlesticks with VWAP and volume, MACD, RSI, and a per-candle EMA-crossover Buy/Sell/Hold
signal — alongside a live next-candle countdown, auto-refresh, and a fundamentals panel.

- **One synced chart** — Price & Volume and MACD & RSI sections share a linked x-axis
  (zoom/pan/crosshair move together).
- **Candles** at 5m or 15m, with a **24-hour view** (pre-market / regular / after-hours)
  and shaded off-hours sessions; all times in 12-hour AM/PM.
- **Time window** selector — trailing 6h / 12h or the full day; every pane's y-axis
  fits the visible window.
- **VWAP** anchored to the 9:30 regular-session open.
- **Volume** overlaid on the price pane (or as its own pane, or off).
- **MACD** (12/26/9 EMA) with a 2-colour histogram and vertical crossover markers.
- **RSI** (14) as a zone-coloured line with a 30–70 band and a horizontal zone legend
  that highlights the live zone.
- **EMA×RSI signal ribbon** — per-candle Buy / Sell / Hold from a day-trading EMA(9/21)
  crossover filtered by RSI; square colour intensity encodes confidence, and hovering a
  candle shows the call plus an ATR-based target. Fully tunable via sidebar sliders
  (periods, intensity threshold, EMA-vs-RSI weight, target multiple) with a reset button.
- **Optional ORB box** + entry/target/stop levels from `qfs_analyzer` (off by default).
- **Auto-refresh** at each candle open, with a live countdown next to the title.
- **Stock information** panel (market cap, P/E, dividend yield, 52-week range, etc.).

Every sidebar control has a hover tooltip (ⓘ) describing what it does.

---

## Notes

- **Data source:** Yahoo Finance via `yfinance` is unofficial and rate-limited; expect
  the occasional hiccup. The **overnight session (8 PM – 4 AM ET) is not available** from
  Yahoo, so those bands appear shaded but empty.
- **`yfinance`** occasionally needs updating if Yahoo changes its endpoints
  (`pip install -U yfinance`).
- The signal ribbon and ORB levels are **mechanical technical readouts, not financial
  advice.**

---

## Author

Created by [Ryan Hussain](https://github.com/splicergroup).
