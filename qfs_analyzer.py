#!/usr/bin/env python3
"""
Quick Flip Scalper (QFS) Strategy Analyzer
Performs analysis AND renders results on an interactive matplotlib chart.

Strategy:
  1. Calculate the 14-period daily ATR; derive a 25% manipulation threshold.
  2. Check the first 15-min candle of the day (09:30 ET). If its range >=
     threshold it is flagged as a liquidity/manipulation candle.
  3. Box that candle's high and low.
  4. Scan subsequent 5-min candles for a reversal:
       LONG  — a bullish pattern (Hammer, Inverted Hammer, Bullish Engulfing)
               whose low sweeps below the box low.
       SHORT — a bearish pattern (Shooting Star, Hanging Man, Bearish
               Engulfing) whose high sweeps above the box high.
  5. Two confirmation filters must pass on the reversal candle:
       - Volume   : volume >= rolling average (genuine participation).
       - VWAP     : long → close above VWAP; short → close below VWAP.
  6. LONG  : entry = break above reversal high, TP = box high, 2:1 R:R.
     SHORT : entry = break below reversal low,  TP = box low,  2:1 R:R.

Usage:
  python qfs_analyzer.py                       # single-day analysis, defaults to MU
  python qfs_analyzer.py AAPL
  python qfs_analyzer.py --backtest            # backtest MU over the last 60 days
  python qfs_analyzer.py --backtest NVDA --days 60
  python qfs_analyzer.py --backtest --tickers MU,NVDA,AAPL,QQQ,AMD

Dependencies:
  pip install yfinance pandas numpy pytz matplotlib mplfinance
"""

import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date as date_type

# ── Optional-dependency guards ────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    sys.exit("ERROR: yfinance not installed.  Run: pip install yfinance")

try:
    import pytz
except ImportError:
    sys.exit("ERROR: pytz not installed.  Run: pip install pytz")

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines   as mlines
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter
except ImportError:
    sys.exit("ERROR: matplotlib not installed.  Run: pip install matplotlib")

try:
    import mplfinance as mpf
except ImportError:
    sys.exit("ERROR: mplfinance not installed.  Run: pip install mplfinance")


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TICKER     = "MU"
ATR_PERIOD         = 14
ATR_THRESHOLD      = 0.25   # 25% of daily ATR → manipulation threshold
RISK_REWARD        = 2.0    # reward : risk
SHADOW_RATIO       = 2.0    # min shadow-to-body multiple for hammer patterns
VOLUME_MA_PERIOD   = 10     # rolling window for the volume average
VOLUME_MULT        = 1.0    # reversal candle volume must be >= MULT × avg
USE_VWAP_FILTER    = False  # require reversal candle on correct side of VWAP
USE_VOLUME_FILTER  = False  # require reversal candle volume >= average
ALLOW_SHORT        = True   # also trade bearish reversals above the box high
MARKET_TZ          = pytz.timezone("America/New_York")
OPEN_HOUR, OPEN_MIN = 9, 30

# Dark-theme palette
C_BG        = "#0d1117"
C_PANEL     = "#161b22"
C_BORDER    = "#30363d"
C_TEXT      = "#c9d1d9"
C_MUTED     = "#8b949e"
C_UP        = "#26a69a"
C_DOWN      = "#ef5350"
C_BOX_FILL  = "#b8860b"   # dark-gold for box shading
C_BOX_LINE  = "#ffa500"   # orange for box boundary lines
C_PERIOD    = "#1e3a5f"   # dark blue for the first-15-min region
C_ENTRY     = "#58a6ff"   # blue
C_TP        = "#3fb950"   # green
C_SL        = "#f85149"   # red
C_SIGNAL    = "#00ff88"   # bright green for the reversal marker
C_VWAP      = "#c678dd"   # purple for the VWAP line


# ── Low-level helpers ─────────────────────────────────────────────────────────
def _body(c):   return abs(c["Close"] - c["Open"])
def _upper(c):  return c["High"]  - max(c["Close"], c["Open"])
def _lower(c):  return min(c["Close"], c["Open"]) - c["Low"]


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - prev).abs(),
         (df["Low"]  - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def localise(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach intraday VWAP and a rolling volume average (in place)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    df["VWAP"]  = (typical * df["Volume"]).cumsum() / cum_vol
    df["VolMA"] = df["Volume"].rolling(VOLUME_MA_PERIOD, min_periods=1).mean()
    return df


# ── Candlestick pattern detectors ─────────────────────────────────────────────
# Long (bullish) patterns — sought when price sweeps BELOW the box low.
def is_hammer(c) -> bool:
    b = _body(c)
    return b > 1e-8 and _lower(c) >= SHADOW_RATIO * b and _upper(c) <= 0.5 * b


def is_inverted_hammer(c) -> bool:
    b = _body(c)
    return b > 1e-8 and _upper(c) >= SHADOW_RATIO * b and _lower(c) <= 0.5 * b


def is_bullish_engulfing(prev, curr) -> bool:
    return (
        prev["Close"] < prev["Open"]
        and curr["Close"] > curr["Open"]
        and curr["Open"]  <= prev["Close"]
        and curr["Close"] >= prev["Open"]
    )


# Short (bearish) patterns — sought when price sweeps ABOVE the box high.
def is_shooting_star(c) -> bool:
    b = _body(c)
    return b > 1e-8 and _upper(c) >= SHADOW_RATIO * b and _lower(c) <= 0.5 * b


def is_hanging_man(c) -> bool:
    b = _body(c)
    return b > 1e-8 and _lower(c) >= SHADOW_RATIO * b and _upper(c) <= 0.5 * b


def is_bearish_engulfing(prev, curr) -> bool:
    return (
        prev["Close"] > prev["Open"]
        and curr["Close"] < curr["Open"]
        and curr["Open"]  >= prev["Close"]
        and curr["Close"] <= prev["Open"]
    )


def _passes_filters(c, direction: str,
                    use_volume: bool = USE_VOLUME_FILTER,
                    use_vwap: bool = USE_VWAP_FILTER) -> bool:
    """Volume + VWAP confirmation on a candidate reversal candle."""
    if use_volume:
        vol_ma = c.get("VolMA", np.nan)
        if not (pd.notna(vol_ma) and c["Volume"] >= VOLUME_MULT * vol_ma):
            return False
    if use_vwap:
        vwap = c.get("VWAP", np.nan)
        if not pd.notna(vwap):
            return False
        if direction == "long"  and not c["Close"] > vwap:
            return False
        if direction == "short" and not c["Close"] < vwap:
            return False
    return True


# ── Core analysis ─────────────────────────────────────────────────────────────
def run_analysis(ticker: str, target_date: date_type = None,
                 allow_short: bool = None, scan_minutes: int = None,
                 use_volume_filter: bool = None,
                 use_vwap_filter: bool = None) -> dict:
    """
    Run the full QFS strategy for *target_date* (or the latest trading day if
    target_date is None).  Returns a result dict used by print_output(),
    plot_chart() and simulate_trade().

    allow_short       : override the module ALLOW_SHORT default (None → default).
                        Pass False for a long-only variant.
    scan_minutes      : if set, the reversal must form within this many minutes
                        of the 09:30 open (e.g. 90 → only the first 90 minutes).
    use_volume_filter : override USE_VOLUME_FILTER (None → module default).
    use_vwap_filter   : override USE_VWAP_FILTER   (None → module default).
    """
    allow_short = ALLOW_SHORT if allow_short is None else allow_short
    use_volume_filter = USE_VOLUME_FILTER if use_volume_filter is None else use_volume_filter
    use_vwap_filter   = USE_VWAP_FILTER   if use_vwap_filter   is None else use_vwap_filter
    r = dict(
        ticker=ticker,
        trade_date=None,
        latest_atr=None,
        manip_threshold=None,
        first_15=None,
        first_15_time=None,
        candle_range=None,
        is_liq=False,
        box_high=None,
        box_low=None,
        day_5m=None,          # full-day 5-min DataFrame (for chart + sim)
        signal=None,
        direction=None,       # "long" | "short"
        entry=None,
        take_profit=None,
        stop_loss=None,
        reward=None,
        risk=None,
        verdict="NO TRADE",
        reason="",
    )

    def _fetch(interval: str, period: str = None,
               start: str = None, end: str = None) -> pd.DataFrame:
        """Localise a yfinance history call."""
        kw = dict(interval=interval)
        if start:
            kw["start"], kw["end"] = start, end
        else:
            kw["period"] = period
        return localise(yf.Ticker(ticker).history(**kw))

    # ── Daily ATR ─────────────────────────────────────────────────────────────
    if target_date:
        daily = _fetch("1d",
                       start=str(target_date - timedelta(days=120)),
                       end=str(target_date + timedelta(days=1)))
    else:
        daily = _fetch("1d", period="90d")

    if daily.empty:
        r["reason"] = f"No daily data for {ticker}"
        return r

    daily["ATR"] = calculate_atr(daily)
    ref_date     = target_date if target_date else datetime.now(MARKET_TZ).date()
    completed    = daily[daily.index.date < ref_date]
    if completed.empty:
        completed = daily

    r["latest_atr"]      = completed["ATR"].iloc[-1]
    r["manip_threshold"] = r["latest_atr"] * ATR_THRESHOLD

    # ── First 15-min candle ───────────────────────────────────────────────────
    if target_date:
        raw_15m = _fetch("15m",
                         start=str(target_date),
                         end=str(target_date + timedelta(days=1)))
    else:
        raw_15m = _fetch("15m", period="5d")

    if raw_15m.empty:
        r["reason"] = "No 15-min data returned"
        return r

    trade_date      = target_date if target_date else raw_15m.index[-1].date()
    r["trade_date"] = trade_date
    day_15m         = raw_15m[raw_15m.index.date == trade_date]
    open_rows       = day_15m[
        (day_15m.index.hour == OPEN_HOUR) & (day_15m.index.minute == OPEN_MIN)
    ]

    if open_rows.empty:
        r["reason"] = f"No 09:30 candle found for {trade_date}"
        return r

    first_15           = open_rows.iloc[0]
    first_15_time      = open_rows.index[0]
    r["first_15"]      = first_15
    r["first_15_time"] = first_15_time
    r["candle_range"]  = first_15["High"] - first_15["Low"]

    # ── Liquidity check ───────────────────────────────────────────────────────
    r["is_liq"] = r["candle_range"] >= r["manip_threshold"]

    # Always grab 5-min data so the chart renders even on non-signal days
    if target_date:
        raw_5m = _fetch("5m",
                        start=str(target_date),
                        end=str(target_date + timedelta(days=1)))
    else:
        raw_5m = _fetch("5m", period="5d")

    if not raw_5m.empty:
        day_5m = raw_5m[raw_5m.index.date == trade_date].copy()
        if not day_5m.empty:
            add_indicators(day_5m)
        r["day_5m"] = day_5m

    if not r["is_liq"]:
        r["reason"] = (
            f"Candle range ${r['candle_range']:.4f} < "
            f"25% ATR threshold ${r['manip_threshold']:.4f}"
        )
        return r

    # ── Box range ─────────────────────────────────────────────────────────────
    r["box_high"] = first_15["High"]
    r["box_low"]  = first_15["Low"]

    # ── 5-min reversal scan ───────────────────────────────────────────────────
    if r["day_5m"] is None or r["day_5m"].empty:
        r["reason"] = "No 5-min data available for scan"
        return r

    scan_start = first_15_time + pd.Timedelta(minutes=15)
    post_open  = r["day_5m"][r["day_5m"].index >= scan_start]
    if scan_minutes is not None:
        window_end = first_15_time + pd.Timedelta(minutes=scan_minutes)
        post_open  = post_open[post_open.index <= window_end]

    signal = None
    prev   = None
    for ts, c in post_open.iterrows():
        pattern, direction = None, None

        # LONG — sweep below the box low looking for a bullish reversal
        if c["Low"] < r["box_low"]:
            if   is_hammer(c):                                pattern, direction = "Hammer",            "long"
            elif is_inverted_hammer(c):                       pattern, direction = "Inverted Hammer",   "long"
            elif prev is not None and is_bullish_engulfing(prev, c):
                pattern, direction = "Bullish Engulfing", "long"

        # SHORT — sweep above the box high looking for a bearish reversal
        if pattern is None and allow_short and c["High"] > r["box_high"]:
            if   is_shooting_star(c):                         pattern, direction = "Shooting Star",      "short"
            elif is_hanging_man(c):                           pattern, direction = "Hanging Man",        "short"
            elif prev is not None and is_bearish_engulfing(prev, c):
                pattern, direction = "Bearish Engulfing", "short"

        if pattern is not None and _passes_filters(
                c, direction, use_volume=use_volume_filter, use_vwap=use_vwap_filter):
            signal = {"pattern": pattern, "direction": direction,
                      "time": ts, "candle": c}
            break

        prev = c

    r["signal"] = signal

    if signal is None:
        filt = []
        if use_volume_filter: filt.append("volume")
        if use_vwap_filter:   filt.append("VWAP")
        extra = f" passing {'/'.join(filt)} filters" if filt else ""
        win = f" within the first {scan_minutes} min" if scan_minutes else ""
        r["reason"] = (
            f"No qualifying reversal{extra} found{win} "
            f"({len(post_open)} 5-min candles scanned)"
        )
        return r

    # ── Trade levels ──────────────────────────────────────────────────────────
    sc        = signal["candle"]
    direction = signal["direction"]

    if direction == "long":
        entry       = sc["High"]
        take_profit = r["box_high"]
        reward      = take_profit - entry
    else:  # short
        entry       = sc["Low"]
        take_profit = r["box_low"]
        reward      = entry - take_profit

    if reward <= 0:
        r["reason"] = "Reversal candle offers no room to the box (reward ≤ 0)"
        r["signal"] = None
        return r

    risk = reward / RISK_REWARD
    stop_loss = entry - risk if direction == "long" else entry + risk

    r.update(
        direction=direction,
        entry=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
        reward=reward,
        risk=risk,
        verdict="BUY" if direction == "long" else "SELL",
        reason=f"{signal['pattern']} at {signal['time'].strftime('%I:%M %p ET')}",
    )
    return r


# ── Trade simulation & backtest ───────────────────────────────────────────────
def simulate_trade(r: dict) -> dict | None:
    """
    Walk forward through the 5-min candles after the signal and resolve the
    trade.  An entry stop must first be triggered, then the first of TP / SL to
    be touched decides the outcome.  When a single candle straddles both levels
    the stop is assumed to fill first (conservative).  Unresolved by the close
    of the session → marked 'timeout' and settled at the last close.

    Returns None if there is no trade.  Otherwise a dict with:
      filled, outcome ('win'|'loss'|'timeout'), pnl, r_multiple, exit_time.
    """
    if r["verdict"] not in ("BUY", "SELL"):
        return None
    day = r.get("day_5m")
    if day is None or day.empty or r["signal"] is None:
        return {"filled": False, "outcome": None, "pnl": 0.0,
                "r_multiple": 0.0, "exit_time": None}

    direction = r["direction"]
    entry, tp, sl, risk = r["entry"], r["take_profit"], r["stop_loss"], r["risk"]
    fwd = day[day.index > r["signal"]["time"]]

    filled = False
    for ts, c in fwd.iterrows():
        if not filled:
            triggered = (c["High"] >= entry) if direction == "long" else (c["Low"] <= entry)
            if not triggered:
                continue
            filled = True

        if direction == "long":
            hit_sl, hit_tp = c["Low"]  <= sl, c["High"] >= tp
        else:
            hit_sl, hit_tp = c["High"] >= sl, c["Low"]  <= tp

        if hit_sl:   # conservative: stop wins ties
            return {"filled": True, "outcome": "loss", "pnl": -risk,
                    "r_multiple": -1.0, "exit_time": ts}
        if hit_tp:
            return {"filled": True, "outcome": "win", "pnl": r["reward"],
                    "r_multiple": RISK_REWARD, "exit_time": ts}

    if not filled:
        return {"filled": False, "outcome": None, "pnl": 0.0,
                "r_multiple": 0.0, "exit_time": None}

    last = fwd.iloc[-1]
    pnl  = (last["Close"] - entry) if direction == "long" else (entry - last["Close"])
    return {"filled": True, "outcome": "timeout", "pnl": pnl,
            "r_multiple": (pnl / risk if risk else 0.0),
            "exit_time": fwd.index[-1]}


def backtest(ticker: str, days: int = 60,
             allow_short: bool = None, scan_minutes: int = None,
             use_volume_filter: bool = None, use_vwap_filter: bool = None) -> dict:
    """
    Replay the QFS strategy day-by-day over the last *days* sessions and
    aggregate performance.  Intraday history from Yahoo is capped at ~60 days;
    sessions that fall outside the rolling window are skipped silently.

    allow_short / scan_minutes / use_volume_filter / use_vwap_filter are
    forwarded to run_analysis (see its docstring).
    """
    days = min(days, 60)
    summary = dict(ticker=ticker, days=days, signals=0, trades=0,
                   wins=0, losses=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                   expectancy_r=0.0, total_r=0.0, total_pnl=0.0, trades_list=[])

    yf_log = logging.getLogger("yfinance")
    prev_level = yf_log.level
    yf_log.setLevel(logging.CRITICAL)   # Yahoo edge-of-window misses are expected
    try:
        try:
            raw = localise(yf.Ticker(ticker).history(period=f"{days}d", interval="15m"))
        except Exception as e:
            summary["error"] = str(e)
            return summary
        if raw.empty:
            summary["error"] = "No intraday data returned"
            return summary

        for d in sorted(set(raw.index.date)):
            try:
                r = run_analysis(ticker, target_date=d,
                                 allow_short=allow_short, scan_minutes=scan_minutes,
                                 use_volume_filter=use_volume_filter,
                                 use_vwap_filter=use_vwap_filter)
            except Exception:
                continue
            if r["verdict"] not in ("BUY", "SELL"):
                continue
            summary["signals"] += 1
            sim = simulate_trade(r)
            if sim is None or not sim["filled"]:
                continue
            summary["trades_list"].append(dict(
                date=d, direction=r["direction"], pattern=r["signal"]["pattern"],
                entry=r["entry"], tp=r["take_profit"], sl=r["stop_loss"],
                outcome=sim["outcome"], pnl=sim["pnl"], r_multiple=sim["r_multiple"],
            ))
    finally:
        yf_log.setLevel(prev_level)

    trades = summary["trades_list"]
    n = len(trades)
    summary["trades"] = n
    if n:
        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        summary["wins"]         = len(wins)
        summary["losses"]       = len(losses)
        summary["win_rate"]     = len(wins) / n
        summary["avg_win"]      = float(np.mean([t["pnl"] for t in wins]))   if wins   else 0.0
        summary["avg_loss"]     = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
        summary["total_r"]      = float(np.sum([t["r_multiple"] for t in trades]))
        summary["expectancy_r"] = summary["total_r"] / n
        summary["total_pnl"]    = float(np.sum([t["pnl"] for t in trades]))
    return summary


# ── Terminal output ───────────────────────────────────────────────────────────
def print_output(r: dict) -> None:
    W = "═" * 62
    T = "─" * 62
    def H(s): print(f"\n[ {s} ]")

    print(f"\n{W}")
    print(f"  Quick Flip Scalper (QFS)  ·  {r['ticker']}")
    print(f"{W}")

    if r["latest_atr"]:
        H("Step 1  Daily ATR")
        print(f"  14-period ATR            :  ${r['latest_atr']:.4f}")
        print(f"  Manipulation threshold   :  ${r['manip_threshold']:.4f}  (25% ATR)")

    if r["first_15"] is not None:
        f   = r["first_15"]
        dir = "BULLISH" if f["Close"] >= f["Open"] else "BEARISH"
        H("Step 2  First 15-min candle (09:30 ET)")
        print(f"  Date   :  {r['trade_date']}")
        print(f"  Time   :  {r['first_15_time'].strftime('%I:%M %p %Z')}")
        print(f"  OHLC   :  O={f['Open']:.4f}  H={f['High']:.4f}  "
              f"L={f['Low']:.4f}  C={f['Close']:.4f}")
        print(f"  Range  :  ${r['candle_range']:.4f}  ({dir})")

    if r["latest_atr"]:
        H("Step 3  Liquidity candle check")
        v = "CONFIRMED ✓" if r["is_liq"] else "NOT MET ✗"
        print(f"  ${r['candle_range']:.4f} >= ${r['manip_threshold']:.4f}?  {v}")

    if r["box_high"]:
        H("Step 4  Box range")
        print(f"  Box HIGH :  ${r['box_high']:.4f}")
        print(f"  Box LOW  :  ${r['box_low']:.4f}")

    print(f"\n{W}")

    if r["verdict"] in ("BUY", "SELL"):
        sig, sc = r["signal"], r["signal"]["candle"]
        is_long = r["direction"] == "long"
        side    = "LONG (BUY)" if is_long else "SHORT (SELL)"
        print(f"  Direction:  {side}")
        print(f"  Pattern  :  {sig['pattern']}")
        print(f"  Candle   :  {sig['time'].strftime('%I:%M %p %Z')}  "
              f"H={sc['High']:.4f}  L={sc['Low']:.4f}  Vol={int(sc['Volume']):,}")
        print()
        print(f"  RESULT: {r['verdict']} SIGNAL")
        print(T)
        edge_e = "break above reversal high" if is_long else "break below reversal low"
        edge_t = "box high (opposite side)"  if is_long else "box low (opposite side)"
        print(f"  ENTRY  ({r['verdict']})       :  ${r['entry']:.4f}  ← {edge_e}")
        print(f"  TAKE PROFIT         :  ${r['take_profit']:.4f}  ← {edge_t}")
        print(f"  STOP LOSS           :  ${r['stop_loss']:.4f}  ← 2:1 R:R")
        print(T)
        print(f"  Reward ${r['reward']:.4f}  |  "
              f"Risk ${r['risk']:.4f}  |  R:R = 1:{RISK_REWARD:.0f}")
    else:
        print(f"  RESULT: NO TRADE")
        print(f"  Reason : {r['reason']}")

    print(f"{W}\n")


def print_backtest(summary: dict) -> None:
    W = "═" * 62
    T = "─" * 62
    s = summary
    print(f"\n{W}")
    print(f"  QFS Backtest  ·  {s['ticker']}  ·  last {s['days']} sessions")
    print(f"{W}")
    if s.get("error"):
        print(f"  ERROR: {s['error']}")
        print(f"{W}\n")
        return
    print(f"  Signals fired      :  {s['signals']}")
    print(f"  Trades filled      :  {s['trades']}")
    if s["trades"]:
        print(f"  Wins / Losses      :  {s['wins']} / {s['losses']}")
        print(f"  Win rate           :  {s['win_rate']*100:.1f}%")
        print(f"  Avg win  / loss    :  ${s['avg_win']:.4f} / ${s['avg_loss']:.4f}")
        print(f"  Expectancy         :  {s['expectancy_r']:+.3f} R / trade")
        print(f"  Total              :  {s['total_r']:+.2f} R   (${s['total_pnl']:+.4f})")
        print(T)
        print("  Date        Side   Pattern             Outcome     R")
        print(T)
        for t in s["trades_list"]:
            print(f"  {str(t['date']):<11} {t['direction']:<5}  "
                  f"{t['pattern']:<18}  {t['outcome']:<8}  {t['r_multiple']:+.2f}")
    else:
        print("  No filled trades in the window.")
    print(f"{W}\n")


# ── Chart ─────────────────────────────────────────────────────────────────────
def plot_chart(r: dict, scan_window_end=None):
    """
    Render the QFS strategy as a dark-themed candlestick chart.
    Returns the matplotlib Figure (caller is responsible for plt.show / st.pyplot
    and for closing the figure with plt.close).

    scan_window_end : optional timestamp; when given, a dashed vertical line and
                      a faint shaded region mark the signal-scan cut-off
                      (e.g. the first-90-minutes window).
    """
    day_5m = r.get("day_5m")
    if day_5m is None or day_5m.empty:
        print("[chart] No 5-min data to plot.")
        return None

    ticker       = r["ticker"]
    trade_date   = r["trade_date"]
    box_high     = r["box_high"]
    box_low      = r["box_low"]
    entry        = r["entry"]
    take_profit  = r["take_profit"]
    stop_loss    = r["stop_loss"]
    signal       = r["signal"]
    direction    = r.get("direction")
    first_15_t   = r["first_15_time"]
    verdict      = r["verdict"]
    is_long      = direction == "long"

    # ── Build mplfinance addplots ──────────────────────────────────────────────
    addplots = []

    # VWAP line
    if "VWAP" in day_5m.columns and day_5m["VWAP"].notna().any():
        addplots.append(
            mpf.make_addplot(day_5m["VWAP"], color=C_VWAP, width=1.1)
        )

    # Reversal-candle marker (triangle up for long, down for short)
    if signal and signal["time"] in day_5m.index:
        offset = (box_high - box_low) * 0.15 if (box_high and box_low) else 0.05
        marker_ser = pd.Series(np.nan, index=day_5m.index)
        if is_long:
            marker_ser[signal["time"]] = signal["candle"]["Low"] - offset
            mk, mc_color = "^", C_SIGNAL
        else:
            marker_ser[signal["time"]] = signal["candle"]["High"] + offset
            mk, mc_color = "v", C_SL
        addplots.append(
            mpf.make_addplot(
                marker_ser, type="scatter",
                markersize=200, marker=mk, color=mc_color,
            )
        )

    # ── mplfinance dark style ─────────────────────────────────────────────────
    mc = mpf.make_marketcolors(
        up=C_UP, down=C_DOWN,
        edge="inherit", wick="inherit",
        volume={"up": C_UP, "down": C_DOWN},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor=C_PANEL,
        edgecolor=C_BORDER,
        figcolor=C_BG,
        gridstyle="--",
        gridcolor="#21262d",
        gridaxis="both",
        rc={
            "axes.labelcolor": C_TEXT,
            "xtick.color":     C_MUTED,
            "ytick.color":     C_MUTED,
            "text.color":      C_TEXT,
        },
    )

    # ── Plot candlestick base ─────────────────────────────────────────────────
    plot_kw = dict(
        type="candle",
        style=style,
        ylabel="Price ($)",
        volume=False,
        returnfig=True,
        figsize=(15, 7),
        datetime_format="%I:%M %p",   # 12-hour x-axis tick labels
    )
    if addplots:
        plot_kw["addplot"] = addplots

    fig, axes = mpf.plot(day_5m, **plot_kw)
    ax = axes[0]

    n          = len(day_5m)
    price_span = day_5m["High"].max() - day_5m["Low"].min()

    # ── Title ─────────────────────────────────────────────────────────────────
    if verdict == "BUY":
        signal_badge, badge_color = "BUY SIGNAL (LONG)", C_TP
    elif verdict == "SELL":
        signal_badge, badge_color = "SELL SIGNAL (SHORT)", C_SL
    else:
        signal_badge, badge_color = "NO TRADE", C_MUTED
    fig.suptitle(
        f"Quick Flip Scalper  ·  {ticker}  ·  {trade_date}",
        color=C_TEXT, fontsize=13, fontweight="bold",
        x=0.5, y=0.97,
    )
    ax.set_title(
        signal_badge,
        color=badge_color, fontsize=11, fontweight="bold", pad=6,
    )

    # ── Box: high-to-low price rectangle spanning the full chart ─────────────
    if box_high is not None and box_low is not None and first_15_t is not None:
        box_mask = day_5m.index >= first_15_t
        box_idxs = np.where(box_mask)[0]
        x_left   = (box_idxs[0] - 0.5) if len(box_idxs) else -0.5
        x_right  = n - 0.5

        box_fill = mpatches.Rectangle(
            (x_left, box_low),
            width=x_right - x_left,
            height=box_high - box_low,
            linewidth=0,
            facecolor=C_BOX_FILL,
            alpha=0.13,
            zorder=2,
        )
        ax.add_patch(box_fill)

        ax.hlines(box_high, x_left, x_right,
                  colors=C_BOX_LINE, linewidth=2.0, zorder=4)
        ax.hlines(box_low,  x_left, x_right,
                  colors=C_BOX_LINE, linewidth=2.0, zorder=4)
        ax.vlines(x_left, box_low, box_high,
                  colors=C_BOX_LINE, linewidth=2.0, zorder=4)

        _label_right(ax, box_high, f"Box H  ${box_high:.2f}", C_BOX_LINE, style="bold")
        _label_right(ax, box_low,  f"Box L  ${box_low:.2f}",  C_BOX_LINE, style="bold")

    # ── Signal-scan window cut-off ─────────────────────────────────────────────
    if scan_window_end is not None:
        in_window = np.where(day_5m.index <= scan_window_end)[0]
        if len(in_window):
            x_cut = in_window[-1] + 0.5
            ax.axvspan(-0.5, x_cut, color=C_PERIOD, alpha=0.10, zorder=1)
            ax.axvline(x_cut, color=C_MUTED, linestyle="--", linewidth=1.2, zorder=3)
            ax.text(
                x_cut, 0.98, "  scan window end",
                transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5, color=C_MUTED,
            )

    # ── Entry / TP / SL lines + arrow markers ─────────────────────────────────
    if entry is not None:
        ax.axhline(entry,       color=C_ENTRY, linewidth=1.6, zorder=4)
        ax.axhline(take_profit, color=C_TP,    linewidth=1.6, zorder=4)
        ax.axhline(stop_loss,   color=C_SL,    linewidth=1.6, zorder=4)

        # Reward band (entry→TP) green, risk band (entry→SL) red — order-agnostic
        ax.axhspan(entry, take_profit, color=C_TP, alpha=0.06, zorder=2)
        ax.axhspan(stop_loss, entry,   color=C_SL, alpha=0.06, zorder=2)

        # Action arrows pointing at each level (direction-aware wording)
        entry_lbl = "BUY (Entry)"  if is_long else "SELL (Entry)"
        tp_lbl    = "SELL (TP)"    if is_long else "BUY (TP)"
        x_text = min(1 + n * 0.06, n * 0.4)
        for level_price, level_color, level_lbl in (
            (entry,       C_ENTRY, entry_lbl),
            (take_profit, C_TP,    tp_lbl),
            (stop_loss,   C_SL,    "STOP LOSS"),
        ):
            ax.annotate(
                f" {level_lbl}  ${level_price:.2f} ",
                xy=(0, level_price), xytext=(x_text, level_price),
                xycoords="data", textcoords="data",
                ha="left", va="center", fontsize=8, fontweight="bold",
                color="white",
                arrowprops=dict(arrowstyle="-|>", color=level_color,
                                lw=2.2, mutation_scale=16),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=level_color,
                          edgecolor="white", linewidth=0.6, alpha=0.92),
                annotation_clip=False, zorder=10,
            )

    # ── Reversal candle callout annotation ────────────────────────────────────
    if signal and signal["time"] in day_5m.index:
        rev_idx = _get_iloc(day_5m, signal["time"])
        rev_c   = signal["candle"]
        offset  = price_span * 0.06
        if is_long:
            xy_y, txt_y, color = rev_c["Low"] - offset * 0.1, rev_c["Low"] - offset * 1.6, C_SIGNAL
        else:
            xy_y, txt_y, color = rev_c["High"] + offset * 0.1, rev_c["High"] + offset * 1.6, C_SL

        ax.annotate(
            f"  {signal['pattern']}\n  {signal['time'].strftime('%I:%M %p ET')}",
            xy=(rev_idx, xy_y),
            xytext=(min(rev_idx + max(4, n // 12), n - 1), txt_y),
            fontsize=8.5, color=color, fontweight="bold",
            arrowprops=dict(
                arrowstyle="->", color=color,
                lw=1.6, connectionstyle="arc3,rad=-0.2",
            ),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#0a1f0a", edgecolor=color, alpha=0.9,
            ),
            zorder=7,
        )

    # ── Info panel (top-left) ─────────────────────────────────────────────────
    lines = []
    if r["latest_atr"]:
        lines += [
            f"ATR 14d   ${r['latest_atr']:.4f}",
            f"Threshold ${r['manip_threshold']:.4f}  (25%)",
            f"Range     ${r['candle_range']:.4f}",
            f"Liq candle  {'Yes' if r['is_liq'] else 'No'}",
        ]
    if entry is not None:
        lines += [
            "",
            f"Side    {'LONG' if is_long else 'SHORT'}",
            f"Reward  ${r['reward']:.4f}",
            f"Risk    ${r['risk']:.4f}",
            f"R:R     1:{RISK_REWARD:.0f}",
        ]
    if lines:
        ax.text(
            0.012, 0.985, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=7.8, va="top", ha="left", color=C_TEXT,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=C_PANEL,
                      edgecolor=C_BORDER, alpha=0.92),
            zorder=8,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []
    if box_high is not None:
        handles.append(mpatches.Patch(
            facecolor=C_BOX_FILL, edgecolor=C_BOX_LINE,
            alpha=0.5, linewidth=1.5,
            label="15-min Box (H/L)",
        ))
    if "VWAP" in day_5m.columns and day_5m["VWAP"].notna().any():
        handles.append(mlines.Line2D([], [], color=C_VWAP, lw=1.1, label="VWAP"))
    if entry is not None:
        handles += [
            mlines.Line2D([], [], color=C_ENTRY, lw=1.6,
                          label=f"Entry  ${entry:.2f}"),
            mlines.Line2D([], [], color=C_TP,    lw=1.6,
                          label=f"Take Profit  ${take_profit:.2f}"),
            mlines.Line2D([], [], color=C_SL,    lw=1.6,
                          label=f"Stop Loss  ${stop_loss:.2f}"),
        ]
    if signal:
        mk = "^" if is_long else "v"
        col = C_SIGNAL if is_long else C_SL
        handles.append(
            mlines.Line2D([], [], marker=mk, color=col, markersize=9,
                          linestyle="None", label=signal["pattern"])
        )
    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8, framealpha=0.92,
            facecolor=C_PANEL, edgecolor=C_BORDER,
            labelcolor=C_TEXT,
        )

    ax.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
    fig.subplots_adjust(right=0.84, top=0.90, bottom=0.08)
    return fig


# ── Chart helper: right-side price label ──────────────────────────────────────
def _label_right(ax, price: float, text: str, color: str,
                 style: str = "normal") -> None:
    ax.annotate(
        f"  {text}",
        xy=(1.0, price), xycoords=("axes fraction", "data"),
        fontsize=8, color=color, va="center", fontweight=style,
        annotation_clip=False,
    )


def _get_iloc(df: pd.DataFrame, ts) -> int:
    loc = df.index.get_loc(ts)
    if isinstance(loc, slice):
        return loc.start
    if isinstance(loc, np.ndarray):
        return int(np.argmax(loc))
    return int(loc)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick Flip Scalper — liquidity candle detector, chart & backtest"
    )
    parser.add_argument(
        "ticker", nargs="?", default=DEFAULT_TICKER,
        help=f"Ticker symbol (default: {DEFAULT_TICKER})",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run a historical backtest instead of single-day analysis",
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="Backtest look-back in calendar days (max 60, default 60)",
    )
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated tickers for a multi-symbol backtest (e.g. MU,NVDA,AAPL)",
    )
    args = parser.parse_args()

    if args.backtest:
        if args.tickers:
            symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        else:
            symbols = [args.ticker.strip().upper()]
        for sym in symbols:
            print_backtest(backtest(sym, days=args.days))
        return

    ticker = args.ticker.strip().upper()
    result = run_analysis(ticker)
    print_output(result)
    fig = plot_chart(result)
    if fig:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
