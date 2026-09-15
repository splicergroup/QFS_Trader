"""
Opening-Range Liquidity-Sweep Reversal — LONG ONLY
====================================================
A focused Streamlit app that hunts for *bullish* reversals only, across the
whole trading session after the 09:30 ET open.

Setup (long only):
  1. The first 15-min candle must be a liquidity candle (range ≥ 25% of the
     14-day ATR) — this defines the opening-range box.
  2. Price sweeps BELOW the box low (a stop-run / liquidity grab).
  3. A bullish reversal prints there — Hammer, Inverted Hammer, or Bullish
     Engulfing — with above-average volume AND a close back above VWAP.
  4. Entry = break above the reversal candle high · TP = box high · 2:1 R:R.

Run:  streamlit run orb_long_app.py
"""

# ── Must be set before any other matplotlib import ────────────────────────────
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pytz
from datetime import date, timedelta

from qfs_analyzer import (
    run_analysis, plot_chart, backtest,
    RISK_REWARD,
)

# Long-only, full-session configuration
ALLOW_SHORT  = False
SCAN_MINUTES = None   # no time-of-day restriction on the reversal

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ORB Long-Only Sweep",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stAlert p { font-size: 1.05rem; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #8b949e; }
    [data-testid="stMetricValue"] { font-size: 1.3rem;  font-weight: 700; }
    .sidebar-title {
        font-size: 1.3rem; font-weight: 700;
        color: #3fb950; margin-bottom: 0.25rem;
    }
    .sidebar-sub { font-size: 0.78rem; color: #8b949e; margin-bottom: 1rem; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_trading_days(ticker: str) -> list[date]:
    tz = pytz.timezone("America/New_York")
    try:
        raw = yf.Ticker(ticker).history(period="60d", interval="15m")
    except Exception:
        return []
    if raw.empty:
        return []
    raw.index = raw.index.tz_convert(tz) if raw.index.tz else raw.index.tz_localize(tz)
    return sorted(set(raw.index.date), reverse=True)


@st.cache_data(ttl=60, show_spinner=False)
def cached_analysis(ticker: str, date_str: str,
                    use_volume: bool, use_vwap: bool) -> dict:
    return run_analysis(
        ticker, target_date=date.fromisoformat(date_str),
        allow_short=ALLOW_SHORT, scan_minutes=SCAN_MINUTES,
        use_volume_filter=use_volume, use_vwap_filter=use_vwap,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_backtest(ticker: str, days: int,
                    use_volume: bool, use_vwap: bool) -> dict:
    return backtest(
        ticker, days=days, allow_short=ALLOW_SHORT, scan_minutes=SCAN_MINUTES,
        use_volume_filter=use_volume, use_vwap_filter=use_vwap,
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "sel_ticker"   not in st.session_state:
    st.session_state.sel_ticker   = "MU"
if "sel_date_str" not in st.session_state:
    st.session_state.sel_date_str = None


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-title">🟢 ORB Long-Only Sweep</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Bullish liquidity-sweep reversals · full session</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    ticker_input = st.text_input(
        "Ticker Symbol",
        value=st.session_state.sel_ticker,
        help="US stock or ETF  (e.g. MU, NVDA, AAPL, QQQ, AMD)",
        placeholder="MU",
    ).strip().upper()
    ticker = ticker_input or "MU"

    if ticker != st.session_state.sel_ticker:
        st.session_state.sel_ticker   = ticker
        st.session_state.sel_date_str = None

    with st.spinner(f"Loading trading days for {ticker}…"):
        trading_days = get_trading_days(ticker)

    if not trading_days:
        st.error(f"No market data found for **{ticker}**. Check the ticker and try again.")
        st.stop()

    if st.button("📅  Jump to Latest Day", use_container_width=True, type="primary"):
        st.session_state.sel_date_str = None

    date_strs = [str(d) for d in trading_days]
    sel_idx = date_strs.index(st.session_state.sel_date_str) \
        if st.session_state.sel_date_str in date_strs else 0

    # ── Prev / Next day buttons (trading_days is newest-first) ─────────────────
    nav_prev, nav_next = st.columns(2)
    if nav_prev.button("◀ Prev Day", use_container_width=True,
                       disabled=sel_idx >= len(date_strs) - 1):
        sel_idx = min(sel_idx + 1, len(date_strs) - 1)
        st.session_state.sel_date_str = date_strs[sel_idx]
    if nav_next.button("Next Day ▶", use_container_width=True,
                       disabled=sel_idx <= 0):
        sel_idx = max(sel_idx - 1, 0)
        st.session_state.sel_date_str = date_strs[sel_idx]

    selected_date: date = st.selectbox(
        "Trading Day",
        options=trading_days,
        index=sel_idx,
        format_func=lambda d: d.strftime("%A,  %b %d  %Y"),
        help="Up to 60 calendar days of intraday data are available",
    )
    st.session_state.sel_date_str = str(selected_date)

    st.markdown("---")
    st.markdown("**Confirmation filters**")
    use_volume = st.checkbox(
        "Require above-average volume", value=False,
        help="Reversal candle volume must beat its rolling average.",
    )
    use_vwap = st.checkbox(
        "Require VWAP reclaim", value=False,
        help="Reversal candle must close back above VWAP.",
    )
    st.caption(
        "Filters are off by default for more signals. Turn them on for "
        "higher-quality but fewer setups."
    )

    st.markdown("---")
    st.markdown("**How it works  ·  LONG ONLY**")
    st.markdown("""
1. **ATR gate** — first 15-min candle range ≥ 25% of 14-day ATR (the box)
2. **Sweep** — price dips **below** the box low (liquidity grab)
3. **Reversal** — Hammer / Inv. Hammer / Bull. Engulfing prints there
4. **Filters** — above-average volume **and** close back above VWAP
5. **Entry** — break above the reversal candle high
6. **TP / SL** — box high · 2:1 R:R
""")
    st.markdown("---")
    st.caption("Data via Yahoo Finance · Long-only · Not financial advice")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"## 🟢 `{ticker}` &nbsp;·&nbsp; {selected_date.strftime('%A, %B %d %Y')}",
    unsafe_allow_html=True,
)
st.caption("Opening-range liquidity-sweep reversal — bullish entries only, full session.")

with st.spinner("Scanning the opening range…"):
    r = cached_analysis(ticker, str(selected_date), use_volume, use_vwap)

# ── Signal banner ─────────────────────────────────────────────────────────────
if r["verdict"] == "BUY":
    st.success(
        f"✅  **BUY SIGNAL (LONG)** — "
        f"Pattern: **{r['signal']['pattern']}** at **{r['signal']['time'].strftime('%I:%M %p ET')}**"
    )
else:
    st.error(f"🚫  **NO LONG SETUP** — {r['reason']}")

# ── Chart ─────────────────────────────────────────────────────────────────────
with st.spinner("Rendering chart…"):
    fig = plot_chart(r)

if fig is not None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    st.info("No intraday chart data available for this date.")

# ── Trade-level metrics ───────────────────────────────────────────────────────
if r["verdict"] == "BUY":
    st.markdown("### Trade Levels  ·  LONG")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Entry (BUY)",  f"${r['entry']:.2f}")
    m2.metric("Take Profit",  f"${r['take_profit']:.2f}",
              delta=f"+${r['reward']:.2f}", delta_color="normal")
    m3.metric("Stop Loss",    f"${r['stop_loss']:.2f}",
              delta=f"-${r['risk']:.2f}",   delta_color="inverse")
    m4.metric("Reward",       f"${r['reward']:.2f}")
    m5.metric("Risk",         f"${r['risk']:.2f}")
    m6.metric("R:R Ratio",    f"1 : {RISK_REWARD:.0f}")

# ── Analysis detail ───────────────────────────────────────────────────────────
with st.expander("📊 Full Analysis Details", expanded=False):
    left, right = st.columns(2)

    with left:
        st.markdown("#### Volatility")
        if r["latest_atr"]:
            st.markdown(
                f"| Metric | Value |\n|---|---|\n"
                f"| ATR (14-period daily) | **${r['latest_atr']:.4f}** |\n"
                f"| Liquidity threshold (25%) | **${r['manip_threshold']:.4f}** |"
            )

        st.markdown("#### First 15-min Candle  *(09:30 ET)*")
        if r["first_15"] is not None:
            f15 = r["first_15"]
            direction = "🟢 BULLISH" if f15["Close"] >= f15["Open"] else "🔴 BEARISH"
            st.markdown(
                f"| Field | Value |\n|---|---|\n"
                f"| Time | **{r['first_15_time'].strftime('%I:%M %p %Z')}** |\n"
                f"| Open | **${f15['Open']:.4f}** |\n"
                f"| High | **${f15['High']:.4f}** |\n"
                f"| Low  | **${f15['Low']:.4f}** |\n"
                f"| Close | **${f15['Close']:.4f}** |\n"
                f"| Range | **${r['candle_range']:.4f}** |\n"
                f"| Direction | {direction} |"
            )
        else:
            st.warning("No 09:30 candle data.")

    with right:
        st.markdown("#### Liquidity Check")
        if r["candle_range"] is not None:
            liq_icon = "✅ Confirmed" if r["is_liq"] else "❌ Not Met"
            st.markdown(
                f"| Check | Value |\n|---|---|\n"
                f"| Status | **{liq_icon}** |\n"
                f"| Candle range | **${r['candle_range']:.4f}** |\n"
                f"| Threshold | **${r['manip_threshold']:.4f}** |"
            )

        if r["box_high"]:
            st.markdown("#### Opening-Range Box")
            st.markdown(
                f"| Level | Price |\n|---|---|\n"
                f"| Box High | **${r['box_high']:.4f}** |\n"
                f"| Box Low  | **${r['box_low']:.4f}** |\n"
                f"| Box Size | **${r['box_high'] - r['box_low']:.4f}** |"
            )

        if r["signal"]:
            sig = r["signal"]
            sc  = sig["candle"]
            st.markdown("#### Bullish Reversal")
            st.markdown(
                f"| Field | Value |\n|---|---|\n"
                f"| Pattern | **{sig['pattern']}** |\n"
                f"| Time | **{sig['time'].strftime('%I:%M %p %Z')}** |\n"
                f"| Candle Open  | **${sc['Open']:.4f}** |\n"
                f"| Candle High  | **${sc['High']:.4f}** |\n"
                f"| Candle Low   | **${sc['Low']:.4f}** |\n"
                f"| Candle Close | **${sc['Close']:.4f}** |\n"
                f"| Candle Volume | **{int(sc['Volume']):,}** |\n"
                f"| Swept | below Box Low **${r['box_low']:.4f}** |"
            )

# ── Backtest panel ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🧪 Backtest (long-only, full session)", expanded=False):
    st.caption(
        "Replays this long-only setup day-by-day and simulates each trade "
        "(entry fill → first of TP / SL). Yahoo caps intraday history at "
        "~60 days, so samples are small. Candle-straddle ties assume the stop "
        "fills first (conservative)."
    )
    bt_days = st.slider("Look-back (sessions)", min_value=10, max_value=60,
                        value=60, step=5)
    if st.button(f"▶  Run backtest for {ticker}", use_container_width=True):
        with st.spinner(f"Backtesting {ticker} over ~{bt_days} sessions…"):
            bt = cached_backtest(ticker, bt_days, use_volume, use_vwap)

        if bt.get("error"):
            st.error(f"Backtest failed: {bt['error']}")
        elif bt["trades"] == 0:
            st.info(f"No filled long trades for {ticker} in the window "
                    f"({bt['signals']} signal(s) fired).")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Trades",     bt["trades"])
            c2.metric("Win rate",   f"{bt['win_rate']*100:.1f}%")
            c3.metric("Expectancy", f"{bt['expectancy_r']:+.2f} R")
            c4.metric("Total",      f"{bt['total_r']:+.1f} R")
            c5.metric("Net P/L",    f"${bt['total_pnl']:+.2f}")

            trades_df = pd.DataFrame(bt["trades_list"]).rename(columns={
                "date": "Date", "direction": "Side", "pattern": "Pattern",
                "entry": "Entry", "tp": "TP", "sl": "SL",
                "outcome": "Outcome", "pnl": "P/L", "r_multiple": "R",
            })
            st.dataframe(
                trades_df.style.format({
                    "Entry": "${:.2f}", "TP": "${:.2f}", "SL": "${:.2f}",
                    "P/L": "${:+.2f}", "R": "{:+.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "⚠️ Past performance is not predictive. Small samples have wide "
                "error bars — treat these numbers as directional, not a guarantee."
            )
