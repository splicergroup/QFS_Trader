"""
Intraday Analysis
=================
A Streamlit + Plotly intraday dashboard for the current trading day, styled
after Robinhood Legend on a greyscale dark theme.  Defaults to MRVL; type any
ticker in the sidebar to switch.  Falls back to the most recent trading day
when today's session has no data yet (weekends, pre-market, holidays).

Layout — one figure, all panes share the x-axis (synced zoom / drag / pan):
  • Price & Volume — candlesticks with hover O/H/L/C, a 09:30-anchored
    regular-session VWAP, the 9:30 open reference line, an optional opening-
    range (ORB) box + trade levels from qfs_analyzer, and a volume sub-pane.
  • MACD & RSI — MACD (12/26/9 EMA) with a 2-colour histogram and vertical
    crossover markers; RSI (14) drawn as a zone-coloured line with a 30–70
    band and a horizontal zone-scale legend that highlights the live zone.

Controls (sidebar):
  • Ticker, candle interval (5m / 15m), time window (Last 6h / 12h / Full day).
  • Toggles for VWAP, volume, MACD, RSI and the ORB box.

Data: 24-hour view (pre-market / regular / after-hours via yfinance prepost),
with the off-hours sessions shaded; all times shown in 12-hour AM/PM.

Run:  streamlit run intraday_analysis_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pytz
from datetime import datetime, timedelta, time as dtime

# Optional: reuse the project's strategy engine for exact ORB levels + signal.
try:
    from qfs_analyzer import run_analysis
    _HAS_ANALYZER = True
except Exception:
    _HAS_ANALYZER = False

# Optional auto-refresh component (feature degrades gracefully if absent).
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False

MARKET_TZ    = pytz.timezone("America/New_York")
DEFAULT_TICKER = "MRVL"
INTERVAL     = "5m"

# Theme colours (mirror .streamlit/config.toml)
# Robinhood Legend candle colours: vivid green up / orange-red down
UP_COLOR   = "#00c805"
DOWN_COLOR = "#ff5000"
# Volume bars use muted shades (green / brick-red) like Legend
VOL_UP     = "#1a9e57"
VOL_DOWN   = "#b54331"
VWAP_COLOR = "#ffa500"
ORB_COLOR  = "#58a6ff"
ENTRY_COLOR = "#e3b341"
# Greyscale dark theme (backgrounds / grid / text — neutral, no colour tint)
GRID_COLOR = "#333333"
BG_COLOR   = "#141414"
TEXT_COLOR = "#d4d4d4"
MUTED_COLOR = "#8a8a8a"
# Session shading (semi-transparent black over the #141414 background)
SHADE_OVERNIGHT = "rgba(0,0,0,0.50)"   # overnight / closed — darkest
SHADE_EXT       = "rgba(0,0,0,0.25)"   # pre-market & after-hours

# RSI styling — the line itself is coloured per-zone (see RSI_ZONES)
RSI_PERIOD = 14
# Colour-coded zones: (low, high, colour, label)
RSI_ZONES = [
    (0,   20,  "#3f6fb2", "Extreme OS"),
    (20,  30,  "#5a90d6", "Oversold"),
    (30,  45,  "#4d7a2e", "Recovery"),
    (45,  55,  "#71a83f", "Neutral"),
    (55,  70,  "#c98a32", "Bullish"),
    (70,  80,  "#9b5a2a", "Overbought"),
    (80, 100,  "#b23a30", "Extreme OB"),
]
# Neutral-zone band + reference lines drawn behind the coloured RSI line
RSI_BAND_FILL = "#1e1e1e"   # dark grey 30–70 shading (just above background)
RSI_BAND_LINE = "#9e9e9e"   # light grey dashed lines at 30 and 70

OPEN_TIME  = dtime(9, 30)
OR_END     = dtime(9, 45)   # first 15 minutes

# MACD parameters (Fast 12 / Slow 26 / Signal 9, exponential)
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
# Robinhood Legend palette
MACD_COLOR   = "#7ecbe3"   # MACD line — pale cyan-blue
SIGNAL_COLOR = "#e8a33d"   # signal line — amber
HIST_POS     = "#00c805"   # histogram above zero — Robinhood green
HIST_NEG     = "#b54331"   # histogram below zero — brick red
XOVER_BULL   = "rgba(0,200,5,0.55)"    # bullish MACD/signal crossover marker
XOVER_BEAR   = "rgba(255,80,0,0.55)"   # bearish MACD/signal crossover marker

# Inference ribbon (EMA 9/21 crossover × RSI): Buy / Sell / Hold square colours.
# Colour INTENSITY (alpha) encodes confidence, so RGB triples are kept too.
SIG_BUY  = "#00c805"   # green
SIG_SELL = "#f23645"   # red
SIG_HOLD = "#eab308"   # yellow
SIG_BUY_RGB, SIG_SELL_RGB, SIG_HOLD_RGB = "0,200,5", "242,54,69", "234,179,8"
SIG_FAST, SIG_SLOW = 9, 21   # day-trading EMA crossover pair
SIG_ATR_THR = 0.75   # EMA spread (in ATRs) for full-intensity confidence
SIG_EMA_W   = 0.6    # confidence weight — EMA spread vs RSI
SIG_TGT_MULT = 1.5   # target distance = N × ATR(14)

# Per-pane target heights (px) — MACD is the tallest indicator pane
PANE_H = {"price": 360, "volume": 140, "macd": 340, "signal": 78, "rsi": 190}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intraday Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Greyscale dark theme — scoped to this app (shared config.toml untouched) */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #141414; color: #d4d4d4;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background-color: #141414; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #111111;
    }
    /* Ticker (text input) field background */
    [data-testid="stSidebar"] div[data-baseweb="input"],
    [data-testid="stSidebar"] div[data-baseweb="base-input"],
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background-color: #222222;
    }
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #8a8a8a; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
    .sidebar-title { font-size: 1.3rem; font-weight: 700; color: #e6e6e6; margin-bottom: 0.25rem; }
    .sidebar-sub   { font-size: 0.78rem; color: #8a8a8a; margin-bottom: 1rem; }
    hr { border-color: #333333; }
</style>
""", unsafe_allow_html=True)


# ── Data helper (cached) ──────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday(ticker: str, interval: str) -> pd.DataFrame:
    """Pull recent intraday bars at *interval* and localise to US/Eastern.

    We grab a few days so weekends/holidays still resolve to the latest
    session and indicator EMAs warm up; the caller slices out a single day.
    prepost=True includes pre-market and after-hours bars (full 24h view).
    """
    raw = yf.Ticker(ticker).history(period="5d", interval=interval, prepost=True)
    if raw.empty:
        return raw
    raw.index = (raw.index.tz_convert(MARKET_TZ) if raw.index.tz
                 else raw.index.tz_localize(MARKET_TZ))
    return raw


def latest_session(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the bars belonging to the most recent trading day present."""
    last_day = df.index.date.max()
    return df[df.index.date == last_day]


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Regular-session VWAP, anchored at the 09:30 open.

    Computed only over the 09:30–16:00 regular session (NaN in pre-market and
    after-hours), so the line begins at the regular open and sits within the
    unshaded session region.
    """
    df = df.copy()
    mask = (df.index.time >= dtime(9, 30)) & (df.index.time < dtime(16, 0))
    reg = df[mask]
    typical = (reg["High"] + reg["Low"] + reg["Close"]) / 3
    cum_vol = reg["Volume"].cumsum()
    vwap = (typical * reg["Volume"]).cumsum() / cum_vol.replace(0, pd.NA)
    df["VWAP"] = float("nan")
    df.loc[mask, "VWAP"] = vwap
    return df


def add_macd(df: pd.DataFrame,
             fast: int = MACD_FAST, slow: int = MACD_SLOW,
             signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """MACD (12/26/9) using exponential moving averages.

    Computed on the full multi-day fetch *before* slicing to one session, so
    the slow EMA is warmed up and the open isn't a cold-start artifact —
    matching how charting platforms carry intraday MACD across the gap.
    `adjust=False` gives the conventional recursive EMA (alpha = 2/(n+1)).
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    return df


def add_signal_emas(df: pd.DataFrame,
                    fast: int = SIG_FAST, slow: int = SIG_SLOW) -> pd.DataFrame:
    """Fast/slow EMAs (9/21) for the day-trading crossover signal.

    Computed on the full multi-day fetch before slicing so the 21-EMA is warm
    at the open (same rationale as MACD/RSI).
    """
    df = df.copy()
    df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Wilder's RSI — the standard 14-period RSI Robinhood Legend plots.

    Computed on the full multi-day fetch before slicing, so the average
    gain/loss are warmed up at the session open (same reason as MACD).
    Wilder smoothing == EMA with alpha = 1/period.
    """
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))   # avg_loss==0 → rs=inf → RSI=100
    return df


def rsi_zone(value: float):
    """Return (colour, label) for the RSI zone containing *value*."""
    for lo, hi, colour, name in RSI_ZONES:
        if lo <= value < hi:
            return colour, name
    return RSI_ZONES[-1][2], RSI_ZONES[-1][3]   # value == 100 → Extreme OB


def rsi_legend_html(value: float) -> str:
    """Horizontal RSI-zone scale (segments sized by range), current zone lit."""
    cur_col, cur_name = rsi_zone(value)
    segs = []
    for lo, hi, col, name in RSI_ZONES:
        active = name == cur_name
        hl = ("outline:3px solid #ffffff;outline-offset:-3px;filter:brightness(1.2);"
              if active else "")
        segs.append(
            f"<div style='flex:{hi - lo} 0 0;background:{col};color:#fff;"
            f"padding:7px 2px;text-align:center;{hl}'>"
            f"<div style='font-weight:700;font-size:0.8rem;'>{name}</div>"
            f"<div style='font-size:0.68rem;opacity:0.85;'>{lo}–{hi}</div></div>"
        )
    ticks = "".join(f"<span>{t}</span>" for t in range(0, 101, 10))
    return (
        "<div style='margin-top:8px;'>"
        "<div style='font-size:0.74rem;color:#8a8a8a;font-weight:700;"
        "letter-spacing:0.06em;margin-bottom:5px;'>RSI ZONE — "
        f"<span style='color:{cur_col};'>{cur_name.upper()} ({value:.1f})</span></div>"
        f"<div style='display:flex;border-radius:8px;overflow:hidden;'>{''.join(segs)}</div>"
        "<div style='display:flex;justify-content:space-between;font-size:0.68rem;"
        f"color:#8a8a8a;margin-top:3px;'>{ticks}</div></div>"
    )


def make_section(panes: list[str], price_secondary_y: bool = False):
    """Build a make_subplots figure sized by PANE_H for the given panes.

    price_secondary_y adds a right-hand y-axis to row 1 (used to overlay volume
    on the price pane).  Returns (fig, row_of, total_px); row_of maps a pane
    name to its 1-based row.
    """
    heights = [PANE_H[p] for p in panes]
    specs = [[{"secondary_y": price_secondary_y if i == 0 else False}]
             for i in range(len(panes))]
    fig = make_subplots(
        rows=len(panes), cols=1, shared_xaxes=True,
        row_heights=[h / sum(heights) for h in heights],
        vertical_spacing=0.05, specs=specs,
    )
    return fig, {name: i + 1 for i, name in enumerate(panes)}, sum(heights)


def style_section(fig, total_px: int):
    """Apply the shared greyscale layout, spikes and intraday rangebreaks."""
    fig.update_layout(
        height=total_px + 46,
        hovermode="x unified",
        showlegend=True,
        # top-right so it doesn't collide with the left-aligned section labels
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="monospace"),
        xaxis_rangeslider_visible=False,
        dragmode="pan",
    )
    # Full 24h day is shown (no rangebreaks); 12-hour AM/PM time labels & hover.
    fig.update_xaxes(
        gridcolor=GRID_COLOR, showspikes=True, spikemode="across",
        spikethickness=1, spikecolor="#8a8a8a",
        tickformat="%I:%M %p", hoverformat="%I:%M %p",
    )


def section_label(fig, row: int, text: str):
    """Bold, left-aligned section header sitting just above a pane.

    Added after the pane has a trace so its axes are realised (annotations
    with row/col only register once the subplot exists).
    """
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.0, y=1.0,
        xanchor="left", yanchor="bottom", yshift=7,
        text=f"<b>{text}</b>", showarrow=False,
        font=dict(size=13, color="#e6e6e6", family="monospace"),
        row=row, col=1,
    )


def view_window(day, view, last_ts, is_today):
    """Return the (lo, hi) x-range to display for *view*.

    'Full day' → 00:00–24:00 ET. 'Last 6h'/'Last 12h' → the trailing window
    ending at the latest data (or now, if the session is live).
    """
    start = MARKET_TZ.localize(datetime.combine(day, dtime(0, 0)))
    end = start + timedelta(days=1)
    if view == "Full day":
        return start, end
    hours = 12 if "12" in view else 6
    right = max(last_ts, datetime.now(MARKET_TZ)) if is_today else last_ts
    right = min(right, end)
    return max(start, right - timedelta(hours=hours)), right


def shade_sessions(fig, day, x_range):
    """Shade pre-market / after-hours / overnight across all panes (regular
    09:30–16:00 left clear) and set the x-range to *x_range*."""
    def et(h, m=0):
        return MARKET_TZ.localize(datetime.combine(day, dtime(h, m)))

    day_start, day_end = et(0), et(0) + timedelta(days=1)
    regions = [
        (day_start, et(4),     SHADE_OVERNIGHT),   # overnight (early)
        (et(4),     et(9, 30), SHADE_EXT),         # pre-market
        (et(16),    et(20),    SHADE_EXT),         # after-hours
        (et(20),    day_end,   SHADE_OVERNIGHT),   # overnight (late)
    ]
    for x0, x1, colour in regions:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=colour, opacity=1.0,
                      line_width=0, layer="below", row="all", col=1)
    fig.update_xaxes(range=list(x_range))


def local_orb_box(df: pd.DataFrame) -> dict | None:
    """First-15-min high/low of the displayed session.

    The 09:30 15-min candle qfs_analyzer uses is exactly the aggregate of the
    first three 5-min bars, so max-high / min-low here matches its box.
    """
    first15 = df[(df.index.time >= OPEN_TIME) & (df.index.time < OR_END)]
    if first15.empty:
        return None
    return {"box_high": float(first15["High"].max()),
            "box_low":  float(first15["Low"].min()),
            "n_bars":   len(first15)}


@st.cache_data(ttl=60, show_spinner=False)
def analyzer_levels(ticker: str, allow_short: bool) -> dict | None:
    """Exact box + liquidity verdict + trade levels from qfs_analyzer.

    Returns None if the analyzer isn't importable or the call fails — the
    caller falls back to local_orb_box().
    """
    if not _HAS_ANALYZER:
        return None
    try:
        r = run_analysis(ticker, allow_short=allow_short)
    except Exception:
        return None
    keys = ("box_high", "box_low", "is_liq", "candle_range", "manip_threshold",
            "verdict", "reason", "trade_date", "direction",
            "entry", "take_profit", "stop_loss", "signal")
    return {k: r.get(k) for k in keys}


@st.cache_data(ttl=300, show_spinner=False)
def get_info(ticker: str) -> dict:
    """Yahoo fundamentals/quote dict for the info panel (empty on failure)."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def _fmt_num(n):
    """Humanise a large number: 1.23B / 45.00M / 12,345."""
    if n in (None, ""):
        return "—"
    n = float(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:,.0f}"


def _fmt_price(n):
    return "—" if n in (None, "") else f"${float(n):,.2f}"


def _fmt2(n):
    return "—" if n in (None, "") else f"{float(n):,.2f}"


def next_candle_open(interval_min: int):
    """(next_open ET datetime, seconds_until) for the next clock-aligned candle."""
    now = datetime.now(MARKET_TZ)
    floor = now.replace(second=0, microsecond=0,
                        minute=(now.minute // interval_min) * interval_min)
    nxt = floor + timedelta(minutes=interval_min)
    return nxt, (nxt - now).total_seconds()


def countdown_html(target_ms: int, step_ms: int, label: str) -> str:
    """A live MM:SS countdown to the next candle open (ticks client-side; loops
    to the following candle on its own so it stays accurate between reruns)."""
    tpl = """
<style>body{margin:0;background:transparent;overflow:hidden;}</style>
<div style="font-family:monospace;color:#d4d4d4;font-size:0.8rem;white-space:nowrap;
            display:flex;align-items:center;justify-content:flex-end;gap:8px;margin-top:10px;">
  <span style="color:#8a8a8a;">⏱ Next __LABEL__ candle</span>
  <span id="cd" style="font-weight:700;font-size:1.05rem;color:#e6e6e6;">--:--</span>
</div>
<script>
let target = __TARGET__; const step = __STEP__;
function fmt(ms){let s=Math.max(0,Math.floor(ms/1000));
  return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');}
function tick(){
  let rem = target - Date.now();
  if (rem <= 0){ while (target <= Date.now()) target += step; rem = target - Date.now(); }
  var el = document.getElementById('cd'); if (el) el.textContent = fmt(rem);
}
tick(); setInterval(tick, 1000);
</script>
"""
    return (tpl.replace("__TARGET__", str(target_ms))
               .replace("__STEP__", str(step_ms))
               .replace("__LABEL__", label))


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-title">📈 Intraday Analysis</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">Current trading day · intraday bars</div>',
                    unsafe_allow_html=True)

ticker = st.sidebar.text_input(
    "Ticker", value=DEFAULT_TICKER,
    help="Stock symbol to chart (e.g. MRVL, NVDA, AAPL) — any Yahoo Finance "
         "ticker.").strip().upper()
interval = st.sidebar.radio(
    "Candle interval", ["5m", "15m"], horizontal=True, index=0,
    help="Size of each candle — 5- or 15-minute bars.")
time_view = st.sidebar.radio(
    "Time window", ["Last 6h", "Last 12h", "Full day"], index=0,
    help="How much of the day to show: a trailing 6h / 12h window, or the full "
         "24-hour session.")
show_vwap = st.sidebar.checkbox(
    "Show VWAP", value=True,
    help="Volume-Weighted Average Price, anchored to the 9:30 regular-session "
         "open (drawn only during regular hours).")
vol_mode = st.sidebar.radio("Volume", ["Overlay", "Separate pane", "Off"],
                            index=0, horizontal=True,
                            help="Overlay: translucent bars at the bottom of the "
                                 "price pane.")
show_volume = vol_mode != "Off"
volume_overlay = vol_mode == "Overlay"
show_macd = st.sidebar.checkbox(
    "Show MACD (12/26/9 EMA)", value=True,
    help="Moving Average Convergence Divergence — histogram, signal line, and "
         "vertical crossover markers.")
show_signal = st.sidebar.checkbox(
    "Show signal (EMA×RSI)", value=True,
    help="Per-candle Buy / Sell / Hold inference from the EMA(9/21) crossover "
         "filtered by RSI; square colour intensity = confidence. Hover a square "
         "for the call and an ATR-based target.")
show_rsi = st.sidebar.checkbox(
    "Show RSI (14)", value=True,
    help="Relative Strength Index (14) as a zone-coloured line with a 30–70 "
         "band and the zone-scale legend below the chart.")
show_orb = st.sidebar.checkbox("Show ORB box", value=False,
                               help="First 15-min opening range — the box "
                                    "qfs_analyzer trades around.")
allow_short = st.sidebar.checkbox("Scan shorts too", value=True, disabled=not show_orb,
                                  help="Include bearish reversal signals in the "
                                       "qfs_analyzer scan.")

# Tunables for the EMA-crossover signal pane (defaults when the pane is hidden)
SIG_DEFAULTS = {"sig_fast": SIG_FAST, "sig_slow": SIG_SLOW,
                "sig_atr_thr": SIG_ATR_THR, "sig_ema_w": SIG_EMA_W,
                "sig_tgt_mult": SIG_TGT_MULT}
if show_signal:
    for _k, _v in SIG_DEFAULTS.items():
        st.session_state.setdefault(_k, _v)   # seed defaults on first run
    with st.sidebar.expander("⚙️ Signal settings", expanded=False):
        # Reset runs before the sliders, so they pick up the restored values
        if st.button("↺ Reset to defaults", use_container_width=True,
                     help="Restore all signal sliders to their default values."):
            for _k, _v in SIG_DEFAULTS.items():
                st.session_state[_k] = _v
        sig_fast = st.slider("Fast EMA", 3, 20, key="sig_fast",
                             help="Fast EMA period for the crossover — shorter "
                                  "reacts quicker (more signals).")
        sig_slow = st.slider("Slow EMA", 10, 60, key="sig_slow",
                             help="Slow EMA period — longer means a smoother, "
                                  "slower-turning trend.")
        if sig_fast >= sig_slow:
            st.caption("⚠️ Fast EMA should be below Slow EMA.")
        sig_atr_thr = st.slider("Trend strength — ATR spread for full intensity",
                                0.25, 2.0, step=0.05, key="sig_atr_thr",
                                help="How wide the EMA gap (in ATRs) must be for a "
                                     "square to reach full colour. Lower = brighter "
                                     "sooner.")
        sig_ema_w = st.slider("Confidence weight — EMA vs RSI", 0.0, 1.0, step=0.05,
                              key="sig_ema_w",
                              help="1.0 = all from EMA spread; 0.0 = all from RSI.")
        sig_tgt_mult = st.slider("Target distance — ATR multiple", 0.5, 4.0, step=0.1,
                                 key="sig_tgt_mult",
                                 help="Target price distance from the close, in "
                                      "ATR(14) multiples.")
else:
    sig_fast, sig_slow = SIG_DEFAULTS["sig_fast"], SIG_DEFAULTS["sig_slow"]
    sig_atr_thr, sig_ema_w = SIG_DEFAULTS["sig_atr_thr"], SIG_DEFAULTS["sig_ema_w"]
    sig_tgt_mult = SIG_DEFAULTS["sig_tgt_mult"]

if st.sidebar.button("🔄 Refresh data",
                     help="Clear cached data and re-pull the latest quotes now."):
    fetch_intraday.clear()
    analyzer_levels.clear()
    get_info.clear()

# Auto-refresh (reruns at each candle open) + a live countdown to that open
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh", value=True,
    help="Reload at each new candle open — every 5 min on 5m, 15 min on 15m.")
_mins = 5 if interval == "5m" else 15
_next_open, _secs_left = next_candle_open(_mins)
if auto_refresh and _HAS_AUTOREFRESH:
    st_autorefresh(interval=max(1000, int(_secs_left * 1000)), key="auto_refresh")
elif auto_refresh and not _HAS_AUTOREFRESH:
    st.sidebar.caption("⚠️ `pip install streamlit-autorefresh` to enable")

if not ticker:
    st.warning("Enter a ticker symbol in the sidebar.")
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} {interval} data…"):
    raw = fetch_intraday(ticker, interval)

if raw.empty:
    st.error(f"No data returned for **{ticker}**. Check the symbol and try again.")
    st.stop()

# Indicators computed on the full multi-day fetch (warmed-up), then sliced.
raw = add_signal_emas(add_rsi(add_macd(raw)), fast=sig_fast, slow=sig_slow)
df = add_vwap(latest_session(raw))
session_date = df.index[0].date()
is_today = session_date == datetime.now(MARKET_TZ).date()

# ── Resolve ORB levels ────────────────────────────────────────────────────────
# Prefer qfs_analyzer's exact numbers when they apply to the displayed session;
# otherwise fall back to the local first-15-min high/low so the box still draws.
box_high = box_low = None
ana = None
if show_orb:
    local = local_orb_box(df)
    if local:
        box_high, box_low = local["box_high"], local["box_low"]
    ana = analyzer_levels(ticker, allow_short)
    ana_matches = bool(ana and ana.get("trade_date") == session_date)
    if ana_matches and ana.get("box_high") is not None:
        box_high, box_low = ana["box_high"], ana["box_low"]
    if not ana_matches:
        ana = None  # verdict/signal belong to a different day — ignore

# ── Header + metrics ──────────────────────────────────────────────────────────
day_label = "Today" if is_today else session_date.strftime("%a %b %d, %Y")
_title_col, _cd_col = st.columns([3, 1])
_title_col.markdown(f"### {ticker} · {session_date:%b %d, %Y}  "
                    f"<span style='color:#8a8a8a;font-size:0.9rem'>({day_label})</span>",
                    unsafe_allow_html=True)
with _cd_col:
    components.html(
        countdown_html(int(_next_open.timestamp() * 1000), _mins * 60_000, interval),
        height=46,
    )
if not is_today:
    st.caption("⚠️ Today's session has no data yet — showing the most recent "
               "trading day.")

# "Open" is anchored to the 09:30 regular open (not the 4:00 pre-market print);
# high/low/last/volume still reflect the full 24h shown.
_reg = df[(df.index.time >= dtime(9, 30)) & (df.index.time < dtime(16, 0))]
o = float((_reg if not _reg.empty else df)["Open"].iloc[0])
h, l, c = df["High"].max(), df["Low"].min(), df["Close"].iloc[-1]
chg = c - o
chg_pct = chg / o * 100 if o else 0
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Open",  f"${o:,.2f}")
m2.metric("High",  f"${h:,.2f}")
m3.metric("Low",   f"${l:,.2f}")
m4.metric("Last",  f"${c:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
m5.metric("Volume", f"{df['Volume'].sum():,.0f}")

# ── ORB levels (informational) ────────────────────────────────────────────────
if show_orb and box_high is not None:
    box_rng = box_high - box_low
    if ana is not None:
        liq = "✅ liquidity candle" if ana.get("is_liq") else "⚪ not a liquidity candle"
        thr, rng = ana.get("manip_threshold"), ana.get("candle_range")
        thr_txt = (f" · OR range ${rng:,.2f} vs 25%-ATR threshold ${thr:,.2f}"
                   if thr is not None and rng is not None else "")
        st.caption(f"ORB: ${box_low:,.2f} – ${box_high:,.2f} "
                   f"(range ${box_rng:,.2f}) · {liq}{thr_txt}")
    else:
        st.caption(f"ORB (first 15 min): ${box_low:,.2f} – ${box_high:,.2f} "
                   f"(range ${box_rng:,.2f}) · qfs_analyzer levels unavailable")

# ── Chart — one figure; all panes share the x-axis so zoom/drag stay in sync ──
panes = ["price"]
if show_volume and not volume_overlay:
    panes.append("volume")
if show_macd:
    panes.append("macd")
if show_signal:
    panes.append("signal")
if show_rsi:
    panes.append("rsi")
fig, row_of, total_px = make_section(panes, price_secondary_y=volume_overlay)

# Visible window — drives the x-range and fits each pane's y-axis to what's shown
view_lo, view_hi = view_window(session_date, time_view, df.index[-1], is_today)
dfw = df[(df.index >= view_lo) & (df.index <= view_hi)]
if dfw.empty:
    dfw = df

# Price pane (row 1)
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price",
    increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
    increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
), row=1, col=1)

if show_vwap and df["VWAP"].notna().any():
    fig.add_trace(go.Scatter(
        x=df.index, y=df["VWAP"], name="VWAP",
        line=dict(color=VWAP_COLOR, width=1.4),
    ), row=1, col=1)

# Regular-session (09:30) open reference line
fig.add_hline(y=o, line_dash="dot", line_color="#8a8a8a", line_width=1,
              annotation_text="9:30 Open", annotation_position="right", row=1, col=1)

# ORB box — shaded opening-range region + high/low lines
if show_orb and box_high is not None:
    fig.add_hrect(y0=box_low, y1=box_high, line_width=0,
                  fillcolor=ORB_COLOR, opacity=0.07, row=1, col=1)
    fig.add_hline(y=box_high, line=dict(color=ORB_COLOR, width=1.2, dash="dash"),
                  annotation_text=f"ORB High {box_high:,.2f}",
                  annotation_position="top left",
                  annotation_font_color=ORB_COLOR, row=1, col=1)
    fig.add_hline(y=box_low, line=dict(color=ORB_COLOR, width=1.2, dash="dash"),
                  annotation_text=f"ORB Low {box_low:,.2f}",
                  annotation_position="bottom left",
                  annotation_font_color=ORB_COLOR, row=1, col=1)

# Trade levels — entry / target / stop, when qfs_analyzer fired a signal today
if show_orb and ana and ana.get("entry") is not None:
    for name, y, color in (("Entry",  ana["entry"],       ENTRY_COLOR),
                           ("Target", ana["take_profit"], UP_COLOR),
                           ("Stop",   ana["stop_loss"],   DOWN_COLOR)):
        fig.add_hline(y=y, line=dict(color=color, width=1.1, dash="dot"),
                      annotation_text=f"{name} {y:,.2f}",
                      annotation_position="right",
                      annotation_font_color=color, row=1, col=1)

if show_volume:
    vol_colors = [VOL_UP if cl >= op else VOL_DOWN
                  for op, cl in zip(df["Open"], df["Close"])]
    _vmax = float(dfw["Volume"].max()) or 1
    if volume_overlay:
        # Translucent bars on a hidden secondary axis, scaled so the tallest
        # bar fills only the bottom ~25% of the price pane.
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, marker_line_width=0, opacity=0.35,
            showlegend=False, hovertemplate="Vol %{y:,.0f}<extra></extra>",
        ), row=1, col=1, secondary_y=True)
        fig.update_yaxes(range=[0, _vmax * 4], showgrid=False, zeroline=False,
                         showticklabels=False, showline=False,
                         secondary_y=True, row=1, col=1)
    else:
        vr = row_of["volume"]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, marker_line_width=0, opacity=0.7,
            showlegend=False,
        ), row=vr, col=1)
        fig.update_yaxes(title_text="Vol", gridcolor=GRID_COLOR,
                         range=[0, _vmax * 1.15], row=vr, col=1)

# MACD pane — Robinhood Legend style: 2-colour histogram + MACD & signal lines
if show_macd:
    mr = row_of["macd"]
    hist = df["MACD_hist"]
    hist_colors = [HIST_POS if (pd.notna(v) and v >= 0) else HIST_NEG for v in hist]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, name="Histogram",
        marker_color=hist_colors, marker_line_width=0, showlegend=False,
        hovertemplate="hist %{y:.3f}<extra></extra>",
    ), row=mr, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD", showlegend=False,
        line=dict(color=MACD_COLOR, width=1.4),
    ), row=mr, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_signal"], name="Signal", showlegend=False,
        line=dict(color=SIGNAL_COLOR, width=1.4),
    ), row=mr, col=1)
    fig.add_hline(y=0, line=dict(color="#8a8a8a", width=1, dash="dot"),
                  row=mr, col=1)

    # Vertical dashed line at each MACD/signal crossover (hist flips sign):
    # green = bullish (MACD up through signal), red = bearish. Added after the
    # traces so the pane's axes are realised and the shapes register.
    mprev = hist.shift(1)
    for ts, cur_h, prev_h in zip(df.index, hist, mprev):
        if pd.isna(prev_h) or pd.isna(cur_h):
            continue
        if prev_h < 0 < cur_h:
            fig.add_vline(x=ts, line=dict(color=XOVER_BULL, width=1, dash="dash"),
                          layer="below", row=mr, col=1)
        elif prev_h > 0 > cur_h:
            fig.add_vline(x=ts, line=dict(color=XOVER_BEAR, width=1, dash="dash"),
                          layer="below", row=mr, col=1)

    # Legend-style readout: MACD (white) · signal (amber) · hist (green/red)
    mlast = df.iloc[-1]
    hcol = HIST_POS if mlast["MACD_hist"] >= 0 else HIST_NEG
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.005, y=0.92,
        xanchor="left", yanchor="top", showarrow=False, align="left",
        text=(f"<b>MACD (12, 26, 9)</b>   "
              f"<span style='color:{TEXT_COLOR}'>{mlast['MACD']:.2f}</span>   "
              f"<span style='color:{SIGNAL_COLOR}'>{mlast['MACD_signal']:.2f}</span>   "
              f"<span style='color:{hcol}'>{mlast['MACD_hist']:.2f}</span>"),
        font=dict(size=12, family="monospace", color=TEXT_COLOR),
        row=mr, col=1,
    )
    _mser = pd.concat([dfw["MACD"], dfw["MACD_signal"], dfw["MACD_hist"]])
    _mlo, _mhi = float(_mser.min()), float(_mser.max())
    _mpad = (_mhi - _mlo) * 0.1 or 0.1
    fig.update_yaxes(title_text="", gridcolor=GRID_COLOR,
                     range=[_mlo - _mpad, _mhi + _mpad], row=mr, col=1)

# Signal pane — day-trading EMA(9/21) crossover + RSI, as a Buy/Sell/Hold ribbon.
# EMA9 > EMA21 = bullish (RSI<70 → Buy); EMA9 < EMA21 = bearish (RSI>30 → Sell);
# the RSI extremes flip a strong trend to Hold.  Square colour INTENSITY (alpha)
# encodes confidence = EMA9/21 spread in ATRs blended with how decisive RSI is.
if show_signal:
    sgr = row_of["signal"]
    _tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    _atr = _tr.rolling(14, min_periods=1).mean()
    _rgb = {"BUY": SIG_BUY_RGB, "SELL": SIG_SELL_RGB, "HOLD": SIG_HOLD_RGB}
    _hex = {"BUY": SIG_BUY, "SELL": SIG_SELL, "HOLD": SIG_HOLD}
    sig_colors, sig_custom = [], []
    last = ("HOLD", 0.0, "—")
    for close_v, ef, es, rsi_v, atr_v in zip(df["Close"], df["EMA_fast"],
                                             df["EMA_slow"], df["RSI"], _atr):
        ema_conf = min(1.0, abs(ef - es) / atr_v / sig_atr_thr) if atr_v else 0.0
        if ef > es and rsi_v < 70:
            lab = "BUY"
            conf = sig_ema_w * ema_conf + (1 - sig_ema_w) * max(0.0, min(1.0, (rsi_v - 50) / 20))
            tgt = close_v + sig_tgt_mult * atr_v
        elif ef < es and rsi_v > 30:
            lab = "SELL"
            conf = sig_ema_w * ema_conf + (1 - sig_ema_w) * max(0.0, min(1.0, (50 - rsi_v) / 20))
            tgt = close_v - sig_tgt_mult * atr_v
        else:
            lab = "HOLD"
            conf = (min(1.0, (rsi_v - 70) / 15) if rsi_v >= 70
                    else min(1.0, (30 - rsi_v) / 15) if rsi_v <= 30 else 0.3)
            tgt = None
        conf = max(0.0, min(1.0, conf))
        sig_colors.append(f"rgba({_rgb[lab]},{0.35 + 0.65 * conf:.2f})")
        tgt_s = f"${tgt:,.2f}" if tgt is not None else "—"
        sig_custom.append([lab, f"{conf * 100:.0f}%", tgt_s])
        last = (lab, conf, tgt_s)

    fig.add_trace(go.Scatter(
        x=df.index, y=[0] * len(df), mode="markers", name="Signal",
        marker=dict(symbol="square", size=10, color=sig_colors, line=dict(width=0)),
        customdata=sig_custom, showlegend=False,
        hovertemplate=("Signal %{customdata[0]} (conf %{customdata[1]}) "
                       "· target %{customdata[2]}<extra></extra>"),
    ), row=sgr, col=1)
    fig.update_yaxes(visible=False, range=[-1, 1], row=sgr, col=1)

    _ll, _lconf, _lt = last
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.005, y=0.96,
        xanchor="left", yanchor="top", showarrow=False, align="left",
        text=(f"<b>Signal (EMA {sig_fast}/{sig_slow} × RSI)</b>   "
              f"<span style='color:{_hex[_ll]}'>{_ll} {_lconf * 100:.0f}%</span>   "
              f"<span style='color:#8a8a8a'>· target {_lt}</span>"),
        font=dict(size=12, family="monospace", color=TEXT_COLOR),
        row=sgr, col=1,
    )

# RSI pane — line coloured by zone (each segment takes its zone's colour)
if show_rsi:
    rr = row_of["rsi"]
    rsi = df["RSI"]
    xs, ys = df.index, rsi.values
    cur = float(rsi.iloc[-1])
    zcol, zname = rsi_zone(cur)

    # Colour each segment by the zone of its left endpoint, then merge runs of
    # the same colour into one trace (sharing the boundary point so the line
    # stays continuous). These are traces, so order doesn't matter.
    seg_cols = [rsi_zone(ys[i])[0] for i in range(len(ys) - 1)]
    i = 0
    while i < len(seg_cols):
        j = i
        while j + 1 < len(seg_cols) and seg_cols[j + 1] == seg_cols[i]:
            j += 1
        fig.add_trace(go.Scatter(
            x=xs[i:j + 2], y=ys[i:j + 2], mode="lines",
            line=dict(color=seg_cols[i], width=2.0),
            showlegend=False, hoverinfo="skip",
        ), row=rr, col=1)
        i = j + 1

    # Transparent full-width trace carries the hover read-out.
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="RSI", showlegend=False,
        line=dict(color="rgba(0,0,0,0)"),
        hovertemplate="RSI %{y:.2f}<extra></extra>",
    ), row=rr, col=1)

    # Dark-grey 30–70 band + light-grey dashed reference lines, behind the line.
    # (added after the trace so the pane's axes are realised and shapes register)
    fig.add_hrect(y0=30, y1=70, fillcolor=RSI_BAND_FILL, opacity=1.0,
                  line_width=0, layer="below", row=rr, col=1)
    for yv in (30, 70):
        fig.add_hline(y=yv, line=dict(color=RSI_BAND_LINE, width=1, dash="dash"),
                      layer="below", row=rr, col=1)

    # Readout: RSI value (white) · zone label (zone colour)
    fig.add_annotation(
        xref="x domain", yref="y domain", x=0.005, y=0.92,
        xanchor="left", yanchor="top", showarrow=False, align="left",
        text=(f"<b>RSI (14)</b>   "
              f"<span style='color:{TEXT_COLOR}'>{cur:.2f}</span>   "
              f"<span style='color:{zcol}'>{zname}</span>"),
        font=dict(size=12, family="monospace", color=TEXT_COLOR),
        row=rr, col=1,
    )
    # Pad the visible range around the windowed data; ticks on zone boundaries.
    rlo = min(25.0, float(dfw["RSI"].min()))
    rhi = max(75.0, float(dfw["RSI"].max()))
    fig.update_yaxes(range=[rlo - 5, rhi + 5],
                     tickvals=[20, 30, 45, 55, 70, 80],
                     gridcolor=GRID_COLOR, row=rr, col=1)

# Section headers (added last so every pane's axes are realised)
section_label(fig, 1, "Price & Volume" if show_volume else "Price")
ind_panes = [p for p in panes if p in ("macd", "rsi")]
if ind_panes:
    section_label(fig, row_of[ind_panes[0]],
                  " & ".join({"macd": "MACD", "rsi": "RSI"}[p] for p in ind_panes))

style_section(fig, total_px)
_plo, _phi = float(dfw["Low"].min()), float(dfw["High"].max())
_ppad = (_phi - _plo) * 0.05 or 0.5
# secondary_y=False so this only sets the price axis, not the volume overlay axis
fig.update_yaxes(gridcolor=GRID_COLOR, title_text="Price ($)",
                 range=[_plo - _ppad, _phi + _ppad], secondary_y=False, row=1, col=1)
shade_sessions(fig, session_date, (view_lo, view_hi))
st.plotly_chart(fig, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True})

# RSI-zone scale legend (current zone highlighted for the latest candle)
if show_rsi:
    st.markdown(rsi_legend_html(float(df["RSI"].iloc[-1])), unsafe_allow_html=True)

# ── Stock information ─────────────────────────────────────────────────────────
st.divider()
st.markdown(f"#### 📋 {ticker} · Stock Information")
st.caption("Regular session")
info = get_info(ticker)
g = info.get
mc = g("marketCap")
dy = g("dividendYield")

r1 = st.columns(5)
r1[0].metric("Market Cap", f"${_fmt_num(mc)}" if mc not in (None, "") else "—")
r1[1].metric("P/E (TTM)", _fmt2(g("trailingPE")))
r1[2].metric("Dividend Yield", f"{float(dy):.2f}%" if dy not in (None, "") else "—")
r1[3].metric("Avg Volume", _fmt_num(g("averageVolume")))
r1[4].metric("52W High", _fmt_price(g("fiftyTwoWeekHigh")))

r2 = st.columns(5)
r2[0].metric("Open", _fmt_price(g("regularMarketOpen") or g("open")))
r2[1].metric("High Today", _fmt_price(g("dayHigh")))
r2[2].metric("Low Today", _fmt_price(g("dayLow")))
r2[3].metric("Volume", _fmt_num(g("regularMarketVolume") or g("volume")))
r2[4].metric("52W Low", _fmt_price(g("fiftyTwoWeekLow")))

st.caption(f"Source: Yahoo Finance via yfinance · {interval} · {len(df)} bars · "
           f"updated {datetime.now(MARKET_TZ):%I:%M:%S %p %Z}")
