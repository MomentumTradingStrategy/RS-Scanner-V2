# app.py
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Relative Strength Stock Screener", layout="wide")

# =========================
# CONFIG
# =========================
DATA_DIR = "Data"
SCREENER_FILE = "Screener_Data.csv"
SPY_FILE = "SPY_Data.csv"
BENCHMARK = "SPY"


def _asof_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def find_file(*candidates: str) -> str:
    """
    Streamlit Cloud runs in /mount/src/<repo>. Local runs elsewhere.
    This searches a few common locations so you don't get FileNotFoundError.
    """
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find file. Tried: {list(candidates)}")


SCREENER_PATH = find_file(
    os.path.join(DATA_DIR, SCREENER_FILE),
    SCREENER_FILE,
    os.path.join(".", DATA_DIR, SCREENER_FILE),
)
SPY_PATH = find_file(
    os.path.join(DATA_DIR, SPY_FILE),
    SPY_FILE,
    os.path.join(".", DATA_DIR, SPY_FILE),
)

# =========================
# CSS (match your RS dashboard vibe)
# =========================
CSS = """
<style>
.block-container {max-width: 1750px; padding-top: 1.0rem; padding-bottom: 2rem;}
.section-title {font-weight: 900; font-size: 1.15rem; margin: 0.65rem 0 0.4rem 0;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0;}

.pl-table-wrap {border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10);}
table.pl-table {border-collapse: collapse; width: 100%; font-size: 13px; table-layout: fixed;}
table.pl-table thead th {
  position: sticky; top: 0;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92);
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.12);
  font-weight: 900;
  white-space: nowrap;
}
table.pl-table tbody td{
  padding: 7px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: middle;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

td.ticker {font-weight: 900; width: 84px;}
td.name {white-space: normal; line-height: 1.15;}
td.price {width: 92px;}
td.rs {width: 64px;}
td.pct {width: 86px; text-align:right;}
th.pct {text-align:right;}

th.rs-gap, td.rs-gap{
  width: 74px !important;
  max-width: 74px !important;
  padding-left: 6px !important;
  padding-right: 6px !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# CSV LOADERS
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


df_raw = load_csv(SCREENER_PATH)
spy_raw = load_csv(SPY_PATH)

# =========================
# COLUMN NORMALIZATION
# =========================
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct_to_decimal(v: pd.Series) -> pd.Series:
    # CSV uses percent values like 12.34 meaning 12.34%
    return to_num(v) / 100.0


# Required columns (your TradingView export + SPY export)
COL_TICKER = "Symbol"
COL_NAME = "Description"
COL_PRICE = "Price"
COL_1D = "Price Change % 1 day"

RET_COLS = {
    "1W": "Performance % 1 week",
    "1M": "Performance % 1 month",
    "3M": "Performance % 3 months",
    "6M": "Performance % 6 months",
    "1Y": "Performance % 1 year",
}

# Defensive checks (so it fails loudly with a clear message)
missing = [c for c in [COL_TICKER, COL_NAME, COL_PRICE, COL_1D, *RET_COLS.values()] if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in {SCREENER_FILE}: {missing}")
    st.stop()

spy_missing = [c for c in ["Symbol", *RET_COLS.values()] if c not in spy_raw.columns]
if spy_missing:
    st.error(f"Missing required columns in {SPY_FILE}: {spy_missing}")
    st.stop()

# =========================
# BUILD RS (relative return vs SPY) + RS ratings (1–99)
# =========================
def compute_rs_table(stocks_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Ticker"] = stocks_df[COL_TICKER].astype(str).str.upper().str.strip()
    out["Name"] = stocks_df[COL_NAME].astype(str)
    out["Price"] = to_num(stocks_df[COL_PRICE])

    # Returns (% columns)
    out["% 1D"] = pct_to_decimal(stocks_df[COL_1D])
    out["% 1W"] = pct_to_decimal(stocks_df[RET_COLS["1W"]])
    out["% 1M"] = pct_to_decimal(stocks_df[RET_COLS["1M"]])
    out["% 3M"] = pct_to_decimal(stocks_df[RET_COLS["3M"]])
    out["% 6M"] = pct_to_decimal(stocks_df[RET_COLS["6M"]])
    out["% 1Y"] = pct_to_decimal(stocks_df[RET_COLS["1Y"]])

    # SPY single-row inputs
    spy_row = spy_df.iloc[0]
    spy_r = {
        "1W": float(pct_to_decimal(pd.Series([spy_row[RET_COLS["1W"]]])).iloc[0]),
        "1M": float(pct_to_decimal(pd.Series([spy_row[RET_COLS["1M"]]])).iloc[0]),
        "3M": float(pct_to_decimal(pd.Series([spy_row[RET_COLS["3M"]]])).iloc[0]),
        "6M": float(pct_to_decimal(pd.Series([spy_row[RET_COLS["6M"]]])).iloc[0]),
        "1Y": float(pct_to_decimal(pd.Series([spy_row[RET_COLS["1Y"]]])).iloc[0]),
    }

    # Relative Return vs SPY: RR = (1+r_stock)/(1+r_spy) - 1
    rr = {}
    for k, col in [("1W", "% 1W"), ("1M", "% 1M"), ("3M", "% 3M"), ("6M", "% 6M"), ("1Y", "% 1Y")]:
        rr[k] = (1.0 + out[col]) / (1.0 + spy_r[k]) - 1.0

    # RS ratings 1–99 = percentile rank of RR across universe
    for k in ["1W", "1M", "3M", "6M", "1Y"]:
        s = pd.to_numeric(rr[k], errors="coerce")
        out[f"RS {k}"] = (s.rank(pct=True) * 99).round().clip(1, 99)

    # Convenience: RS gap (shift) = RS 1M - RS 1Y (signed)
    out["RS GAP (1M-1Y)"] = (to_num(out["RS 1M"]) - to_num(out["RS 1Y"])).round().astype("Int64")

    return out


df = compute_rs_table(df_raw, spy_raw)

# =========================
# UI CONTROLS
# =========================
RS_KEYS = ["1W", "1M", "3M", "6M", "1Y"]

st.title("Relative Strength Stock Screener")
st.caption(f"As of: {_asof_ts()} • RS Benchmark: {BENCHMARK}")

with st.sidebar:
    st.subheader("Controls")

    rank_by = st.selectbox("Rank by", [f"RS {k}" for k in ["1W", "1M", "3M", "6M", "1Y"]], index=1)
    min_primary = st.slider("Minimum RS (Primary)", 1, 99, 70)

    scan_mode = st.selectbox(
        "Scan Mode",
        [
            "Primary timeframe only",
            "All timeframes (1Y→6M→3M→1M)",
            "Accelerating",
            "Decelerating",
        ],
        index=0,
    )

    # Only show accel/decel knobs when relevant
    accel_decel_strength = None
    require_smooth = None
    sort_mode = None

    if scan_mode in ("Accelerating", "Decelerating"):
        accel_decel_strength = st.slider("Acceleration/Deceleration Strength (RS Gap)", 0, 60, 15)
        require_smooth = st.checkbox("Require smooth trend (1Y↔6M↔3M↔1M)", value=True)
        sort_mode = st.selectbox("Sort results by", ["RS Gap (shift)", "Primary RS"], index=0)

    max_results = st.slider("Max Results", 25, 400, 200, step=25)

# =========================
# FILTER LOGIC
# =========================
def is_monotonic_accel(r1y, r6m, r3m, r1m) -> bool:
    # accelerating: strength improving into shorter horizon
    return (r1y <= r6m) and (r6m <= r3m) and (r3m <= r1m)


def is_monotonic_decel(r1y, r6m, r3m, r1m) -> bool:
    # decelerating: losing momentum into shorter horizon
    return (r1y >= r6m) and (r6m >= r3m) and (r3m >= r1m)


def apply_scan(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()

    # Base filter: minimum on primary timeframe
    d = d[pd.to_numeric(d[rank_by], errors="coerce") >= float(min_primary)]

    if scan_mode == "Primary timeframe only":
        return d

    if scan_mode == "All timeframes (1Y→6M→3M→1M)":
        # Enforce ALL four longer horizons (skip 1W entirely)
        for col in ["RS 1Y", "RS 6M", "RS 3M", "RS 1M"]:
            d = d[pd.to_numeric(d[col], errors="coerce") >= float(min_primary)]
        return d

    # Accelerating / Decelerating
    r1y = pd.to_numeric(d["RS 1Y"], errors="coerce")
    r6m = pd.to_numeric(d["RS 6M"], errors="coerce")
    r3m = pd.to_numeric(d["RS 3M"], errors="coerce")
    r1m = pd.to_numeric(d["RS 1M"], errors="coerce")
    gap = pd.to_numeric(d["RS GAP (1M-1Y)"], errors="coerce")

    if scan_mode == "Accelerating":
        d = d[gap >= float(accel_decel_strength)]
        if require_smooth:
            mask = [is_monotonic_accel(a, b, c, e) for a, b, c, e in zip(r1y, r6m, r3m, r1m)]
            d = d[pd.Series(mask, index=d.index)]

    if scan_mode == "Decelerating":
        d = d[gap <= -float(accel_decel_strength)]
        if require_smooth:
            mask = [is_monotonic_decel(a, b, c, e) for a, b, c, e in zip(r1y, r6m, r3m, r1m)]
            d = d[pd.Series(mask, index=d.index)]

    return d


df_f = apply_scan(df)

# Sorting
if scan_mode in ("Accelerating", "Decelerating"):
    if sort_mode == "RS Gap (shift)":
        # accelerating: biggest positive first; decelerating: most negative first
        asc = True if scan_mode == "Decelerating" else False
        df_f = df_f.sort_values("RS GAP (1M-1Y)", ascending=asc)
    else:
        df_f = df_f.sort_values(rank_by, ascending=False)
else:
    df_f = df_f.sort_values(rank_by, ascending=False)

df_f = df_f.head(max_results)

# =========================
# RENDER HELPERS
# =========================
def rs_bg(v):
    try:
        v = float(v)
    except Exception:
        return ""
    if np.isnan(v):
        return ""
    x = (v - 1) / 98.0
    if x < 0.5:
        r = 255
        g = int(80 + (x / 0.5) * (180 - 80))
    else:
        r = int(255 - ((x - 0.5) / 0.5) * (255 - 40))
        g = 200
    b = 60
    return (
        f"background-color: rgb({r},{g},{b}); color:#0B0B0B; "
        "font-weight:900; border-radius:6px; padding:2px 6px; "
        "display:inline-block; min-width:32px; text-align:center;"
    )


def gap_bg(v):
    # Signed shift badge: green for +, red for -
    try:
        iv = int(v)
    except Exception:
        return ""
    if iv > 0:
        return "background: rgba(80,255,120,0.16); color:#7CFC9A; font-weight:950; border-radius:6px; padding:2px 6px; display:inline-block; min-width:40px; text-align:center;"
    if iv < 0:
        return "background: rgba(255,80,80,0.18); color:#FF6B6B; font-weight:950; border-radius:6px; padding:2px 6px; display:inline-block; min-width:40px; text-align:center;"
    return "background: rgba(255,200,60,0.16); color: rgba(255,200,60,0.98); font-weight:950; border-radius:6px; padding:2px 6px; display:inline-block; min-width:40px; text-align:center;"


def pct_style(v):
    try:
        v = float(v)
    except Exception:
        return ""
    if np.isnan(v):
        return ""
    if v > 0:
        return "color:#7CFC9A; font-weight:800;"
    if v < 0:
        return "color:#FF6B6B; font-weight:800;"
    return "opacity:0.9; font-weight:700;"


def fmt_price(v):
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return ""


def fmt_pct(v):
    try:
        return f"{float(v):.2%}"
    except Exception:
        return ""


def render_table_html(d: pd.DataFrame, columns: list[str], height_px: int = 720):
    ths = []
    for c in columns:
        cls = ""
        if c.startswith("% "):
            cls = ' class="pct"'
        if c == "RS GAP (1M-1Y)":
            cls = ' class="rs-gap"'
        ths.append(f"<th{cls}>{c}</th>")
    thead = "".join(ths)

    trs = []
    for _, row in d.iterrows():
        tds = []
        for c in columns:
            val = row.get(c, "")
            td_class = ""
            if c == "Ticker":
                td_class = "ticker"
            elif c == "Name":
                td_class = "name"
            elif c == "Price":
                td_class = "price"
            elif c.startswith("RS "):
                td_class = "rs"
            elif c.startswith("% "):
                td_class = "pct"
            elif c == "RS GAP (1M-1Y)":
                td_class = "rs-gap"

            if c == "Price":
                cell = fmt_price(val)
            elif c.startswith("% "):
                txt = fmt_pct(val)
                stl = pct_style(val)
                cell = f'<span style="{stl}">{txt}</span>' if txt else ""
            elif c.startswith("RS "):
                txt = "" if pd.isna(val) else f"{float(val):.0f}"
                stl = rs_bg(val)
                cell = f'<span style="{stl}">{txt}</span>' if txt else ""
            elif c == "RS GAP (1M-1Y)":
                if pd.isna(val):
                    cell = ""
                else:
                    txt = f"{int(val):d}"
                    stl = gap_bg(val)
                    cell = f'<span style="{stl}">{txt}</span>'
            else:
                cell = "" if pd.isna(val) else str(val)

            tds.append(f'<td class="{td_class}">{cell}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")

    table = f"""
    <div class="pl-table-wrap" style="max-height:{height_px}px; overflow:auto;">
      <table class="pl-table">
        <thead><tr>{thead}</tr></thead>
        <tbody>
          {''.join(trs)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table, unsafe_allow_html=True)


# =========================
# DISPLAY
# =========================
st.markdown('<div class="section-title">Scanner Results</div>', unsafe_allow_html=True)
st.caption(f"Universe: {len(df):,} • Matches: {len(df_f):,}")

# Columns:
base_cols = [
    "Ticker",
    "Name",
    "Price",
    "RS 1W",
    "RS 1M",
    "RS 3M",
    "RS 6M",
    "RS 1Y",
]
pct_cols = ["% 1D", "% 1W", "% 1M", "% 3M", "% 6M", "% 1Y"]

# Show RS GAP column ONLY in Accelerating / Decelerating
if scan_mode in ("Accelerating", "Decelerating"):
    show_cols = base_cols + ["RS GAP (1M-1Y)"] + pct_cols
else:
    show_cols = base_cols + pct_cols

render_table_html(df_f, show_cols, height_px=820)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    """
**How RS is Calculated:**
- Each stock is compared to **SPY** over each timeframe (relative return vs SPY).
- Then all stocks in your screener universe are ranked against each other and assigned an **RS rating (1–99)**.
"""
)


