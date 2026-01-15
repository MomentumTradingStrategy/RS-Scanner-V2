from datetime import datetime, timezone
import re

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Relative Strength Stock Scanner", layout="wide")

BENCHMARK = "SPY"

# Fixed file inside your GitHub repo
DATA_FILE = "data/Ticker-Price-Data.csv"


def _asof_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ============================================================
# CSS (SAME STYLE AS YOUR RS DASHBOARD)
# ============================================================
CSS = """
<style>
.block-container {max-width: 1750px; padding-top: 1.0rem; padding-bottom: 2rem;}
.section-title {font-weight: 900; font-size: 1.15rem; margin: 0.65rem 0 0.4rem 0;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0;}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
}
.card h3{margin:0 0 8px 0; font-size: 1.02rem; font-weight: 950;}
.card .hint{opacity:0.72; font-size:0.88rem; margin-top:-2px; margin-bottom:10px;}

.pl-table-wrap {border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10);}
table.pl-table {border-collapse: collapse; width: 100%; font-size: 13px;}
table.pl-table thead th {
  position: sticky; top: 0;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92);
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.12);
  font-weight: 900;
}
table.pl-table tbody td{
  padding: 7px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: middle;
}
td.mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
td.ticker {font-weight: 900;}
td.name {white-space: normal; line-height: 1.15;}

.pill{
  display:inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-weight: 950;
  font-size: 0.82rem;
  border: 1px solid rgba(255,255,255,0.12);
}
.pill-red{background: rgba(255,80,80,0.16); color:#FF6B6B;}
.pill-amber{background: rgba(255,200,60,0.16); color: rgba(255,200,60,0.98);}
.pill-green{background: rgba(80,255,120,0.16); color:#7CFC9A;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
SPECIAL_TICKER_MAP = {
    "BRK-A": "BRK.A",
    "BRK-B": "BRK.B",
    "BRKA": "BRK.A",
    "BRKB": "BRK.B",
}

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = t.replace(" ", "")
    t = t.replace("/", "-")
    t = SPECIAL_TICKER_MAP.get(t, t)
    return t


def to_float_pct_series(s: pd.Series) -> pd.Series:
    """
    Converts a Series of values like:
      1.23, "1.23", "1.23%", "-0.5 %"
    into fractional returns:
      1.23% -> 0.0123
    Assumes bare numbers are in percent units (1.23 means 1.23%).
    """
    if s.dtype.kind in "if":
        return pd.to_numeric(s, errors="coerce") / 100.0

    ss = s.astype(str).str.strip()
    ss = ss.str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    ss = ss.str.replace(r"[^0-9\.\-\+]", "", regex=True)
    out = pd.to_numeric(ss, errors="coerce") / 100.0
    return out


def find_col(df: pd.DataFrame, keys: list[str]) -> str | None:
    cols = [str(c) for c in df.columns]
    for c in cols:
        name = c.lower().strip()
        if any(k in name for k in keys):
            return c
    return None


def rs_bg(v):
    try:
        v = float(v)
    except:
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
    return f"background-color: rgb({r},{g},{b}); color:#0B0B0B; font-weight:900; border-radius:6px; padding:2px 6px; display:inline-block; min-width:32px; text-align:center;"


def pct_style(v):
    try:
        v = float(v)
    except:
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
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"${float(v):,.2f}"
    except:
        return ""


def fmt_pct(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{float(v):.2%}"
    except:
        return ""


def fmt_rs(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{float(v):.0f}"
    except:
        return ""


def render_table_html(df: pd.DataFrame, columns: list[str], height_px: int = 720):
    th = "".join([f"<th>{c}</th>" for c in columns])

    trs = []
    for _, row in df.iterrows():
        tds = []
        for c in columns:
            val = row.get(c, "")
            td_class = ""
            if c == "Ticker":
                td_class = "ticker"
            elif c == "Name":
                td_class = "name"

            if c == "Price":
                cell_html = fmt_price(val)
            elif c.startswith("% "):
                txt = fmt_pct(val)
                stl = pct_style(val)
                cell_html = f'<span style="{stl}">{txt}</span>' if stl and txt else (txt or "")
            elif c.startswith("RS "):
                txt = fmt_rs(val)
                stl = rs_bg(val)
                cell_html = f'<span style="{stl}">{txt}</span>' if stl and txt else (txt or "")
            else:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cell_html = ""
                else:
                    cell_html = str(val)

            tds.append(f'<td class="{td_class}">{cell_html}</td>')

        trs.append("<tr>" + "".join(tds) + "</tr>")

    table = f"""
    <div class="pl-table-wrap" style="max-height:{height_px}px; overflow:auto;">
      <table class="pl-table">
        <thead><tr>{th}</tr></thead>
        <tbody>
          {''.join(trs)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table, unsafe_allow_html=True)


# ============================================================
# LOAD DATA (FROM GITHUB FILE)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df_raw = load_data(DATA_FILE)
except Exception as e:
    st.error(
        f"Could not load `{DATA_FILE}`.\n\n"
        "Make sure the file exists in your repo at that exact path.\n\n"
        f"Error: {e}"
    )
    st.stop()

if df_raw.empty:
    st.error("Data file loaded but is empty.")
    st.stop()

# detect core columns
ticker_col = find_col(df_raw, ["ticker", "symbol"]) or df_raw.columns[0]
price_col  = find_col(df_raw, ["price", "last", "close"])

daily_col  = find_col(df_raw, ["daily", "1d", "day"])
weekly_col = find_col(df_raw, ["weekly", "1w", "week", "wk"])
monthly_col= find_col(df_raw, ["monthly", "1m", "month", "mo"])

# optional future cols
m3_col = find_col(df_raw, ["3m", "quarter", "qtr"])
m6_col = find_col(df_raw, ["6m", "half", "semi"])
y1_col = find_col(df_raw, ["1y", "12m", "annual", "year", "yr"])

# Build normalized frame
df = df_raw.copy()
df["Ticker"] = df[ticker_col].astype(str).map(normalize_ticker)
df = df[df["Ticker"].astype(str).str.len() > 0].copy()
df = df.drop_duplicates(subset=["Ticker"], keep="first")

if price_col:
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
else:
    df["Price"] = np.nan

# Return columns: convert to fractional returns
def safe_return(colname: str | None) -> pd.Series:
    if not colname:
        return pd.Series(np.nan, index=df.index)
    return to_float_pct_series(df[colname])

df["r_1d"] = safe_return(daily_col)
df["r_1w"] = safe_return(weekly_col)
df["r_1m"] = safe_return(monthly_col)

# Future horizons: allow NaN now
df["r_3m"] = safe_return(m3_col) if m3_col else np.nan
df["r_6m"] = safe_return(m6_col) if m6_col else np.nan
df["r_1y"] = safe_return(y1_col) if y1_col else np.nan

bench = normalize_ticker(BENCHMARK)
if bench not in set(df["Ticker"]):
    st.error(f"Benchmark {BENCHMARK} is missing as a row in your file. Add SPY to the dataset.")
    st.stop()

bench_row = df[df["Ticker"] == bench].iloc[0]

b_1d = float(bench_row["r_1d"]) if np.isfinite(bench_row["r_1d"]) else np.nan
b_1w = float(bench_row["r_1w"]) if np.isfinite(bench_row["r_1w"]) else np.nan
b_1m = float(bench_row["r_1m"]) if np.isfinite(bench_row["r_1m"]) else np.nan
b_3m = float(bench_row["r_3m"]) if np.isfinite(bench_row["r_3m"]) else np.nan
b_6m = float(bench_row["r_6m"]) if np.isfinite(bench_row["r_6m"]) else np.nan
b_1y = float(bench_row["r_1y"]) if np.isfinite(bench_row["r_1y"]) else np.nan

# Relative return vs SPY: RR = (1+r_stock)/(1+r_spy) - 1
def rel_ret(r: pd.Series, b: float) -> pd.Series:
    if not np.isfinite(b):
        return pd.Series(np.nan, index=r.index)
    return (1.0 + r) / (1.0 + b) - 1.0

df["rr_1d"] = rel_ret(df["r_1d"], b_1d)
df["rr_1w"] = rel_ret(df["r_1w"], b_1w)
df["rr_1m"] = rel_ret(df["r_1m"], b_1m)
df["rr_3m"] = rel_ret(df["r_3m"], b_3m)
df["rr_6m"] = rel_ret(df["r_6m"], b_6m)
df["rr_1y"] = rel_ret(df["r_1y"], b_1y)

# RS 1–99 percentile rank on relative returns
def to_rs_1_99(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return (x.rank(pct=True) * 99).round().clip(1, 99)

df["RS 1D"] = to_rs_1_99(df["rr_1d"])
df["RS 1W"] = to_rs_1_99(df["rr_1w"])
df["RS 1M"] = to_rs_1_99(df["rr_1m"])
df["RS 3M"] = to_rs_1_99(df["rr_3m"])  # will be NaN until data exists
df["RS 6M"] = to_rs_1_99(df["rr_6m"])
df["RS 1Y"] = to_rs_1_99(df["rr_1y"])

# Keep the raw % change (for display)
df["% 1D"] = df["r_1d"]
df["% 1W"] = df["r_1w"]
df["% 1M"] = df["r_1m"]
df["% 3M"] = df["r_3m"]
df["% 6M"] = df["r_6m"]
df["% 1Y"] = df["r_1y"]


# ============================================================
# UI (HEADER)
# ============================================================
st.title("Relative Strength Stock Scanner")
st.caption(f"As of: {_asof_ts()} • RS Benchmark: {BENCHMARK}")

# Info card
st.markdown(
    f"""
<div class="card">
  <h3>Universe Loaded</h3>
  <div class="hint">This scanner uses your uploaded dataset from GitHub (no user uploads).</div>
  <div class="small-muted">File: <b>{DATA_FILE}</b> • Rows: <b>{len(df):,}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.subheader("Scanner Controls")

    rs_cols_order = ["RS 1D", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"]
    available_rs = [c for c in rs_cols_order if c in df.columns]

    primary_tf = st.selectbox(
        "Primary Timeframe",
        available_rs,
        index=2 if "RS 1M" in available_rs else 0
    )

    rs_min = st.slider("Minimum RS Rating", 1, 99, 90, 1)

    mode = st.selectbox(
        "Scan Mode",
        [
            "Primary timeframe only",
            "All timeframes >= threshold",
            "Accelerating (long → short improving)",
            "Decelerating (short → long weakening)",
        ],
        index=0
    )

    max_results = st.slider("Max Results", 25, 2000, 200, step=25)

    hide_benchmark = st.checkbox("Hide SPY row", value=True)


# ============================================================
# SCAN LOGIC
# ============================================================
df_out = df.copy()

if hide_benchmark:
    df_out = df_out[df_out["Ticker"] != bench].copy()

# Choose which RS columns count (even if some are mostly NaN, it’s fine)
rs_cols_present = [c for c in ["RS 1D", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"] if c in df_out.columns]

if mode == "Primary timeframe only":
    df_f = df_out[df_out[primary_tf] >= rs_min].copy()

elif mode == "All timeframes >= threshold":
    cond = True
    for c in rs_cols_present:
        cond = cond & (df_out[c] >= rs_min)
    df_f = df_out[cond].copy()

elif mode == "Accelerating (long → short improving)":
    # Use longest available vs shortest available
    long_tf = rs_cols_present[-1]
    short_tf = rs_cols_present[0]
    df_f = df_out[(df_out[short_tf] > df_out[long_tf]) & (df_out[primary_tf] >= rs_min)].copy()

else:  # Decelerating
    long_tf = rs_cols_present[-1]
    short_tf = rs_cols_present[0]
    df_f = df_out[(df_out[short_tf] < df_out[long_tf]) & (df_out[primary_tf] >= rs_min)].copy()

# Sort best → worst by primary, then RS 1Y as tie-break (if present)
tie = "RS 1Y" if "RS 1Y" in df_f.columns else primary_tf
df_f = df_f.sort_values([primary_tf, tie], ascending=[False, False])

st.markdown('<div class="section-title">Scanner Results</div>', unsafe_allow_html=True)
st.markdown(f'<div class="small-muted">Matches: <b>{len(df_f):,}</b></div>', unsafe_allow_html=True)

# Display columns like your dashboard
show_cols = [
    "Ticker",
    "Price",
    "RS 1D",
    "RS 1W",
    "RS 1M",
    "RS 3M",
    "RS 6M",
    "RS 1Y",
    "% 1D",
    "% 1W",
    "% 1M",
    "% 3M",
    "% 6M",
    "% 1Y",
]

# Ensure all exist
for c in show_cols:
    if c not in df_f.columns:
        df_f[c] = np.nan

render_table_html(df_f[show_cols].head(max_results), show_cols, height_px=950)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    """
**How RS is Calculated (from your uploaded dataset)**  
- Your file provides % change for each horizon (Daily/Weekly/Monthly/etc.).  
- We compute **relative return vs SPY**:  
  **RR = (1 + r_stock) / (1 + r_SPY) − 1**  
- Then we percentile-rank RR across your universe into **RS 1–99**.  
"""
)
