from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Relative Strength Stock Scanner", layout="wide")

BENCHMARK = "SPY"
DATA_FILE = "data/Screener_Data.csv"
SPY_FILE = "data/SPY_Data.csv"


def _asof_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# =========================
# CSS (MATCH YOUR DASHBOARD)
# =========================
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
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = t.replace(" ", "")
    t = t.replace("/", "-")
    return t


def to_float_pct_series(s: pd.Series) -> pd.Series:
    """
    Converts percent-units to fractional returns.
    Example: 12.3 -> 0.123, "12.3%" -> 0.123
    """
    if s is None:
        return pd.Series(np.nan)

    if getattr(s.dtype, "kind", "") in "if":
        return pd.to_numeric(s, errors="coerce") / 100.0

    ss = s.astype(str).str.strip()
    ss = ss.str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    ss = ss.str.replace(r"[^0-9\.\-\+]", "", regex=True)
    return pd.to_numeric(ss, errors="coerce") / 100.0


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = [str(c) for c in df.columns]
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        cand_l = cand.lower().strip()
        if cand_l in low:
            return low[cand_l]
    # fallback: contains match
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
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
    return (
        f"background-color: rgb({r},{g},{b}); color:#0B0B0B; font-weight:900; "
        "border-radius:6px; padding:2px 6px; display:inline-block; min-width:32px; text-align:center;"
    )


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


# Sparkline from a small set of RR values (we don't have full price series)
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def spark_from_values(vals: list[float], n=26):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return "", []
    s = pd.Series(vals)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        mid = len(SPARK_CHARS) // 2
        levels = [mid] * n
        return (SPARK_CHARS[mid] * n), levels

    scaled = (s - lo) / (hi - lo)
    idx = (scaled * (len(SPARK_CHARS) - 1)).round().astype(int).clip(0, len(SPARK_CHARS) - 1).tolist()

    # stretch to length n
    out_levels = []
    for lv in idx:
        out_levels.extend([lv] * max(1, n // len(idx)))
    out_levels = (out_levels + [idx[-1]] * n)[:n]

    spark = "".join(SPARK_CHARS[i] for i in out_levels)
    return spark, out_levels


def spark_html(spark: str, levels: list[int]):
    if not spark or not levels or len(spark) != len(levels):
        return ""

    def level_to_rgb(lv: int):
        t = lv / 7.0
        if t <= 0.5:
            k = t / 0.5
            r1, g1, b1 = 255, 80, 80
            r2, g2, b2 = 255, 200, 60
            r = int(r1 + (r2 - r1) * k)
            g = int(g1 + (g2 - g1) * k)
            b = int(b1 + (b2 - b1) * k)
        else:
            k = (t - 0.5) / 0.5
            r1, g1, b1 = 255, 200, 60
            r2, g2, b2 = 80, 255, 120
            r = int(r1 + (r2 - r1) * k)
            g = int(g1 + (g2 - g1) * k)
            b = int(b1 + (b2 - b1) * k)
        return r, g, b

    spans = []
    for ch, lv in zip(spark, levels):
        r, g, b = level_to_rgb(int(lv))
        spans.append(f'<span style="color: rgb({r},{g},{b}); font-weight:900;">{ch}</span>')
    return "".join(spans)


def render_table_html(df: pd.DataFrame, columns: list[str], height_px: int = 900):
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
            elif c in ("RS Trend",):
                td_class = "mono"

            if c == "Price":
                cell_html = fmt_price(val)
            elif c.startswith("% "):
                txt = fmt_pct(val)
                stl = pct_style(val)
                cell_html = f'<span style="{stl}">{txt}</span>' if stl and txt != "" else txt
            elif c.startswith("RS "):
                txt = fmt_rs(val)
                stl = rs_bg(val)
                cell_html = f'<span style="{stl}">{txt}</span>' if stl and txt != "" else txt
            elif c == "RS Trend":
                cell_html = spark_html(str(val), row.get("__spark_levels", []))
            else:
                cell_html = "" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)

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
# LOAD DATA
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def required_cols_check(df: pd.DataFrame, cols: list[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"{label} is missing required columns: {missing}")
        st.stop()


df_raw = load_csv(DATA_FILE)
spy_raw = load_csv(SPY_FILE)

if df_raw.empty:
    st.error(f"{DATA_FILE} loaded but is empty.")
    st.stop()
if spy_raw.empty:
    st.error(f"{SPY_FILE} loaded but is empty.")
    st.stop()

# map columns (supports minor header variations)
c_symbol = find_col(df_raw, ["Symbol"])
c_name = find_col(df_raw, ["Description", "Name"])
c_price = find_col(df_raw, ["Price", "Last"])

c_1d = find_col(df_raw, ["Price Change % 1 day", "1 day", "daily"])
c_1w = find_col(df_raw, ["Performance % 1 week", "1 week", "weekly"])
c_1m = find_col(df_raw, ["Performance % 1 month", "1 month", "monthly"])
c_3m = find_col(df_raw, ["Performance % 3 months", "3 months", "quarter"])
c_6m = find_col(df_raw, ["Performance % 6 months", "6 months", "half"])
c_1y = find_col(df_raw, ["Performance % 1 year", "1 year", "annual"])

if not c_symbol:
    st.error("Universe CSV must include a Symbol column (or similar).")
    st.stop()

# SPY file uses same headers
spy_symbol = find_col(spy_raw, ["Symbol"]) or c_symbol
spy_1d = find_col(spy_raw, [c_1d]) if c_1d else None
spy_1w = find_col(spy_raw, [c_1w]) if c_1w else None
spy_1m = find_col(spy_raw, [c_1m]) if c_1m else None
spy_3m = find_col(spy_raw, [c_3m]) if c_3m else None
spy_6m = find_col(spy_raw, [c_6m]) if c_6m else None
spy_1y = find_col(spy_raw, [c_1y]) if c_1y else None

# Build universe frame
df = pd.DataFrame()
df["Ticker"] = df_raw[c_symbol].astype(str).map(normalize_ticker)
df = df[df["Ticker"].str.len() > 0].drop_duplicates(subset=["Ticker"]).copy()
df["Name"] = df_raw[c_name].astype(str) if c_name else df["Ticker"]
df["Price"] = pd.to_numeric(df_raw[c_price], errors="coerce") if c_price else np.nan

df["r_1d"] = to_float_pct_series(df_raw[c_1d]) if c_1d else np.nan
df["r_1w"] = to_float_pct_series(df_raw[c_1w]) if c_1w else np.nan
df["r_1m"] = to_float_pct_series(df_raw[c_1m]) if c_1m else np.nan
df["r_3m"] = to_float_pct_series(df_raw[c_3m]) if c_3m else np.nan
df["r_6m"] = to_float_pct_series(df_raw[c_6m]) if c_6m else np.nan
df["r_1y"] = to_float_pct_series(df_raw[c_1y]) if c_1y else np.nan

# Pull SPY returns from SPY file
spy_raw["__sym__"] = spy_raw[spy_symbol].astype(str).map(normalize_ticker)
spy_row = spy_raw[spy_raw["__sym__"] == normalize_ticker(BENCHMARK)]
if spy_row.empty:
    st.error(f"No row found for {BENCHMARK} in {SPY_FILE}. Make sure Symbol=SPY exists.")
    st.stop()
spy_row = spy_row.iloc[0]

def spy_ret(col):
    if not col or col not in spy_raw.columns:
        return np.nan
    return float(to_float_pct_series(pd.Series([spy_row[col]])).iloc[0])

b_1d = spy_ret(spy_1d)
b_1w = spy_ret(spy_1w)
b_1m = spy_ret(spy_1m)
b_3m = spy_ret(spy_3m)
b_6m = spy_ret(spy_6m)
b_1y = spy_ret(spy_1y)

# Relative return vs SPY
def rel_ret(r: pd.Series, b: float) -> pd.Series:
    if not np.isfinite(b):
        return pd.Series(np.nan, index=r.index)
    return (1.0 + r) / (1.0 + b) - 1.0

df["rr_1w"] = rel_ret(df["r_1w"], b_1w)
df["rr_1m"] = rel_ret(df["r_1m"], b_1m)
df["rr_3m"] = rel_ret(df["r_3m"], b_3m)
df["rr_6m"] = rel_ret(df["r_6m"], b_6m)
df["rr_1y"] = rel_ret(df["r_1y"], b_1y)

def to_rs_1_99(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return (x.rank(pct=True) * 99).round().clip(1, 99)

df["RS 1W"] = to_rs_1_99(df["rr_1w"])
df["RS 1M"] = to_rs_1_99(df["rr_1m"])
df["RS 3M"] = to_rs_1_99(df["rr_3m"])
df["RS 6M"] = to_rs_1_99(df["rr_6m"])
df["RS 1Y"] = to_rs_1_99(df["rr_1y"])

# Display % columns (absolute)
df["% 1D"] = df["r_1d"]
df["% 1W"] = df["r_1w"]
df["% 1M"] = df["r_1m"]
df["% 3M"] = df["r_3m"]
df["% 6M"] = df["r_6m"]
df["% 1Y"] = df["r_1y"]

# RS Trend sparkline (based on RR values across horizons)
spark_vals = []
spark_levels = []
for i in range(len(df)):
    vals = [
        df.iloc[i]["rr_1y"],
        df.iloc[i]["rr_6m"],
        df.iloc[i]["rr_3m"],
        df.iloc[i]["rr_1m"],
        df.iloc[i]["rr_1w"],
    ]
    sp, lv = spark_from_values(vals, n=26)
    spark_vals.append(sp)
    spark_levels.append(lv)

df["RS Trend"] = spark_vals
df["__spark_levels"] = spark_levels


# ============================================================
# UI
# ============================================================
st.title("Relative Strength Stock Scanner")
st.caption(f"As of: {_asof_ts()} • RS Benchmark: {BENCHMARK} • Source: CSV vs CSV")

with st.sidebar:
    st.subheader("Controls")

    primary_tf = st.selectbox(
        "Rank by",
        ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"],
        index=1,
        key="primary_tf",
    )

    rs_min = st.slider("Minimum RS (Primary)", 1, 99, 70, 1, key="rs_min")

    mode = st.selectbox(
        "Scan Mode",
        [
            "Primary timeframe only",
            "All timeframes >= threshold",
            "Accelerating (1Y→6M→3M→1M improving)",
            "Decelerating (1M→3M→6M→1Y weakening)",
        ],
        index=0,
        key="mode",
    )

    max_results = st.slider("Max Results", 25, 2000, 200, 25, key="max_results")
    hide_benchmark = st.checkbox("Hide SPY row", value=True, key="hide_benchmark")

# ============================================================
# SCAN LOGIC
# ============================================================
df_out = df.copy()
if hide_benchmark:
    df_out = df_out[df_out["Ticker"] != normalize_ticker(BENCHMARK)].copy()

rs_cols = ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"]

if mode == "Primary timeframe only":
    df_f = df_out[df_out[primary_tf].fillna(0) >= rs_min].copy()

elif mode == "All timeframes >= threshold":
    cond = True
    for c in rs_cols:
        cond = cond & (df_out[c].fillna(0) >= rs_min)
    df_f = df_out[cond].copy()

elif mode.startswith("Accelerating"):
    # Only use 1Y/6M/3M/1M (no 1W per your rule)
    cond = (df_out[primary_tf].fillna(0) >= rs_min)
    cond = cond & (df_out["RS 6M"] >= df_out["RS 1Y"])
    cond = cond & (df_out["RS 3M"] >= df_out["RS 6M"])
    cond = cond & (df_out["RS 1M"] >= df_out["RS 3M"])
    df_f = df_out[cond].copy()

else:
    # Decelerating: 1M <= 3M <= 6M <= 1Y
    cond = (df_out[primary_tf].fillna(0) >= rs_min)
    cond = cond & (df_out["RS 3M"] >= df_out["RS 1M"])
    cond = cond & (df_out["RS 6M"] >= df_out["RS 3M"])
    cond = cond & (df_out["RS 1Y"] >= df_out["RS 6M"])
    df_f = df_out[cond].copy()

tie = "RS 1Y" if "RS 1Y" in df_f.columns else primary_tf
df_f = df_f.sort_values([primary_tf, tie], ascending=[False, False])

st.markdown('<div class="section-title">Scanner Results</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="small-muted">Universe: <b>{len(df_out):,}</b> • Matches: <b>{len(df_f):,}</b></div>',
    unsafe_allow_html=True
)

show_cols = [
    "Ticker",
    "Name",
    "Price",
    "RS Trend",
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

render_table_html(df_f[show_cols].head(max_results), show_cols, height_px=950)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    """
**How RS is Calculated (MarketSurge-style framework):**  
- Your CSV supplies each stock’s % change for 1W/1M/3M/6M/1Y (and 1D if included).  
- Your SPY CSV supplies SPY’s % change for the same horizons.  
- Relative return vs SPY: **RR = (1 + r_stock) / (1 + r_SPY) − 1**  
- RS rating: percentile-rank RR across your universe into **1–99**.  
"""
)
