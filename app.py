from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# yfinance is used for SPY benchmark returns (live)
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Relative Strength Stock Scanner", layout="wide")

BENCHMARK = "SPY"

# If you want to keep repo path later, swap this back to "data/Ticker-Price-Data.csv"
DATA_FILE = "data/Screener Tool (web app)_2026-02-05.csv"


def _asof_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ============================================================
# CSS (SAME STYLE AS YOUR RS DASHBOARD)
# ============================================================
CSS = """
<style>
/* Keep the app looking like your dark dashboard */
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
def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = t.replace(" ", "")
    t = t.replace("/", "-")
    return t


def to_float_pct_series(s: pd.Series) -> pd.Series:
    """
    Values in your CSV are typically numeric percents:
      5.2 means 5.2%
    Converts to fractional returns:
      0.052
    """
    if s is None:
        return pd.Series(np.nan)

    if s.dtype.kind in "if":
        return pd.to_numeric(s, errors="coerce") / 100.0

    ss = s.astype(str).str.strip()
    ss = ss.str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    ss = ss.str.replace(r"[^0-9\.\-\+]", "", regex=True)
    return pd.to_numeric(ss, errors="coerce") / 100.0


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
        f"background-color: rgb({r},{g},{b}); color:#0B0B0B; font-weight:900; "
        "border-radius:6px; padding:2px 6px; display:inline-block; min-width:32px; text-align:center;"
    )


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
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"${float(v):,.2f}"
    except Exception:
        return ""


def fmt_pct(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{float(v):.2%}"
    except Exception:
        return ""


def fmt_num(v, digits=2):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{float(v):,.{digits}f}"
    except Exception:
        return ""


def fmt_rs(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{float(v):.0f}"
    except Exception:
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
                # non-% non-RS columns: keep compact numeric formatting if possible
                if isinstance(val, (int, float, np.floating)) and not (isinstance(val, float) and np.isnan(val)):
                    cell_html = fmt_num(val, 2)
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
# DATA LOAD
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def fetch_spy_returns() -> dict:
    """
    Compute benchmark returns off SPY (Adj Close) using trading-day approximations:
      1D  = 1
      1W  = 5
      1M  = 21
      3M  = 63
      6M  = 126
      1Y  = 252
    """
    periods = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}

    # pull enough history for 252 trading days + buffer
    hist = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        return {k: np.nan for k in periods}

    px = hist["Close"].dropna()
    if len(px) < 260:
        # still try, but may have NaNs for longer horizons
        pass

    out = {}
    last = float(px.iloc[-1])
    for k, n in periods.items():
        if len(px) > n:
            prev = float(px.iloc[-(n + 1)])
            out[k] = (last / prev) - 1.0
        else:
            out[k] = np.nan
    return out


try:
    df_raw = load_data(DATA_FILE)
except Exception as e:
    st.error(
        f"Could not load `{DATA_FILE}`.\n\n"
        "If you moved the file path, update DATA_FILE.\n\n"
        f"Error: {e}"
    )
    st.stop()

if df_raw.empty:
    st.error("Data file loaded but is empty.")
    st.stop()

# ============================================================
# MAP YOUR CSV COLUMNS
# ============================================================
# This matches your uploaded file headers exactly.
COL_TICKER = "Symbol"
COL_NAME = "Description"
COL_PRICE = "Price"

COL_R_1D = "Price Change % 1 day"
COL_R_1W = "Performance % 1 week"
COL_R_1M = "Performance % 1 month"
COL_R_3M = "Performance % 3 months"
COL_R_6M = "Performance % 6 months"
COL_R_1Y = "Performance % 1 year"

COL_VOLCHG_1D = "Volume Change % 1 day"
COL_VOLCHG_1W = "Volume Change % 1 week"
COL_VOLCHG_1M = "Volume Change % 1 month"

COL_RVOL_1D = "Relative Volume 1 day"
COL_RVOL_1W = "Relative Volume 1 week"
COL_RVOL_1M = "Relative Volume 1 month"

COL_EPS_Q = "Earnings per share diluted growth %, Quarterly YoY"
COL_EPS_A = "Earnings per share diluted growth %, Annual YoY"
COL_REV_Q = "Revenue growth %, Quarterly YoY"
COL_REV_A = "Revenue growth %, Annual YoY"

COL_ROE = "Return on equity %, Trailing 12 months"
COL_PRETAX = "Pretax margin %, Trailing 12 months"

COL_52W_HIGH = "High 52 weeks"

# Optional (only if you add later)
OPT_ATR = None
OPT_ADR = None
for c in df_raw.columns:
    cl = str(c).lower()
    if "atr" in cl:
        OPT_ATR = c
    if "adr" in cl:
        OPT_ADR = c

# ============================================================
# BUILD NORMALIZED FRAME
# ============================================================
df = df_raw.copy()
df["Ticker"] = df[COL_TICKER].astype(str).map(normalize_ticker)
df["Name"] = df[COL_NAME].astype(str) if COL_NAME in df.columns else ""
df["Price"] = pd.to_numeric(df[COL_PRICE], errors="coerce") if COL_PRICE in df.columns else np.nan

# returns from CSV -> fractional
df["r_1d"] = to_float_pct_series(df[COL_R_1D]) if COL_R_1D in df.columns else np.nan
df["r_1w"] = to_float_pct_series(df[COL_R_1W]) if COL_R_1W in df.columns else np.nan
df["r_1m"] = to_float_pct_series(df[COL_R_1M]) if COL_R_1M in df.columns else np.nan
df["r_3m"] = to_float_pct_series(df[COL_R_3M]) if COL_R_3M in df.columns else np.nan
df["r_6m"] = to_float_pct_series(df[COL_R_6M]) if COL_R_6M in df.columns else np.nan
df["r_1y"] = to_float_pct_series(df[COL_R_1Y]) if COL_R_1Y in df.columns else np.nan

# live SPY benchmark returns
spy_ret = fetch_spy_returns()
b_1d, b_1w, b_1m, b_3m, b_6m, b_1y = (
    spy_ret.get("1D", np.nan),
    spy_ret.get("1W", np.nan),
    spy_ret.get("1M", np.nan),
    spy_ret.get("3M", np.nan),
    spy_ret.get("6M", np.nan),
    spy_ret.get("1Y", np.nan),
)

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
df["RS 3M"] = to_rs_1_99(df["rr_3m"])
df["RS 6M"] = to_rs_1_99(df["rr_6m"])
df["RS 1Y"] = to_rs_1_99(df["rr_1y"])

# display %s
df["% 1D"] = df["r_1d"]
df["% 1W"] = df["r_1w"]
df["% 1M"] = df["r_1m"]
df["% 3M"] = df["r_3m"]
df["% 6M"] = df["r_6m"]
df["% 1Y"] = df["r_1y"]

# volume change + rvol
def safe_num(col):
    return pd.to_numeric(df_raw[col], errors="coerce") if col in df_raw.columns else np.nan

df["VolChg% 1D"] = safe_num(COL_VOLCHG_1D) / 100.0 if COL_VOLCHG_1D in df_raw.columns else np.nan
df["VolChg% 1W"] = safe_num(COL_VOLCHG_1W) / 100.0 if COL_VOLCHG_1W in df_raw.columns else np.nan
df["VolChg% 1M"] = safe_num(COL_VOLCHG_1M) / 100.0 if COL_VOLCHG_1M in df_raw.columns else np.nan

df["RVOL 1D"] = safe_num(COL_RVOL_1D)
df["RVOL 1W"] = safe_num(COL_RVOL_1W)
df["RVOL 1M"] = safe_num(COL_RVOL_1M)

# fundamentals
df["EPS% Q YoY"] = safe_num(COL_EPS_Q) / 100.0 if COL_EPS_Q in df_raw.columns else np.nan
df["EPS% A YoY"] = safe_num(COL_EPS_A) / 100.0 if COL_EPS_A in df_raw.columns else np.nan
df["REV% Q YoY"] = safe_num(COL_REV_Q) / 100.0 if COL_REV_Q in df_raw.columns else np.nan
df["REV% A YoY"] = safe_num(COL_REV_A) / 100.0 if COL_REV_A in df_raw.columns else np.nan

df["ROE% TTM"] = safe_num(COL_ROE) / 100.0 if COL_ROE in df_raw.columns else np.nan
df["Pretax% TTM"] = safe_num(COL_PRETAX) / 100.0 if COL_PRETAX in df_raw.columns else np.nan

# distance from 52w high
df["High 52W"] = pd.to_numeric(df_raw[COL_52W_HIGH], errors="coerce") if COL_52W_HIGH in df_raw.columns else np.nan
df["% From 52W High"] = np.where(
    (np.isfinite(df["Price"]) & np.isfinite(df["High 52W"]) & (df["High 52W"] > 0)),
    (df["High 52W"] - df["Price"]) / df["High 52W"],
    np.nan,
)

# optional ATR/ADR
if OPT_ATR:
    df["ATR%"] = to_float_pct_series(df_raw[OPT_ATR])
if OPT_ADR:
    df["ADR%"] = to_float_pct_series(df_raw[OPT_ADR])


# ============================================================
# UI (HEADER)
# ============================================================
st.title("Relative Strength Stock Scanner")
st.caption(f"As of: {_asof_ts()} • RS Benchmark: {BENCHMARK} (live via yfinance)")


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.subheader("Scanner Controls")

    rs_cols_order = ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y", "RS 1D"]
    available_rs = [c for c in rs_cols_order if c in df.columns]

    primary_tf = st.selectbox(
        "Rank by Timeframe (Top by RS)",
        available_rs,
        index=1 if "RS 1M" in available_rs else 0,
        key="primary_tf",
    )

    rs_min = st.slider("Minimum RS Rating (Primary)", 1, 99, 90, 1, key="rs_min")

    mode = st.selectbox(
        "Scan Mode",
        [
            "Primary timeframe only",
            "All selected timeframes >= threshold",
            "Accelerating (1Y→6M→3M→1M improving)",
            "Decelerating (1M→3M→6M→1Y weakening)",
        ],
        index=0,
        key="mode",
    )

    # Choose which timeframes are enforced for the "All selected" mode
    enforce_tfs = st.multiselect(
        "Timeframes to Enforce (for modes that need it)",
        ["RS 1Y", "RS 6M", "RS 3M", "RS 1M", "RS 1W", "RS 1D"],
        default=["RS 1Y", "RS 6M", "RS 3M", "RS 1M"],
        key="enforce_tfs",
        help="Use this for the 'All selected timeframes' mode. For Accel/Decel we use only 1Y/6M/3M/1M.",
    )

    # For acceleration/deceleration: allow deselecting among 1Y/6M/3M/1M only
    accel_set = st.multiselect(
        "Accel/Decel Timeframes (exclude 1W/1D)",
        ["RS 1Y", "RS 6M", "RS 3M", "RS 1M"],
        default=["RS 1Y", "RS 6M", "RS 3M", "RS 1M"],
        key="accel_set",
        help="This controls which horizons participate in Accel/Decel. (No 1W/1D.)",
    )

    st.markdown("---")
    st.subheader("Extra Filters")

    # RVOL filters
    min_rvol_1d = st.number_input("Min RVOL 1D", value=0.0, step=0.1, key="min_rvol_1d")
    min_rvol_1w = st.number_input("Min RVOL 1W", value=0.0, step=0.1, key="min_rvol_1w")
    min_rvol_1m = st.number_input("Min RVOL 1M", value=0.0, step=0.1, key="min_rvol_1m")

    # Volume change %
    min_volchg_1d = st.number_input("Min Vol Change % 1D", value=-999.0, step=5.0, key="min_volchg_1d") / 100.0
    min_volchg_1w = st.number_input("Min Vol Change % 1W", value=-999.0, step=5.0, key="min_volchg_1w") / 100.0
    min_volchg_1m = st.number_input("Min Vol Change % 1M", value=-999.0, step=5.0, key="min_volchg_1m") / 100.0

    # Fundamental growth filters
    min_eps_q = st.number_input("Min EPS % Quarterly YoY", value=-999.0, step=10.0, key="min_eps_q") / 100.0
    min_eps_a = st.number_input("Min EPS % Annual YoY", value=-999.0, step=10.0, key="min_eps_a") / 100.0
    min_rev_q = st.number_input("Min Revenue % Quarterly YoY", value=-999.0, step=10.0, key="min_rev_q") / 100.0
    min_rev_a = st.number_input("Min Revenue % Annual YoY", value=-999.0, step=10.0, key="min_rev_a") / 100.0

    min_roe = st.number_input("Min ROE % (TTM)", value=-999.0, step=5.0, key="min_roe") / 100.0
    min_pretax = st.number_input("Min Pretax Margin % (TTM)", value=-999.0, step=5.0, key="min_pretax") / 100.0

    # 52W High distance
    max_dist_52w = st.slider("Max % Below 52W High", 0, 80, 20, 1, key="max_dist_52w") / 100.0

    # Optional ADR/ATR filters (only show if columns exist)
    if "ADR%" in df.columns:
        min_adr = st.number_input("Min ADR %", value=0.0, step=0.1, key="min_adr") / 100.0
    else:
        min_adr = None

    if "ATR%" in df.columns:
        min_atr = st.number_input("Min ATR %", value=0.0, step=0.1, key="min_atr") / 100.0
    else:
        min_atr = None

    st.markdown("---")
    max_results = st.slider("Max Results", 25, 2000, 200, step=25, key="max_results")


# ============================================================
# SCAN LOGIC
# ============================================================
df_out = df.copy()

# Apply extra filters (only if columns exist / finite)
def apply_min(series: pd.Series, min_val: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s >= min_val


# RVOL
if "RVOL 1D" in df_out.columns and min_rvol_1d > 0:
    df_out = df_out[apply_min(df_out["RVOL 1D"], min_rvol_1d)]
if "RVOL 1W" in df_out.columns and min_rvol_1w > 0:
    df_out = df_out[apply_min(df_out["RVOL 1W"], min_rvol_1w)]
if "RVOL 1M" in df_out.columns and min_rvol_1m > 0:
    df_out = df_out[apply_min(df_out["RVOL 1M"], min_rvol_1m)]

# Vol change % (stored as fractional)
if "VolChg% 1D" in df_out.columns:
    df_out = df_out[df_out["VolChg% 1D"].fillna(-np.inf) >= min_volchg_1d]
if "VolChg% 1W" in df_out.columns:
    df_out = df_out[df_out["VolChg% 1W"].fillna(-np.inf) >= min_volchg_1w]
if "VolChg% 1M" in df_out.columns:
    df_out = df_out[df_out["VolChg% 1M"].fillna(-np.inf) >= min_volchg_1m]

# Fundamentals (stored as fractional)
for col, minv in [
    ("EPS% Q YoY", min_eps_q),
    ("EPS% A YoY", min_eps_a),
    ("REV% Q YoY", min_rev_q),
    ("REV% A YoY", min_rev_a),
    ("ROE% TTM", min_roe),
    ("Pretax% TTM", min_pretax),
]:
    if col in df_out.columns:
        df_out = df_out[df_out[col].fillna(-np.inf) >= minv]

# 52w distance: keep names within max distance below 52W high
if "% From 52W High" in df_out.columns:
    df_out = df_out[df_out["% From 52W High"].fillna(np.inf) <= max_dist_52w]

# Optional ADR/ATR
if min_adr is not None and "ADR%" in df_out.columns:
    df_out = df_out[df_out["ADR%"].fillna(-np.inf) >= min_adr]
if min_atr is not None and "ATR%" in df_out.columns:
    df_out = df_out[df_out["ATR%"].fillna(-np.inf) >= min_atr]


# RS modes
rs_cols_present = [c for c in ["RS 1D", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"] if c in df_out.columns]

if mode == "Primary timeframe only":
    df_f = df_out[df_out[primary_tf] >= rs_min].copy()

elif mode == "All selected timeframes >= threshold":
    enforce = [c for c in enforce_tfs if c in df_out.columns]
    if len(enforce) == 0:
        df_f = df_out[df_out[primary_tf] >= rs_min].copy()
    else:
        cond = True
        for c in enforce:
            cond = cond & (df_out[c] >= rs_min)
        df_f = df_out[cond].copy()

elif mode.startswith("Accelerating"):
    # accelerating: long -> short improving (example: 1Y 55, 6M 63, 3M 77, 1M 80)
    order = ["RS 1Y", "RS 6M", "RS 3M", "RS 1M"]
    order = [c for c in order if c in df_out.columns and c in accel_set]
    if len(order) < 2:
        df_f = df_out[df_out[primary_tf] >= rs_min].copy()
    else:
        cond = (df_out[primary_tf] >= rs_min)
        for a, b in zip(order[:-1], order[1:]):
            cond = cond & (df_out[b] >= df_out[a])
        df_f = df_out[cond].copy()

else:
    # decelerating: short -> long weakening (example: 1M 72, 3M 80, 6M 87, 1Y 99)
    order = ["RS 1M", "RS 3M", "RS 6M", "RS 1Y"]
    # reuse same accel_set but reversed membership
    allow = set(accel_set)
    order = [c for c in order if c in df_out.columns and c in allow]
    if len(order) < 2:
        df_f = df_out[df_out[primary_tf] >= rs_min].copy()
    else:
        cond = (df_out[primary_tf] >= rs_min)
        for a, b in zip(order[:-1], order[1:]):
            cond = cond & (df_out[a] <= df_out[b])
        df_f = df_out[cond].copy()

# sort by chosen RS timeframe, tiebreaker by RS 1Y if present
tie = "RS 1Y" if "RS 1Y" in df_f.columns else primary_tf
df_f = df_f.sort_values([primary_tf, tie], ascending=[False, False])

# ============================================================
# OUTPUT TABLE
# ============================================================
st.markdown('<div class="section-title">Scanner Results</div>', unsafe_allow_html=True)
st.markdown(f'<div class="small-muted">Matches: <b>{len(df_f):,}</b></div>', unsafe_allow_html=True)

show_cols = [
    "Ticker",
    "Name",
    "Price",
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
    "RVOL 1D",
    "RVOL 1W",
    "RVOL 1M",
    "VolChg% 1D",
    "VolChg% 1W",
    "VolChg% 1M",
    "EPS% Q YoY",
    "EPS% A YoY",
    "REV% Q YoY",
    "REV% A YoY",
    "ROE% TTM",
    "Pretax% TTM",
    "% From 52W High",
]

# add optional columns if present
if "ADR%" in df_f.columns:
    show_cols.append("ADR%")
if "ATR%" in df_f.columns:
    show_cols.append("ATR%")

for c in show_cols:
    if c not in df_f.columns:
        df_f[c] = np.nan

render_table_html(df_f[show_cols].head(max_results), show_cols, height_px=950)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
**How RS is Calculated (vs SPY)**  
- Stock returns come from your CSV performance columns (1W/1M/3M/6M/1Y and 1D price change).  
- SPY returns are pulled live from yfinance and computed on trading-day approximations (5/21/63/126/252).  
- Relative return: **RR = (1 + r_stock) / (1 + r_SPY) − 1**  
- RS is the percentile-rank of RR across your universe mapped to **1–99**.  
"""
)
