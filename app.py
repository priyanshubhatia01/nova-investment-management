import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import nova_core as nc

st.set_page_config(page_title="NOVA | Investment Management", page_icon="🧭", layout="wide")

# ------------------------------------------------------------------ look & feel
INK, PANEL, EDGE = "#0b1115", "#121c22", "#223540"
TEAL, AMBER, CORAL, MUTED, TEXT = "#2dd4bf", "#fbbf24", "#fb7185", "#8aa4ad", "#e6f1f2"
BUCKET_COLOR = {"Large": TEAL, "Mid": AMBER, "Small": CORAL}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
.stApp {{background: radial-gradient(1200px 600px at 8% -10%, #12333a 0%, {INK} 55%); color:{TEXT}; font-family:'Inter',sans-serif;}}
h1,h2,h3,h4 {{font-family:'Space Grotesk',sans-serif; color:{TEXT}; letter-spacing:-.01em;}}
[data-testid="stMetric"] {{background:{PANEL}; border:1px solid {EDGE}; border-left:3px solid {TEAL}; padding:14px 16px; border-radius:10px;}}
[data-testid="stMetricLabel"] {{color:{MUTED};}}
[data-testid="stSidebar"] {{background:#0e171c; border-right:1px solid {EDGE};}}
.hero {{padding:6px 0 2px 0;}}
.hero .brand {{font-family:'Space Grotesk'; font-weight:700; font-size:2.6rem; letter-spacing:.18em; color:{TEXT};}}
.hero .brand span {{color:{TEAL};}}
.hero .tag {{color:{MUTED}; font-size:.85rem; letter-spacing:.12em; text-transform:uppercase;}}
.tape {{display:flex; gap:8px; overflow-x:auto; padding:8px 0 12px 0;}}
.chip {{white-space:nowrap; background:{PANEL}; border:1px solid {EDGE}; border-radius:999px; padding:5px 12px; font-size:.8rem;}}
.chip b {{font-weight:600;}} .up {{color:{TEAL};}} .down {{color:{CORAL};}}
.pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; color:{INK};}}
.card {{background:{PANEL}; border:1px solid {EDGE}; border-radius:12px; padding:14px 18px;}}
.small {{color:{MUTED}; font-size:.82rem;}}
</style>
""", unsafe_allow_html=True)

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=TEXT), margin=dict(l=10, r=10, t=45, b=10))


def style(fig, **kw):
    fig.update_layout(**{**PLOT, **kw})
    return fig


def inr(v, short=True):
    if v is None or not np.isfinite(v):
        return "—"
    if short and abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if short and abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.0f}"


def pct(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%"


def short(sym):
    return sym.replace(".NS", "")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🧭 Control room")
    chosen = {}
    for bucket, syms in nc.UNIVERSE.items():
        chosen[bucket] = st.multiselect(
            f"{bucket} caps", syms, default=syms, format_func=short, key=f"uni_{bucket}")
    selected = [s for b in nc.UNIVERSE for s in chosen[b]]

    window_label = st.select_slider("Analysis window", ["1y", "2y", "3y", "5y"], value="3y",
                                    help="Used for optimisation, backtests, correlations and the goal simulator.")
    rf = st.slider("Risk-free rate (% p.a.)", 0.0, 10.0, 7.0, 0.25) / 100
    fee_bps = st.slider("Trading cost (bps per trade)", 0, 50, 10,
                        help="Applied to every virtual buy/sell. 10 bps = 0.10%.")
    if st.button("↻ Refresh market data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### Ranking weights")
    st.caption("Illustrative. Justify your own weights in the report.")
    w_size = st.slider("Size", 0, 50, 15)
    w_growth = st.slider("Growth (sales + EPS)", 0, 50, 25)
    w_profit = st.slider("Profitability (ROE, ROCE)", 0, 50, 20)
    w_value = st.slider("Valuation", 0, 50, 15)
    w_risk = st.slider("Risk", 0, 50, 15)
    w_tech = st.slider("Technicals", 0, 50, 10)

if len(selected) < 3:
    st.error("Select at least 3 stocks in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ data
def _roce(obj):
    """Approximate ROCE = EBIT / (Total assets - Current liabilities), latest annual statements."""
    try:
        fin, bs = obj.financials, obj.balance_sheet
        ebit = float(fin.loc["EBIT"].dropna().iloc[0])
        assets = float(bs.loc["Total Assets"].dropna().iloc[0])
        cur_liab = float(bs.loc["Current Liabilities"].dropna().iloc[0])
        cap_emp = assets - cur_liab
        return ebit / cap_emp * 100 if cap_emp > 0 else np.nan
    except Exception:
        return np.nan


def _clean(hist):
    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    return hist[~hist.index.duplicated(keep="last")]


@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbols: tuple):
    ohlc, rows, failed = {}, [], []
    for sym in symbols:
        try:
            obj = yf.Ticker(sym)
            hist = obj.history(period="5y", auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist:
                failed.append(sym)
                continue
            hist = _clean(hist)
            ohlc[sym] = hist
            try:
                info = obj.info or {}
            except Exception:
                info = {}
            roe = info.get("returnOnEquity")
            growth = info.get("revenueGrowth")
            rows.append({
                "Ticker": sym,
                "Company": info.get("shortName", short(sym)),
                "Sector": info.get("sector", "—"),
                "Market cap (₹)": info.get("marketCap", np.nan),
                "ROE (%)": roe * 100 if roe is not None else np.nan,
                "ROCE (%)": np.nan if sym in nc.FINANCIALS else _roce(obj),
                "P/E": info.get("trailingPE", np.nan),
                "PEG": info.get("pegRatio", np.nan),
                "P/BV": info.get("priceToBook", np.nan),
                "EPS": info.get("trailingEps", np.nan),
                "Sales growth YoY (%)": growth * 100 if growth is not None else np.nan,
                "Beta": info.get("beta", np.nan),
            })
        except Exception:
            failed.append(sym)
    bench = pd.Series(dtype=float)
    try:
        b = yf.Ticker(nc.BENCHMARK).history(period="5y", auto_adjust=True)
        bench = _clean(b)["Close"].rename("Nifty 50")
    except Exception:
        pass
    closes = pd.DataFrame({s: h["Close"] for s, h in ohlc.items()}).sort_index().ffill(limit=3)
    return closes, ohlc, pd.DataFrame(rows), bench, failed, datetime.now().strftime("%d %b %Y, %H:%M")


with st.spinner("Pulling market data from Yahoo Finance…"):
    closes_all, ohlc, fund, bench, failed, fetched_at = load_data(tuple(selected))

if closes_all.empty or fund.empty:
    st.error("No price data came back. Check the internet connection, then press Refresh market data.")
    st.stop()
if failed:
    st.warning("No data for: " + ", ".join(short(f) for f in failed) + ". They are left out.")

avail = list(closes_all.columns)
years_back = int(window_label[0])
cut = closes_all.index[-1] - pd.DateOffset(years=years_back)
closes = closes_all[closes_all.index >= cut]
returns_full = closes.pct_change()
returns = returns_full.dropna()  # common window across all selected stocks

# ------------------------------------------------------------------ factsheet + scores
fund = fund.copy()
fund["Bucket"] = fund["Ticker"].map(nc.BUCKET_OF)
last = closes_all.ffill().iloc[-1]
prev = closes_all.ffill().iloc[-2]
fund["Price (₹)"] = fund["Ticker"].map(last)
fund["Day change (%)"] = fund["Ticker"].map((last / prev - 1) * 100)
fund["1Y return (%)"] = fund["Ticker"].map(
    closes_all.ffill().apply(lambda c: (c.iloc[-1] / c.iloc[-252] - 1) * 100 if c.dropna().shape[0] > 252 else np.nan))
vol_ann = returns_full.std() * np.sqrt(nc.TRADING_DAYS) * 100
fund["Volatility (%)"] = fund["Ticker"].map(vol_ann)
ma50 = closes_all.rolling(50).mean().iloc[-1]
ma200 = closes_all.rolling(200).mean().iloc[-1]
fund["vs 50-day MA"] = fund["Ticker"].map(last / ma50 - 1)
fund["vs 200-day MA"] = fund["Ticker"].map(last / ma200 - 1)


def pscore(s, higher=True):
    return pd.to_numeric(s, errors="coerce").rank(pct=True, ascending=higher).fillna(0.5) * 100


fund["S size"] = pscore(fund["Market cap (₹)"])
fund["S growth"] = (pscore(fund["Sales growth YoY (%)"]) + pscore(fund["EPS"])) / 2
fund["S profit"] = (pscore(fund["ROE (%)"]) + pscore(fund["ROCE (%)"])) / 2
fund["S value"] = (pscore(fund["P/E"], False) + pscore(fund["PEG"], False) + pscore(fund["P/BV"], False)) / 3
fund["S risk"] = (pscore(fund["Beta"], False) + pscore(fund["Volatility (%)"], False)) / 2
fund["S tech"] = ((fund["vs 50-day MA"] > 0).astype(float) * 50 + (fund["vs 200-day MA"] > 0).astype(float) * 50)
wts = np.array([w_size, w_growth, w_profit, w_value, w_risk, w_tech], dtype=float)
wts = wts / wts.sum() if wts.sum() else np.ones(6) / 6
fund["Score"] = (fund[["S size", "S growth", "S profit", "S value", "S risk", "S tech"]].values * wts).sum(axis=1)
fund = fund.sort_values("Score", ascending=False).reset_index(drop=True)
score_map = dict(zip(fund["Ticker"], fund["Score"]))
price_map = dict(zip(fund["Ticker"], fund["Price (₹)"]))

# ------------------------------------------------------------------ virtual account
START = 10_000_000.0
for k, v in {"cash": START, "holdings": {}, "cost": {}, "ledger": [], "realised": 0.0,
             "fees": 0.0, "plan": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def trade(side, sym, qty):
    px_ = float(price_map[sym])
    value = qty * px_
    fee = value * fee_bps / 10_000
    h, c = st.session_state.holdings, st.session_state.cost
    if side == "BUY":
        if value + fee > st.session_state.cash + 1e-6:
            return False, f"Not enough cash for {qty} × {short(sym)} (needs {inr(value + fee, False)})."
        st.session_state.cash -= value + fee
        h[sym] = h.get(sym, 0) + qty
        c[sym] = c.get(sym, 0.0) + value + fee
    else:
        own = h.get(sym, 0)
        if qty > own:
            return False, f"You hold only {own} share(s) of {short(sym)}."
        avg = c.get(sym, 0.0) / own if own else 0.0
        st.session_state.cash += value - fee
        st.session_state.realised += (value - fee) - avg * qty
        c[sym] = c.get(sym, 0.0) - avg * qty
        h[sym] = own - qty
        if h[sym] == 0:
            h.pop(sym, None)
            c.pop(sym, None)
    st.session_state.fees += fee
    st.session_state.ledger.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Action": side,
                                    "Ticker": sym, "Qty": int(qty), "Price (₹)": round(px_, 2),
                                    "Value (₹)": round(value, 2), "Fee (₹)": round(fee, 2)})
    return True, f"{side} {qty} × {short(sym)} at ₹{px_:,.2f}"


holdings = st.session_state.holdings
invested = sum(q * price_map.get(s, 0) for s, q in holdings.items())
cost_basis = sum(st.session_state.cost.values())
portfolio_value = st.session_state.cash + invested
unrealised = invested - cost_basis

with st.sidebar:
    st.divider()
    if st.button("Reset virtual account to ₹1 crore"):
        for k in ("holdings", "cost", "ledger"):
            st.session_state[k] = {} if k != "ledger" else []
        st.session_state.cash, st.session_state.realised = START, 0.0
        st.session_state.fees, st.session_state.plan = 0.0, None
        st.rerun()

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero"><div class="brand">NO<span>V</span>A</div>
<div class="tag">Investment management · research desk + virtual ₹1 crore book</div></div>
""", unsafe_allow_html=True)

chips = "".join(
    f"<div class='chip'><b>{short(t)}</b> ₹{p:,.1f} <span class='{'up' if d >= 0 else 'down'}'>{d:+.2f}%</span></div>"
    for t, p, d in zip(fund["Ticker"], fund["Price (₹)"], fund["Day change (%)"])
)
st.markdown(f"<div class='tape'>{chips}</div>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Book value", inr(portfolio_value), f"{(portfolio_value / START - 1) * 100:+.2f}%")
m2.metric("Cash", inr(st.session_state.cash))
m3.metric("Invested", inr(invested))
m4.metric("Unrealised P/L", inr(unrealised))
m5.metric("Realised P/L", inr(st.session_state.realised))
st.caption(f"Data fetched {fetched_at} (cached up to 15 min) · Yahoo Finance via yfinance, may be delayed · "
           f"Student simulator, not investment advice.")

tabs = st.tabs(["🏠 Command centre", "🔎 Screener", "🔬 Stock lab", "🧩 Portfolio builder",
                "⏪ Backtest & risk", "🎯 Goal simulator", "💹 Trade desk", "📘 Method"])
t_home, t_scr, t_lab, t_build, t_bt, t_goal, t_trade, t_method = tabs

# ================================================================== HOME
with t_home:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("Universe map")
        tm = fund.copy()
        tm["Size"] = tm["Market cap (₹)"].fillna(tm["Market cap (₹)"].median()).fillna(1.0)
        tm["Name"] = tm["Ticker"].map(short)
        fig = px.treemap(tm, path=[px.Constant("NOVA universe"), "Bucket", "Name"], values="Size",
                         color="Day change (%)", color_continuous_scale=[CORAL, "#1b2a31", TEAL],
                         color_continuous_midpoint=0)
        fig.update_traces(texttemplate="%{label}<br>%{color:+.2f}%",
                          hovertemplate="%{label}<br>Day change %{color:+.2f}%<extra></extra>")
        st.plotly_chart(style(fig, height=430, coloraxis_showscale=False))
        st.caption("Rectangle size = market capitalisation. Colour = today's move.")
    with c2:
        st.subheader("Book allocation")
        alloc = [{"Item": short(s), "Value": q * price_map.get(s, 0), "Bucket": nc.BUCKET_OF.get(s, "")}
                 for s, q in holdings.items()]
        alloc.append({"Item": "Cash", "Value": st.session_state.cash, "Bucket": "Cash"})
        adf = pd.DataFrame(alloc)
        fig = px.pie(adf, names="Item", values="Value", hole=.62, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(style(fig, height=300, showlegend=False))
        bk = adf.groupby("Bucket")["Value"].sum().reset_index()
        bk["Share"] = bk["Value"] / bk["Value"].sum() * 100
        fig = px.bar(bk, x="Share", y="Bucket", orientation="h", color="Bucket",
                     color_discrete_map={**BUCKET_COLOR, "Cash": MUTED})
        st.plotly_chart(style(fig, height=170, showlegend=False, title="Exposure by size bucket (%)"))
    st.subheader("Top-ranked right now")
    st.dataframe(fund[["Ticker", "Company", "Bucket", "Price (₹)", "1Y return (%)", "Score"]].head(6)
                 .style.format(precision=2, na_rep="—"), hide_index=True)

# ================================================================== SCREENER
with t_scr:
    st.subheader("Screener")
    f1, f2, f3 = st.columns(3)
    sel_b = f1.multiselect("Size bucket", list(nc.UNIVERSE), default=list(nc.UNIVERSE))
    min_score = f2.slider("Minimum composite score", 0, 100, 0)
    max_pe = f3.number_input("Maximum P/E (0 = no limit)", min_value=0.0, value=0.0, step=5.0)
    view = fund[fund["Bucket"].isin(sel_b) & (fund["Score"] >= min_score)]
    if max_pe > 0:
        view = view[(view["P/E"] <= max_pe) | view["P/E"].isna()]
    cols = ["Ticker", "Company", "Sector", "Bucket", "Price (₹)", "Market cap (₹)", "ROE (%)", "ROCE (%)", "P/E",
            "PEG", "P/BV", "EPS", "Sales growth YoY (%)", "Beta", "Volatility (%)", "vs 200-day MA", "Score"]
    st.dataframe(view[cols].style.format(precision=2, na_rep="—"), hide_index=True)
    st.download_button("⬇ Download screener (CSV)", view[cols].to_csv(index=False).encode(), "nova_screener.csv")
    sv = view.sort_values("Score").assign(Name=lambda d: d["Ticker"].map(short))
    fig = px.bar(sv, x="Score", y="Name", orientation="h", color="Bucket", color_discrete_map=BUCKET_COLOR)
    st.plotly_chart(style(fig, height=140 + 28 * len(view), yaxis_title=None, title="Composite score"))
    fac = {"S size": "Size", "S growth": "Growth", "S profit": "Profit", "S value": "Value",
           "S risk": "Risk", "S tech": "Technicals"}
    hm = view.set_index(view["Ticker"].map(short))[list(fac)].rename(columns=fac)
    fig = go.Figure(go.Heatmap(z=hm.values, x=list(hm.columns), y=list(hm.index), zmin=0, zmax=100,
                               colorscale=[[0, "#12242b"], [1, TEAL]], text=np.round(hm.values, 0),
                               texttemplate="%{text}"))
    st.plotly_chart(style(fig, height=140 + 28 * len(view), yaxis_autorange="reversed",
                          title="Where each stock scores well (0–100)"))
    st.caption("ROCE is approximated as EBIT ÷ (total assets − current liabilities) from the latest annual statements; "
               "it is left blank for banks/NBFCs (HDFCBANK, SUNDARMFIN, KARURVYSYA), where it is not meaningful. "
               "Blank values receive a neutral score of 50.")

# ================================================================== STOCK LAB
with t_lab:
    st.subheader("Stock lab")
    a, b, c = st.columns([1, 1, 2])
    sym = a.selectbox("Stock", avail, format_func=short)
    chart_kind = b.radio("Chart", ["Candles", "Line"], horizontal=True)
    span = c.select_slider("Span", ["3mo", "6mo", "1y", "2y", "5y"], value="1y")
    days = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}[span]
    h = ohlc[sym].tail(days)
    full = ohlc[sym]["Close"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.04)
    if chart_kind == "Candles":
        fig.add_trace(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                     name="Price", increasing_line_color=TEAL, decreasing_line_color=CORAL), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=h.index, y=h["Close"], name="Price", line=dict(color=TEAL, width=2)), 1, 1)
    for n_, col in ((20, AMBER), (50, "#a78bfa"), (200, "#60a5fa")):
        ma = full.rolling(n_).mean().reindex(h.index)
        fig.add_trace(go.Scatter(x=h.index, y=ma, name=f"MA{n_}", line=dict(width=1.2, color=col)), 1, 1)
    r_ = nc.rsi(full).reindex(h.index)
    fig.add_trace(go.Scatter(x=h.index, y=r_, name="RSI(14)", line=dict(color=AMBER)), 2, 1)
    fig.add_hline(y=70, line_dash="dot", line_color=CORAL, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=TEAL, row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(style(fig, height=560, title=f"{short(sym)} · {fund.set_index('Ticker').loc[sym, 'Company']}"))

    hi, lo = full.tail(252).max(), full.tail(252).min()
    pos = (full.iloc[-1] - lo) / (hi - lo) if hi > lo else np.nan
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Price", f"₹{full.iloc[-1]:,.2f}")
    k2.metric("52-week range position", pct(pos, 0), help="0% = at the 52-week low, 100% = at the high")
    for kcol, lab, n_ in ((k3, "1M", 21), (k4, "6M", 126), (k5, "1Y", 252)):
        kcol.metric(f"{lab} return", pct(full.iloc[-1] / full.iloc[-n_ - 1] - 1) if len(full) > n_ else "—")

    st.markdown("##### Compare with peers and the Nifty 50")
    peers = st.multiselect("Add stocks", [s for s in avail if s != sym], default=[], format_func=short)
    cmp_days = days
    lines = {short(s): closes_all[s].tail(cmp_days) for s in [sym] + peers}
    if not bench.empty:
        lines["Nifty 50"] = bench.tail(cmp_days)
    fig = go.Figure()
    for name, ser in lines.items():
        ser = ser.dropna()
        fig.add_trace(go.Scatter(x=ser.index, y=ser / ser.iloc[0] * 100, name=name,
                                 line=dict(width=3 if name == short(sym) else 1.6,
                                           dash="dot" if name == "Nifty 50" else "solid")))
    st.plotly_chart(style(fig, height=360, title="Rebased to 100", yaxis_title=None))

# ================================================================== PORTFOLIO BUILDER
with t_build:
    st.subheader("Portfolio builder")
    if returns.shape[0] < 120 or returns.shape[1] < 3:
        st.warning("Not enough overlapping history for the selected stocks and window. Try a shorter window or "
                   "different stocks.")
        weights = None
        st.session_state["weights"] = None
    else:
        if returns.index[0] > cut + pd.Timedelta(days=45):
            st.info(f"The newest listing limits the common history to {returns.index[0]:%d %b %Y} onward "
                    f"({returns.shape[0]} trading days).")
        left, right = st.columns([1, 1.6])
        with left:
            st.markdown("##### 1 · Risk profile")
            answers = []
            for i, (q, opts) in enumerate(nc.QUIZ):
                answers.append(opts.index(st.radio(q, opts, index=1, key=f"quiz{i}")))
            prof_name, prof_limits, prof_score = nc.profile_from_answers(answers)
            st.markdown(f"<div class='card'>Your profile: <b>{prof_name}</b> "
                        f"<span class='small'>(score {prof_score}/{3 * len(nc.QUIZ)})</span></div>",
                        unsafe_allow_html=True)
        with right:
            st.markdown("##### 2 · Construction rules")
            r1, r2 = st.columns(2)
            method = r1.selectbox("Method", ["Maximum Sharpe", "Minimum variance", "Risk parity",
                                             "Score-tilted", "Equal weight"])
            max_w = r2.slider("Max weight per stock (%)", 8, 50, 20) / 100
            st.caption("Size-bucket limits (start from your profile, adjust freely)")
            limits = {}
            bcols = st.columns(3)
            for col, bucket in zip(bcols, nc.UNIVERSE):
                lo_, hi_ = prof_limits[bucket]
                v = col.slider(f"{bucket} (%)", 0, 100, (int(lo_ * 100), int(hi_ * 100)),
                               key=f"lim_{bucket}_{prof_name}")
                limits[bucket] = (v[0] / 100, v[1] / 100)

        cols_ok = list(returns.columns)
        in_bucket = {b: any(nc.BUCKET_OF.get(t) == b for t in cols_ok) for b in nc.UNIVERSE}
        limits = {b: l for b, l in limits.items() if in_bucket[b]}
        if sum(l[0] for l in limits.values()) > 1 or sum(l[1] for l in limits.values()) < 1:
            st.error("Bucket limits cannot add up to 100%. Raise the maxima or lower the minimums.")
            weights = None
            st.session_state["weights"] = None
        else:
            n_in = {b: sum(nc.BUCKET_OF.get(t) == b for t in cols_ok) for b in limits}
            capacity_short = [b for b in limits if n_in[b] * max_w < limits[b][0] - 1e-9]
            weights = nc.optimise(method, returns, rf=rf, max_w=max_w, bucket_limits=limits, scores=score_map)
            if method == "Equal weight":
                st.caption("Equal weight ignores the size-bucket limits and the per-stock cap.")
            else:
                bsum = {b: float(sum(weights[t] for t in cols_ok if nc.BUCKET_OF.get(t) == b)) for b in limits}
                broke = [b for b in limits if not (limits[b][0] - .005 <= bsum[b] <= limits[b][1] + .005)]
                if capacity_short or broke or weights.max() > max_w + .005:
                    st.warning("These rules cannot all be satisfied at once (for example a bucket minimum that "
                               "the per-stock cap makes unreachable). The result below falls back to equal weights. "
                               "Loosen the limits or raise the cap.")
            mu, cov = nc.annual_stats(returns)
            ret_, vol_, shp_ = nc.port_perf(weights.values, mu, cov, rf)
            eff_n = 1 / float((weights ** 2).sum())
            st.session_state["weights"] = weights
            st.session_state["cfg"] = dict(method=method, max_w=max_w, limits=limits)

            st.markdown("##### 3 · Result")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Historical return (ann.)", pct(ret_), help="Mean of past daily returns × 252. Not a forecast.")
            k2.metric("Volatility (ann.)", pct(vol_))
            k3.metric("Sharpe", f"{shp_:.2f}")
            k4.metric("Effective holdings", f"{eff_n:.1f}", help="1 ÷ Σw². Higher = more diversified.")

            g1, g2 = st.columns([1, 1.15])
            wdf = pd.DataFrame({"Ticker": weights.index, "Weight (%)": weights.values * 100})
            wdf["Bucket"] = wdf["Ticker"].map(nc.BUCKET_OF)
            wdf["Amount (₹)"] = weights.values * START
            wdf["Shares"] = [int(np.floor(START * w / price_map[t])) if t in price_map else 0
                             for t, w in weights.items()]
            wdf["Name"] = wdf["Ticker"].map(short)
            shown = wdf[wdf["Weight (%)"] > 0.05].sort_values("Weight (%)", ascending=False)
            with g1:
                fig = px.pie(shown, names="Name", values="Weight (%)", hole=.55, color="Bucket",
                             color_discrete_map=BUCKET_COLOR)
                fig.update_traces(textinfo="label+percent", sort=False)
                st.plotly_chart(style(fig, height=340, showlegend=False, title="Target weights"))
            with g2:
                rc = nc.risk_contributions(weights.values, cov)
                cmp_df = pd.DataFrame({"Name": [short(t) for t in weights.index],
                                       "Capital weight": weights.values * 100,
                                       "Risk share": rc.values * 100})
                cmp_df = cmp_df[(cmp_df["Capital weight"] > .05)].sort_values("Capital weight", ascending=False)
                fig = go.Figure()
                fig.add_bar(x=cmp_df["Name"], y=cmp_df["Capital weight"], name="Capital weight %", marker_color=TEAL)
                fig.add_bar(x=cmp_df["Name"], y=cmp_df["Risk share"], name="Share of portfolio risk %",
                            marker_color=AMBER)
                st.plotly_chart(style(fig, height=340, barmode="group", title="Money vs risk"))
            st.dataframe(shown[["Ticker", "Bucket", "Weight (%)", "Amount (₹)", "Shares"]]
                         .style.format({"Weight (%)": "{:.2f}", "Amount (₹)": "{:,.0f}"}), hide_index=True)

            b1, b2 = st.columns([1, 1])
            if b1.button("📤 Send to Trade desk as a rebalance plan"):
                st.session_state.plan = weights.copy()
                st.success("Plan saved. Open the Trade desk tab to review and apply the orders.")
            b2.download_button("⬇ Weights (CSV)", wdf[["Ticker", "Bucket", "Weight (%)", "Amount (₹)", "Shares"]]
                               .to_csv(index=False).encode(), "nova_weights.csv")

            st.markdown("##### Risk-return map")
            cloud = nc.random_portfolios(returns, n=3000, rf=rf, max_w=max(max_w, .35))
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=cloud["Volatility (%)"], y=cloud["Return (%)"], mode="markers",
                                       marker=dict(size=4, color=cloud["Sharpe"], colorscale="Tealgrn",
                                                   showscale=True, colorbar=dict(title="Sharpe")),
                                       name="Random portfolios"))
            ind_r, ind_v = mu * 100, np.sqrt(np.diag(cov.values)) * 100
            fig.add_trace(go.Scatter(x=ind_v, y=ind_r.values, mode="markers+text", text=[short(t) for t in mu.index],
                                     textposition="top center", marker=dict(size=8, color=MUTED), name="Single stocks"))
            fig.add_trace(go.Scatter(x=[vol_ * 100], y=[ret_ * 100], mode="markers",
                                     marker=dict(size=18, symbol="star", color=AMBER, line=dict(width=1, color="white")),
                                     name=f"Your portfolio ({method})"))
            st.plotly_chart(style(fig, height=470, xaxis_title="Volatility % (ann.)", yaxis_title="Return % (ann.)"))

            st.markdown("##### Diversification check")
            corr = returns.corr()
            lab = [short(t) for t in corr.columns]
            fig = go.Figure(go.Heatmap(z=corr.values, x=lab, y=lab, zmin=-0.2, zmax=1, colorscale="Tealgrn",
                                       text=np.round(corr.values, 2), texttemplate="%{text}"))
            st.plotly_chart(style(fig, height=520, yaxis_autorange="reversed"))
            pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().sort_values(ascending=False)
            if len(pairs):
                (ta, tb), cv = pairs.index[0], pairs.iloc[0]
                st.caption(f"Most similar pair: {short(ta)} & {short(tb)} (correlation {cv:.2f}). "
                           f"Holding both adds less diversification than the stock count suggests.")

# ================================================================== BACKTEST
with t_bt:
    st.subheader("Backtest & downside risk")
    weights = st.session_state.get("weights")
    cfg = st.session_state.get("cfg")
    if weights is None or cfg is None or returns.shape[0] < 120 or not set(weights.index) <= set(returns.columns):
        st.info("Build a portfolio first in the Portfolio builder tab.")
    else:
        b1, b2, b3 = st.columns(3)
        reb = b1.selectbox("Rebalancing", ["Monthly", "Quarterly", "Yearly", "Never"])
        oos = b2.checkbox("Out-of-sample test", value=True,
                          help="Fit weights on the first 70% of the window, then test only on the last 30%.")
        cap0 = b3.number_input("Starting capital (₹)", 100_000, 1_000_000_000, int(START), step=100_000)
        if oos:
            split = int(len(returns) * .7)
            train, test = returns.iloc[:split], returns.iloc[split:]
            w_bt = nc.optimise(cfg["method"], train, rf=rf, max_w=cfg["max_w"],
                               bucket_limits=cfg["limits"], scores=score_map)
            st.caption(f"Weights fitted on data up to {train.index[-1]:%d %b %Y}; results below cover "
                       f"{test.index[0]:%d %b %Y} → {test.index[-1]:%d %b %Y} (data the optimiser never saw).")
            rt = test
        else:
            w_bt, rt = weights, returns
            st.warning("In-sample: the weights were chosen using this same data, so results look better than "
                       "they could have been in real time.")
        eq = pd.Series(1 / rt.shape[1], index=rt.columns)
        paths = {"NOVA portfolio": nc.backtest(rt, w_bt, reb, cap0),
                 "Equal weight (same stocks)": nc.backtest(rt, eq, reb, cap0)}
        if not bench.empty:
            bb = bench[(bench.index >= rt.index[0]) & (bench.index <= rt.index[-1])]
            if len(bb) > 5:
                paths["Nifty 50"] = bb / bb.iloc[0] * cap0
        fig = go.Figure()
        colors = {"NOVA portfolio": TEAL, "Equal weight (same stocks)": AMBER, "Nifty 50": MUTED}
        for n_, s_ in paths.items():
            fig.add_trace(go.Scatter(x=s_.index, y=s_.values, name=n_, line=dict(color=colors[n_], width=2.4)))
        st.plotly_chart(style(fig, height=400, title="Growth of capital", yaxis_title="₹"))

        mt = pd.DataFrame({n_: nc.summary_metrics(s_, rf) for n_, s_ in paths.items()}).T
        fm = mt.copy()
        for c_ in ["CAGR", "Volatility", "Max drawdown", "Best day", "Worst day"]:
            fm[c_] = fm[c_].map(lambda v: pct(v, 1))
        for c_ in ["Sharpe", "Sortino"]:
            fm[c_] = fm[c_].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "—")
        st.dataframe(fm)

        d1, d2 = st.columns(2)
        with d1:
            dd = nc.drawdown(paths["NOVA portfolio"]) * 100
            fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color=CORAL)))
            st.plotly_chart(style(fig, height=300, title="Drawdown from peak (%)"))
        with d2:
            pv = paths["NOVA portfolio"]
            monthly = pv.groupby([pv.index.year, pv.index.month]).last().pct_change().dropna() * 100
            if len(monthly):
                hm = monthly.unstack(level=1)
                fig = go.Figure(go.Heatmap(z=hm.values, x=[f"{m:02d}" for m in hm.columns], y=[str(y) for y in hm.index],
                                           colorscale=[[0, CORAL], [.5, "#1b2a31"], [1, TEAL]], zmid=0,
                                           text=np.round(hm.values, 1), texttemplate="%{text}"))
                st.plotly_chart(style(fig, height=300, title="Monthly returns (%)"))

        pr = (rt * w_bt).sum(axis=1)
        var95, cvar95 = nc.var_cvar(pr, .95)
        var99, _ = nc.var_cvar(pr, .99)
        v1, v2, v3 = st.columns(3)
        v1.metric("1-day VaR (95%)", inr(var95 * cap0, False), help="On 95% of days the loss was smaller than this.")
        v2.metric("1-day CVaR (95%)", inr(cvar95 * cap0, False), help="Average loss on the worst 5% of days.")
        v3.metric("1-day VaR (99%)", inr(var99 * cap0, False))
        st.caption("Historical-simulation VaR from daily returns; ignores costs, tax and liquidity. "
                   "It describes the past window, not a ceiling on future losses.")

# ================================================================== GOAL SIMULATOR
with t_goal:
    st.subheader("Three-year goal simulator")
    weights = st.session_state.get("weights")
    if weights is None or returns.shape[0] < 120 or not set(weights.index) <= set(returns.columns):
        st.info("Build a portfolio first in the Portfolio builder tab.")
    else:
        g1, g2, g3 = st.columns(3)
        target = g1.selectbox("Target after 3 years", [20_000_000, 21_500_000, "Custom"],
                              format_func=lambda v: v if isinstance(v, str) else inr(v))
        if target == "Custom":
            target = g1.number_input("Custom target (₹)", 10_100_000, 500_000_000, 25_000_000, step=500_000)
        haircut = g2.slider("Return haircut (pp per year)", 0, 15, 0,
                            help="Subtracts this from the historical yearly return. Use it to stress-test a strong bull-market window.")
        horizon = g3.slider("Horizon (years)", 1, 5, 3)
        pr = (returns[weights.index] * weights).sum(axis=1)
        sim = nc.goal_simulation(pr, START, target, years=horizon, haircut_pp=haircut)
        need = nc.required_cagr(START, target, horizon)
        hist_cagr = (1 + pr).prod() ** (nc.TRADING_DAYS / len(pr)) - 1
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Required CAGR", pct(need))
        s2.metric("Past CAGR of this mix", pct(hist_cagr), help="In-sample, over the analysis window.")
        s3.metric("Chance of reaching target", pct(sim["p_target"], 0))
        s4.metric("Chance of ending below start", pct(sim["p_loss"], 0))

        x = sim["grid_years"]
        fig = go.Figure()
        P = sim["percentiles"]
        fig.add_trace(go.Scatter(x=x, y=P[95], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=P[5], fill="tonexty", fillcolor="rgba(45,212,191,.12)", line=dict(width=0),
                                 name="5–95% range"))
        fig.add_trace(go.Scatter(x=x, y=P[75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=P[25], fill="tonexty", fillcolor="rgba(45,212,191,.28)", line=dict(width=0),
                                 name="25–75% range"))
        fig.add_trace(go.Scatter(x=x, y=P[50], name="Median path", line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=x, y=START * (1 + need) ** x, name="Required path",
                                 line=dict(color=AMBER, dash="dash", width=2)))
        st.plotly_chart(style(fig, height=430, xaxis_title="Years", yaxis_title="₹",
                              title=f"{len(pr):,} days of history, resampled into 4,000 possible futures"))

        hist = px.histogram(x=sim["final"] / 1e7, nbins=60, color_discrete_sequence=[TEAL])
        hist.add_vline(x=target / 1e7, line_color=AMBER, line_dash="dash", annotation_text="Target")
        hist.add_vline(x=START / 1e7, line_color=CORAL, line_dash="dot", annotation_text="Start")
        st.plotly_chart(style(hist, height=300, xaxis_title="Final value (₹ crore)", yaxis_title="Paths",
                              title="Where the simulated portfolios end up"))

        rows = []
        for hc in (0, 5, 10):
            s_ = nc.goal_simulation(pr, START, target, years=horizon, haircut_pp=hc)
            rows.append({"Scenario": "Past returns repeat" if hc == 0 else f"Returns {hc} pp lower per year",
                         "Median outcome": inr(s_["median"]), "Chance of target": pct(s_["p_target"], 0),
                         "Chance of loss": pct(s_["p_loss"], 0)})
        st.markdown("##### Stress table")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Bootstrap simulation: daily returns from the analysis window are drawn at random with replacement. "
                   "It keeps the shape of past returns but breaks volatility clustering and assumes the future resembles "
                   "this window. Treat it as a stress-testing aid, not a prediction.")

# ================================================================== TRADE DESK
with t_trade:
    st.subheader("Trade desk")
    if st.session_state.get("flash"):
        kind, text = st.session_state.pop("flash")
        st.success("Done: " + text)
    cA, cB = st.columns([1, 1.3])
    with cA:
        sym_t = st.selectbox("Stock", avail, format_func=short, key="trade_sym")
        px_t = float(price_map[sym_t])
        qty = st.number_input("Shares", min_value=1, value=10, step=1)
        st.caption(f"Price ₹{px_t:,.2f} · value {inr(qty * px_t, False)} · fee {inr(qty * px_t * fee_bps / 1e4, False)}")
        bcol, scol = st.columns(2)
        for col, side in ((bcol, "BUY"), (scol, "SELL")):
            if col.button(side.title() + " virtually", key=f"btn_{side}", type="primary" if side == "BUY" else "secondary"):
                ok, msg = trade(side, sym_t, int(qty))
                if ok:
                    st.session_state.flash = ("ok", msg)
                    st.rerun()
                st.error(msg)
    with cB:
        st.markdown("##### Holdings")
        if holdings:
            hrows = []
            for s_, q_ in holdings.items():
                avg = st.session_state.cost.get(s_, 0) / q_
                hrows.append({"Ticker": s_, "Bucket": nc.BUCKET_OF.get(s_), "Shares": q_, "Avg cost (₹)": avg,
                              "Price (₹)": price_map.get(s_), "Value (₹)": q_ * price_map.get(s_, 0),
                              "P/L (₹)": q_ * (price_map.get(s_, 0) - avg),
                              "P/L (%)": (price_map.get(s_, 0) / avg - 1) * 100})
            st.dataframe(pd.DataFrame(hrows).style.format(precision=2, na_rep="—"), hide_index=True)
        else:
            st.info("No holdings yet — all ₹1 crore is cash.")
    st.caption(f"Fees paid so far: {inr(st.session_state.fees, False)}")

    st.markdown("##### Rebalance plan from the Portfolio builder")
    plan = st.session_state.plan
    if plan is None:
        st.caption("No plan yet. Build a portfolio and press “Send to Trade desk”.")
    else:
        orders = nc.rebalance_orders(plan, holdings, price_map, st.session_state.cash, fee_bps)
        orders = orders[orders["Order (shares)"] != 0] if not orders.empty else orders
        if orders.empty:
            st.success("Your book already matches the plan.")
        else:
            st.dataframe(orders.style.format({"Price (₹)": "{:,.2f}", "Order value (₹)": "{:,.0f}",
                                              "Target weight (%)": "{:.2f}"}), hide_index=True)
            st.caption("Sells run first, then buys. Whole shares only; leftover cash stays in the account.")
            if st.button("✅ Apply these orders"):
                msgs, bad = [], []
                for _, r in orders.iterrows():
                    if r["Order (shares)"] < 0:
                        ok, m = trade("SELL", r["Ticker"], int(-r["Order (shares)"]))
                        (msgs if ok else bad).append(m)
                for _, r in orders.iterrows():
                    if r["Order (shares)"] > 0:
                        ok, m = trade("BUY", r["Ticker"], int(r["Order (shares)"]))
                        (msgs if ok else bad).append(m)
                st.session_state.plan = None
                if bad:
                    st.warning("Some orders were skipped: " + " | ".join(bad))
                st.rerun()

    st.markdown("##### Ledger")
    if st.session_state.ledger:
        led = pd.DataFrame(st.session_state.ledger[::-1])
        st.dataframe(led, hide_index=True)
        st.download_button("⬇ Ledger (CSV)", led.to_csv(index=False).encode(), "nova_ledger.csv")
    else:
        st.caption("No trades yet.")
    st.warning("Trades live only in this browser session. They reset if the app restarts or you open a new session.")

# ================================================================== METHOD
with t_method:
    st.subheader("Method, assumptions and limits")
    st.markdown(f"""
**Universe.** 15 NSE stocks in three size buckets chosen by the project: Large (HDFCBANK, RELIANCE, TCS, LT, BHARTIARTL),
Mid (POLYCAB, PERSISTENT, COFORGE, SUNDARMFIN, COROMANDEL), Small (SONACOMS, SYRMA, AEGISLOG, NAVINFLUOR, KARURVYSYA).
Bucket labels are your own classification; actual market caps shown in the screener may not match them.

**Ranking.** Each factor is converted to a percentile among the selected stocks (lower is better for P/E, PEG, P/BV, beta
and volatility), grouped into six categories and combined with the sidebar weights, normalised to 100%.
Missing data gets a neutral 50. ROCE is approximated from annual statements and left blank for banks and NBFCs.

**Portfolio builder.** Long-only, fully invested, per-stock cap, and size-bucket limits that start from the risk-profile quiz.
Methods: equal weight; minimum variance; maximum Sharpe; risk parity (equal risk contributions); score-tilted
(weights pulled toward the composite score). Inputs are annualised mean returns and the covariance matrix of daily returns
over the analysis window. Maximum Sharpe is very sensitive to the historical mean, so expect concentrated answers; the
per-stock cap is there to limit that.

**Backtest.** Daily compounding with optional periodic rebalancing to target weights. No tax, slippage or brokerage.
The out-of-sample option fits on the first 70% of the window and tests on the last 30%.

**Risk.** Historical-simulation VaR and CVaR on daily portfolio returns, drawdown, Sharpe and Sortino using your risk-free rate.

**Goal simulator.** Bootstrap of past daily portfolio returns over the horizon; outputs the chance of reaching the target.
Required CAGR = (Target ÷ Start)^(1/years) − 1, which is {pct(nc.required_cagr(START, 20_000_000))} for ₹2.00 Cr and
{pct(nc.required_cagr(START, 21_500_000))} for ₹2.15 Cr over three years.

**Limits to state in your report.** Yahoo Finance data via an unofficial package can be delayed or incomplete; the optimiser
overfits history; a three-year window of Indian mid/small caps may include an unusually strong period; correlations rise in
sell-offs; the simulator ignores regime changes. This is an educational tool and not investment advice.
""")
    st.latex(r"\text{Sharpe}=\frac{R_p-R_f}{\sigma_p}\qquad \text{VaR}_{95}=-\,Q_{5\%}(r_p)")

st.divider()
st.caption("NOVA Investment Management · academic prototype · no real trades are placed")
