"""NOVA analytics core: portfolio maths with no Streamlit / network dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

# --------------------------------------------------------------------------
# Stock universe (cap bucket is the project's own classification)
# --------------------------------------------------------------------------
UNIVERSE = {
    "Large": ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "LT.NS", "BHARTIARTL.NS"],
    "Mid": ["POLYCAB.NS", "PERSISTENT.NS", "COFORGE.NS", "SUNDARMFIN.NS", "COROMANDEL.NS"],
    "Small": ["SONACOMS.NS", "SYRMA.NS", "AEGISLOG.NS", "NAVINFLUOR.NS", "KARURVYSYA.NS"],
}
BUCKET_OF = {t: b for b, ts in UNIVERSE.items() for t in ts}
ALL_TICKERS = [t for ts in UNIVERSE.values() for t in ts]
BENCHMARK = "^NSEI"  # Nifty 50

# Banks / NBFCs: ROCE from EBIT / capital employed is not meaningful for them.
FINANCIALS = {"HDFCBANK.NS", "SUNDARMFIN.NS", "KARURVYSYA.NS"}


# --------------------------------------------------------------------------
# Return / risk statistics
# --------------------------------------------------------------------------
def annual_stats(returns: pd.DataFrame):
    """Annualised mean vector and covariance matrix from daily returns."""
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    return mu, cov


def port_perf(w, mu, cov, rf):
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _bounds(tickers, max_w, bucket_min=None):
    return [(0.0, max_w) for _ in tickers]


def _bucket_constraints(tickers, bucket_limits):
    """bucket_limits: {bucket: (min, max)} as fractions of the portfolio."""
    cons = []
    for bucket, (lo, hi) in (bucket_limits or {}).items():
        idx = np.array([1.0 if BUCKET_OF.get(t) == bucket else 0.0 for t in tickers])
        if idx.sum() == 0:
            continue
        cons.append({"type": "ineq", "fun": lambda w, i=idx, lo=lo: float(i @ w) - lo})
        cons.append({"type": "ineq", "fun": lambda w, i=idx, hi=hi: hi - float(i @ w)})
    return cons


def optimise(method, returns, rf=0.07, max_w=0.25, bucket_limits=None, scores=None):
    """Return weights (pd.Series) for one of the supported methods.

    methods: Equal weight, Minimum variance, Maximum Sharpe, Risk parity, Score-tilted
    """
    tickers = list(returns.columns)
    n = len(tickers)
    mu, cov = annual_stats(returns)
    max_w = max(max_w, 1.0 / n + 1e-9)  # keep the problem feasible
    x0 = np.full(n, 1.0 / n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    cons += _bucket_constraints(tickers, bucket_limits)
    bnds = _bounds(tickers, max_w)

    if method == "Equal weight":
        return pd.Series(x0, index=tickers)

    if method == "Score-tilted":
        s = pd.Series(scores, dtype=float).reindex(tickers).fillna(50.0).clip(lower=1.0)
        w0 = (s / s.sum()).values
        # blend toward equal weight and enforce the cap by projecting with the optimiser
        res = minimize(lambda w: np.sum((w - w0) ** 2), x0, bounds=bnds,
                       constraints=cons, method="SLSQP")
        w = res.x if res.success else x0
    elif method == "Minimum variance":
        res = minimize(lambda w: w @ cov.values @ w, x0, bounds=bnds,
                       constraints=cons, method="SLSQP")
        w = res.x if res.success else x0
    elif method == "Maximum Sharpe":
        def neg_sharpe(w):
            r, v, s = port_perf(w, mu, cov, rf)
            return -s if np.isfinite(s) else 1e3
        res = minimize(neg_sharpe, x0, bounds=bnds, constraints=cons, method="SLSQP")
        w = res.x if res.success else x0
    elif method == "Risk parity":
        def rp_obj(w):
            vol = np.sqrt(w @ cov.values @ w)
            if vol == 0:
                return 1e3
            rc = w * (cov.values @ w) / vol
            return float(np.sum((rc - rc.mean()) ** 2))
        res = minimize(rp_obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
        w = res.x if res.success else x0
    else:
        raise ValueError(f"Unknown method: {method}")

    w = np.clip(w, 0, None)
    w[w < 1e-4] = 0.0  # drop numerical dust
    w = w / w.sum() if w.sum() > 0 else x0
    return pd.Series(w, index=tickers)


def random_portfolios(returns, n=3000, rf=0.07, max_w=0.35, seed=7):
    """Random long-only portfolios for an efficient-frontier cloud."""
    rng = np.random.default_rng(seed)
    mu, cov = annual_stats(returns)
    k = returns.shape[1]
    out = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(k) * 0.8)
        if w.max() > max_w:
            continue
        r, v, s = port_perf(w, mu, cov, rf)
        out.append((r * 100, v * 100, s))
    return pd.DataFrame(out, columns=["Return (%)", "Volatility (%)", "Sharpe"])


def risk_contributions(w, cov):
    w = np.asarray(w, dtype=float)
    vol = np.sqrt(w @ cov.values @ w)
    if vol == 0:
        return pd.Series(np.zeros(len(w)), index=cov.index)
    rc = w * (cov.values @ w) / vol
    return pd.Series(rc / rc.sum(), index=cov.index)


# --------------------------------------------------------------------------
# Backtest and drawdown
# --------------------------------------------------------------------------
def backtest(returns, weights, rebalance="Monthly", capital=10_000_000.0):
    """Portfolio value path with periodic rebalancing back to target weights.

    rebalance: 'Never' (buy & hold), 'Monthly', 'Quarterly', 'Yearly'.
    Transaction costs, tax and slippage are ignored.
    """
    r = returns.fillna(0.0)
    w_t = pd.Series(weights, index=r.columns).values.astype(float)
    holdings_w = w_t.copy()
    value = capital
    values = []
    last_key = None
    for dt, row in r.iterrows():
        key = {"Monthly": (dt.year, dt.month),
               "Quarterly": (dt.year, (dt.month - 1) // 3),
               "Yearly": (dt.year,)}.get(rebalance)
        if rebalance != "Never" and last_key is not None and key != last_key:
            holdings_w = w_t.copy()
        last_key = key
        growth = float(holdings_w @ (1.0 + row.values))
        value *= growth
        holdings_w = holdings_w * (1.0 + row.values) / growth
        values.append(value)
    return pd.Series(values, index=r.index, name="Portfolio")


def drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return series / peak - 1.0


def summary_metrics(value: pd.Series, rf=0.07):
    """CAGR, volatility, Sharpe, Sortino, max drawdown from a value path."""
    rets = value.pct_change().dropna()
    if len(rets) < 3:
        return {}
    years = (value.index[-1] - value.index[0]).days / 365.25
    cagr = (value.iloc[-1] / value.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sharpe = (cagr - rf) / vol if vol > 0 else np.nan
    sortino = (cagr - rf) / downside if downside and downside > 0 else np.nan
    mdd = drawdown(value).min()
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe,
            "Sortino": sortino, "Max drawdown": mdd,
            "Best day": rets.max(), "Worst day": rets.min()}


def var_cvar(portfolio_returns: pd.Series, level=0.95):
    """Historical one-day VaR / CVaR as positive loss fractions."""
    x = portfolio_returns.dropna()
    if len(x) < 30:
        return np.nan, np.nan
    q = np.quantile(x, 1 - level)
    tail = x[x <= q]
    return float(-q), float(-tail.mean()) if len(tail) else np.nan


# --------------------------------------------------------------------------
# Goal simulation (Monte Carlo, bootstrapped from history)
# --------------------------------------------------------------------------
def goal_simulation(portfolio_returns: pd.Series, start, target, years=3,
                    n_paths=4000, seed=11, haircut_pp=0.0):
    """Bootstrap daily returns; return the paths summary and P(reach target)."""
    x = portfolio_returns.dropna().values
    if len(x) < 60:
        return None
    rng = np.random.default_rng(seed)
    steps = int(years * TRADING_DAYS)
    draws = rng.choice(x, size=(n_paths, steps), replace=True)
    draws = draws - (haircut_pp / 100.0) / TRADING_DAYS  # stress: lower yearly return
    paths = start * np.cumprod(1.0 + draws, axis=1)
    paths = np.concatenate([np.full((n_paths, 1), start), paths], axis=1)
    final = paths[:, -1]
    grid = np.linspace(0, steps, years * 12 + 1).astype(int)
    pct = {q: np.percentile(paths[:, grid], q, axis=0) for q in (5, 25, 50, 75, 95)}
    return {
        "grid_years": grid / TRADING_DAYS,
        "percentiles": pct,
        "final": final,
        "p_target": float((final >= target).mean()),
        "p_loss": float((final < start).mean()),
        "median": float(np.median(final)),
    }


def required_cagr(start, target, years=3):
    return (target / start) ** (1 / years) - 1


# --------------------------------------------------------------------------
# Risk-profile questionnaire -> suggested cap mix
# --------------------------------------------------------------------------
QUIZ = [
    ("If your portfolio fell 20% in a quarter, you would…",
     ["Sell to stop the pain", "Hold and wait", "Hold and add a little", "Buy aggressively"]),
    ("Your main goal for this simulation is…",
     ["Protect capital", "Steady growth", "Beat the market", "Maximise growth"]),
    ("How long could you leave this money untouched?",
     ["Under 1 year", "1-3 years", "3-5 years", "5+ years"]),
    ("Which experience feels closest to your comfort zone?",
     ["Fixed-deposit style stability", "Occasional dips are fine",
      "Big swings are acceptable", "I like high-conviction bets"]),
]

PROFILES = [
    ("Defensive", {"Large": (0.60, 0.80), "Mid": (0.10, 0.30), "Small": (0.00, 0.10)}),
    ("Balanced", {"Large": (0.45, 0.65), "Mid": (0.20, 0.35), "Small": (0.05, 0.20)}),
    ("Growth", {"Large": (0.30, 0.50), "Mid": (0.25, 0.40), "Small": (0.10, 0.30)}),
    ("Aggressive", {"Large": (0.15, 0.35), "Mid": (0.30, 0.45), "Small": (0.20, 0.40)}),
]


def profile_from_answers(answers):
    """answers: list of ints 0..3. Returns (name, bucket limits, score)."""
    score = int(sum(answers))
    max_score = 3 * len(QUIZ)
    frac = score / max_score if max_score else 0
    idx = min(int(frac * len(PROFILES)), len(PROFILES) - 1)
    name, limits = PROFILES[idx]
    return name, limits, score


# --------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------
def rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# --------------------------------------------------------------------------
# Rebalance helper
# --------------------------------------------------------------------------
def rebalance_orders(target_w: pd.Series, holdings: dict, prices: dict, cash: float, fee_bps: float = 0.0):
    """Share-level orders to move current holdings to target weights.

    Returns DataFrame with current/target value, order shares (+buy/-sell).
    Whole shares only; sells are listed first so cash is available for buys.
    """
    total = cash + sum(holdings.get(t, 0) * prices.get(t, 0) for t in prices)
    total *= 1.0 - 2.0 * fee_bps / 10_000  # keep a cash cushion for trading costs
    rows = []
    for t, w in target_w.items():
        px = prices.get(t, np.nan)
        if not np.isfinite(px) or px <= 0:
            continue
        cur_sh = int(holdings.get(t, 0))
        tgt_sh = int(np.floor(total * w / px))
        rows.append({"Ticker": t, "Price (₹)": px, "Current shares": cur_sh,
                     "Target shares": tgt_sh, "Order (shares)": tgt_sh - cur_sh,
                     "Order value (₹)": (tgt_sh - cur_sh) * px,
                     "Target weight (%)": w * 100})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Order (shares)").reset_index(drop=True)
    return df
