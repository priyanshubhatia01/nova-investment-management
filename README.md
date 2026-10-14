# NOVA Investment Management — Setup Guide (v2)

## What is new in this version
- **New 15-stock universe** in three size buckets
  - Large: HDFCBANK, RELIANCE, TCS, LT, BHARTIARTL
  - Mid: POLYCAB, PERSISTENT, COFORGE, SUNDARMFIN, COROMANDEL
  - Small: SONACOMS, SYRMA, AEGISLOG, NAVINFLUOR, KARURVYSYA
  - Note: the Navin Fluorine ticker on Yahoo Finance is spelled **NAVINFLUOR.NS**.
- **Command centre:** universe heat-map, ticker strip, book allocation.
- **Screener:** filters, factor heat-map, approximate ROCE (blank for banks/NBFCs).
- **Stock lab:** candlesticks, moving averages, RSI, 52-week position, peer and Nifty 50 comparison.
- **Portfolio builder:** risk-profile quiz, five construction methods, size-bucket limits, per-stock cap,
  money-vs-risk chart, risk-return map, correlation heat-map.
- **Backtest & risk:** rebalancing options, out-of-sample test, drawdown, monthly return grid, VaR/CVaR.
- **Goal simulator:** thousands of simulated futures, chance of reaching ₹2.00 Cr / ₹2.15 Cr, stress table.
- **Trade desk:** trading costs, cost-basis P/L, one-click rebalance from the builder, ledger download.

## Part A — Install Python (one-time)
1. Open https://www.python.org/downloads/ and download Python 3.11 or 3.12.
2. Run the installer and tick **Add Python to PATH**, then **Install Now**.

## Part B — Extract and open a command window
1. Right-click the ZIP and choose **Extract All**.
2. Open the `NOVA_Investment_Management` folder. You should see `app.py`, `nova_core.py`, `requirements.txt`.
3. Click the address bar, type `cmd`, press Enter.

## Part C — Run (one command at a time)
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
The site opens at http://localhost:8501.

## Part D — Put it online so anyone can open it
1. Sign in to https://github.com/ and create a **public** repository named `nova-investment-management`.
2. Upload these files to the top level of the repository: `app.py`, `nova_core.py`, `requirements.txt`.
   (`app.py` will not start without `nova_core.py`.)
3. Add the theme file: in the repository click **Add file → Create new file**, type the name
   `.streamlit/config.toml` (typing the slash creates the folder), paste the contents of the `config.toml` file from this ZIP, and click **Commit**.
   Without it, visitors using a light browser theme may see dark text on a dark background.
4. Open https://share.streamlit.io/, sign in with GitHub and click **Create app**.
5. Choose your repository, branch `main`, main file `app.py`.
6. Open **Advanced settings** and pick Python 3.12, then click **Deploy**.
7. When it finishes, copy the `https://….streamlit.app` link. Open it in a private/incognito window to confirm anyone can see it.

Free apps go to sleep after a period of no visits. If the page says it is asleep, click **Yes, get this app back up**.
Open the link yourself shortly before the professor does. If Yahoo Finance rejects requests from the shared server, press **Refresh market data** or reboot the app from the Streamlit menu.

## Important limitations
- Data comes from Yahoo Finance through the unofficial `yfinance` package; it can be delayed, incomplete or occasionally blocked.
- ROCE is approximated as EBIT ÷ (total assets − current liabilities). Check it against company filings before quoting it.
- The size buckets are the project's own classification, not an official SEBI/AMFI list.
- Optimised weights are fitted to history and can overfit. Use the out-of-sample option and say so in the report.
- The goal simulator resamples past returns; it is a stress-testing aid, not a forecast.
- Virtual trades live in the browser session and reset if the app restarts.
- Educational project, not investment advice.
