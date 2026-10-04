"""
Trading research agent: Indian stocks (NSE) + crypto.
Features: data fetch, strategy library, walk-forward backtest, fundamental screener,
paper trading with risk controls. NO live orders by default.

Install:  pip install yfinance ccxt pandas numpy scikit-learn
Run:      python trading_agent.py backtest RELIANCE.NS
          python trading_agent.py backtest BTC-USD
          python trading_agent.py screen
          python trading_agent.py paper BTC-USD

NOT financial advice. Past performance does not predict future returns.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# ---------------- Data ----------------
def get_data(symbol, period="5y"):
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ---------------- Indicators ----------------
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)

def features(df):
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change()
    f["ret5"] = c.pct_change(5)
    f["ret20"] = c.pct_change(20)
    f["rsi"] = rsi(c)
    f["vol20"] = f["ret1"].rolling(20).std()
    f["ma_gap"] = c / c.rolling(50).mean() - 1
    return f

# ---------------- Strategies (return position series: 1 long, 0 flat) ----------------
def strat_trend(df):
    c = df["Close"]
    return (c.rolling(20).mean() > c.rolling(100).mean()).astype(int)

def strat_meanrev(df):
    r = rsi(df["Close"])
    pos = pd.Series(np.nan, index=df.index)
    pos[r < 30] = 1
    pos[r > 55] = 0
    return pos.ffill().fillna(0)

def strat_ml(df, train_years=2, step=63):
    """Walk-forward Random Forest: retrain every `step` days, predict next-day direction."""
    f = features(df)
    y = (df["Close"].pct_change().shift(-1) > 0).astype(int)
    data = f.join(y.rename("y")).dropna()
    pos = pd.Series(0, index=df.index, dtype=float)
    win = 252 * train_years
    for i in range(win, len(data), step):
        train = data.iloc[i - win:i]
        test = data.iloc[i:i + step]
        m = RandomForestClassifier(n_estimators=200, max_depth=4,
                                   min_samples_leaf=20, random_state=0)
        m.fit(train.drop(columns="y"), train["y"])
        p = m.predict_proba(test.drop(columns="y"))[:, 1]
        pos.loc[test.index] = (p > 0.55).astype(int)
    return pos

STRATEGIES = {"trend": strat_trend, "meanrev": strat_meanrev, "ml": strat_ml}

# ---------------- Backtest ----------------
def backtest(df, pos, fee=0.001):
    """Position decided at close t is applied to return of t+1 (no lookahead). fee = per-trade cost."""
    ret = df["Close"].pct_change()
    p = pos.shift(1).fillna(0)
    cost = p.diff().abs().fillna(0) * fee
    strat = p * ret - cost
    eq = (1 + strat.fillna(0)).cumprod()
    bh = (1 + ret.fillna(0)).cumprod()
    years = len(df) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd,
            "BuyHoldCAGR": bh.iloc[-1] ** (1 / years) - 1,
            "Trades": int(p.diff().abs().sum())}

def run_backtests(symbol):
    df = get_data(symbol)
    # hold out the last 20% untouched to check for overfitting
    split = int(len(df) * 0.8)
    for name, fn in STRATEGIES.items():
        pos = fn(df)
        ins = backtest(df.iloc[:split], pos.iloc[:split])
        oos = backtest(df.iloc[split:], pos.iloc[split:])
        print(f"\n[{name}] in-sample : {fmt(ins)}")
        print(f"[{name}] OUT-sample: {fmt(oos)}")

def fmt(d):
    return (f"CAGR {d['CAGR']:.1%} | B&H {d['BuyHoldCAGR']:.1%} | "
            f"Sharpe {d['Sharpe']:.2f} | MaxDD {d['MaxDD']:.1%} | trades {d['Trades']}")

# ---------------- Fundamental screener (Indian stocks) ----------------
NSE_UNIVERSE = ["TATAMOTORS.NS", "BHARTIARTL.NS", "ITC.NS", "BAJFINANCE.NS",
                "BALKRISIND.NS", "SYNGENE.NS", "HAL.NS", "BEL.NS", "IRFC.NS",
                "POLYCAB.NS", "DIXON.NS", "TRENT.NS", "PERSISTENT.NS", "KPITTECH.NS"]

def screen(universe=NSE_UNIVERSE):
    print(screen_df(universe).round(3))
    print("\nScreen output = shortlist for research, not buy signals.")

def screen_df(universe=NSE_UNIVERSE):
    rows = []
    for s in universe:
        try:
            i = yf.Ticker(s).info
            rows.append({
                "symbol": s,
                "pe": i.get("trailingPE"),
                "roe": i.get("returnOnEquity"),
                "rev_growth": i.get("revenueGrowth"),
                "earn_growth": i.get("earningsGrowth"),
                "debt_to_equity": i.get("debtToEquity"),
            })
        except Exception as e:
            print("skip", s, e)
    df = pd.DataFrame(rows).set_index("symbol")
    # simple rank score: high ROE & growth, low PE & debt
    score = (df["roe"].rank() + df["rev_growth"].rank() + df["earn_growth"].rank()
             - df["pe"].rank() - df["debt_to_equity"].rank())
    df["score"] = score
    return df.sort_values("score", ascending=False)

# ---------------- Paper trading with risk controls ----------------
class PaperBroker:
    def __init__(self, cash=100000, risk_per_trade=0.01, stop=0.05):
        self.cash, self.qty, self.entry = cash, 0.0, 0.0
        self.risk, self.stop = risk_per_trade, stop

    def step(self, price, signal):
        if signal == 1 and self.qty == 0:
            size = (self.cash * self.risk) / self.stop  # risk 1% of capital per trade
            self.qty = min(size, self.cash) / price
            self.cash -= self.qty * price
            self.entry = price
            print(f"BUY  {self.qty:.4f} @ {price:.2f}")
        elif self.qty > 0 and (signal == 0 or price < self.entry * (1 - self.stop)):
            self.cash += self.qty * price
            print(f"SELL {self.qty:.4f} @ {price:.2f}  cash={self.cash:.2f}")
            self.qty = 0

def paper(symbol):
    df = get_data(symbol, "1y")
    sig = strat_trend(df)
    b = PaperBroker()
    for price, s in zip(df["Close"], sig):
        b.step(price, s)
    print("Final equity:", round(b.cash + b.qty * df["Close"].iloc[-1], 2))

# ---------------- Live execution (intentionally disabled) ----------------
# Crypto: ccxt (e.g. ccxt.binance / ccxt.coindcx / ccxt.wazirx) -> exchange.create_order(...)
# Indian stocks: broker APIs (Zerodha Kite Connect, Angel One SmartAPI, Upstox)
# Only wire these after months of paper trading, with hard daily loss limits.

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == "backtest":
        run_backtests(sys.argv[2])
    elif sys.argv[1] == "screen":
        screen()
    elif sys.argv[1] == "paper":
        paper(sys.argv[2])
