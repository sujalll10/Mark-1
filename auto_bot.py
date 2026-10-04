"""
auto_bot.py - runs once a day (via GitHub Actions), fully automatic.

What it does each run:
 1. Downloads data for the WATCH list (Indian stocks + crypto).
 2. Weekly (or if nothing is active): searches many strategy variants, keeps only
    those that beat buy-and-hold OUT-OF-SAMPLE on most symbols. If none qualify,
    the bot stays in cash (that is a valid, honest result).
 3. Generates signals, applies risk rules, and trades.
 4. Logs every trade in a journal, then "learns": pauses strategies that keep losing.
 5. Writes a daily report to reports/YYYY-MM-DD.md and reports/latest.md.

SAFETY: LIVE = False means PAPER trading only. No real money moves.
NOT financial advice.
"""
import json
import os
import datetime as dt
import numpy as np
import pandas as pd
from trading_agent import get_data, backtest, screen_df, rsi

# ---------------- Settings ----------------
LIVE = False                 # keep False until you have 2-3 months of good paper results
WATCH = ["ITC.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "HAL.NS", "BEL.NS",
         "BTC-USD", "ETH-USD"]
CAPITAL = 100000             # paper money (notional units)
MAX_POS_PCT = 0.10           # max 10% of equity per position
STOP = 0.05                  # 5% stop loss
DAILY_LOSS_LIMIT = 0.02      # no new buys if equity fell 2% since last run
FEE = 0.001                  # 0.1% per side
STATE_FILE = "state.json"

# ---------------- Strategy families ----------------
def make_trend(fast, slow):
    return lambda df: (df["Close"].rolling(fast).mean()
                       > df["Close"].rolling(slow).mean()).astype(int)

def make_breakout(n):
    def f(df):
        c = df["Close"]
        hi = c.rolling(n).max().shift(1)
        lo = c.rolling(max(n // 2, 5)).min().shift(1)
        pos = pd.Series(np.nan, index=df.index)
        pos[c > hi] = 1
        pos[c < lo] = 0
        return pos.ffill().fillna(0)
    return f

def make_meanrev(lo, hi):
    def f(df):
        r = rsi(df["Close"])
        pos = pd.Series(np.nan, index=df.index)
        pos[r < lo] = 1
        pos[r > hi] = 0
        return pos.ffill().fillna(0)
    return f

FAMS = {"trend": make_trend, "breakout": make_breakout, "meanrev": make_meanrev}
GRID = ([{"fam": "trend", "params": p} for p in [(10, 50), (20, 100), (50, 200)]]
        + [{"fam": "breakout", "params": (n,)} for n in (20, 55, 100)]
        + [{"fam": "meanrev", "params": p} for p in [(25, 55), (30, 55), (30, 60)]])

def label(spec):
    return f"{spec['fam']}{tuple(spec['params'])}"

def build(spec):
    return FAMS[spec["fam"]](*spec["params"])

# ---------------- Strategy discovery ----------------
def discover(data):
    """Keep variants that work out-of-sample on most symbols (last 30% of data)."""
    results = []
    for spec in GRID:
        sharpes, wins = [], 0
        for sym, df in data.items():
            pos = build(spec)(df)
            split = int(len(df) * 0.7)
            o = backtest(df.iloc[split:], pos.iloc[split:], FEE)
            sharpes.append(o["Sharpe"])
            wins += o["CAGR"] > o["BuyHoldCAGR"]
        results.append((spec, float(np.mean(sharpes)), wins / max(len(data), 1)))
    results.sort(key=lambda x: x[1], reverse=True)
    keep = [r for r in results if r[1] > 0.5 and r[2] >= 0.6][:3]
    return [r[0] for r in keep], results

# ---------------- State ----------------
def load():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": CAPITAL, "positions": {}, "journal": [], "active": [],
            "paused": [], "equity": CAPITAL, "last_equity": CAPITAL}

def save(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=1)

# ---------------- Execution ----------------
def place_order(sym, side, qty, price):
    if LIVE:
        # Connect your broker here (Zerodha Kite Connect, Angel One SmartAPI, ccxt for crypto).
        raise NotImplementedError("Broker not connected. Paper mode only.")
    return f"[PAPER] {side} {qty:.4f} {sym} @ {price:.2f}"

def equity_of(st, prices):
    return st["cash"] + sum(p["qty"] * prices.get(s, p["entry"])
                            for s, p in st["positions"].items())

# ---------------- Learning from the trade journal ----------------
def learn(st):
    stats = {}
    for t in st["journal"]:
        for s in t["strategies"]:
            d = stats.setdefault(s, {"n": 0, "wins": 0, "pnl": 0.0})
            d["n"] += 1
            d["wins"] += t["pnl"] > 0
            d["pnl"] += t["pnl"]
    st["paused"] = [s for s, d in stats.items()
                    if d["n"] >= 10 and d["pnl"] < 0 and d["wins"] / d["n"] < 0.4]
    return stats

# ---------------- Main run ----------------
def run():
    today = dt.date.today()
    st = load()
    data = {}
    for s in WATCH:
        try:
            d = get_data(s, "3y")
            if not d.empty:
                data[s] = d
        except Exception as e:
            print("skip", s, e)

    prices = {s: float(d["Close"].iloc[-1]) for s, d in data.items()}
    actions, notes = [], []

    # weekly strategy search (Sundays) or if nothing active
    ranking = []
    if not st["active"] or today.weekday() == 6:
        st["active"], ranking = discover(data)
        notes.append("Strategy search ran. " + (
            "Qualified: " + ", ".join(label(s) for s in st["active"])
            if st["active"] else "No strategy beat buy-and-hold out-of-sample, so the bot stays in cash."))

    live_specs = [s for s in st["active"] if label(s) not in st["paused"]]
    eq_now = equity_of(st, prices)
    halted = eq_now < st["last_equity"] * (1 - DAILY_LOSS_LIMIT)
    if halted:
        notes.append("Daily loss limit hit: no new buys today.")

    for sym, df in data.items():
        price = prices[sym]
        votes = [label(sp) for sp in live_specs if int(build(sp)(df).iloc[-1]) == 1]
        want_long = bool(live_specs) and len(votes) / len(live_specs) >= 0.5
        pos = st["positions"].get(sym)

        if pos:
            reason = None
            if price <= pos["entry"] * (1 - STOP):
                reason = "stop-loss"
            elif not want_long:
                reason = "signal-exit"
            if reason:
                actions.append(place_order(sym, "SELL", pos["qty"], price))
                pnl = pos["qty"] * (price * (1 - FEE) - pos["entry"] * (1 + FEE))
                st["cash"] += pos["qty"] * price * (1 - FEE)
                st["journal"].append({"date": str(today), "sym": sym, "entry": pos["entry"],
                                      "exit": price, "qty": pos["qty"], "pnl": pnl,
                                      "reason": reason, "strategies": pos["strategies"]})
                del st["positions"][sym]
        elif want_long and not halted:
            size = min(eq_now * MAX_POS_PCT, st["cash"])
            if size > 100:
                qty = size / price / (1 + FEE)
                actions.append(place_order(sym, "BUY", qty, price))
                st["cash"] -= qty * price * (1 + FEE)
                st["positions"][sym] = {"qty": qty, "entry": price,
                                        "date": str(today), "strategies": votes}

    stats = learn(st)
    eq_end = equity_of(st, prices)
    day_change = eq_end / st["last_equity"] - 1
    st["equity"], st["last_equity"] = eq_end, eq_end

    # shortlist for research (not buy signals)
    try:
        shortlist = screen_df().head(5).round(3).to_string()
    except Exception as e:
        shortlist = f"Screener unavailable today ({e})"

    report(today, st, prices, actions, notes, stats, ranking, shortlist, eq_end, day_change)
    save(st)

# ---------------- Daily report ----------------
def report(today, st, prices, actions, notes, stats, ranking, shortlist, eq, chg):
    L = [f"# Daily report - {today}", "",
         f"**Mode:** {'LIVE' if LIVE else 'PAPER (no real money)'}",
         f"**Equity:** {eq:,.0f} ({chg:+.2%} vs last run) | **Cash:** {st['cash']:,.0f}", ""]
    L.append("## Actions today")
    L += [f"- {a}" for a in actions] or ["- None"]
    L += ["", "## Notes"] + ([f"- {n}" for n in notes] or ["- None"])

    L += ["", "## Open positions"]
    for s, p in st["positions"].items():
        gain = prices.get(s, p["entry"]) / p["entry"] - 1
        L.append(f"- {s}: entry {p['entry']:.2f}, now {prices.get(s, 0):.2f} ({gain:+.1%}), via {', '.join(p['strategies'])}")
    if not st["positions"]:
        L.append("- None")

    L += ["", "## Mistakes review (last 7 closed trades)"]
    recent = st["journal"][-7:]
    for t in recent:
        tag = "WIN" if t["pnl"] > 0 else "LOSS"
        lesson = ""
        if t["pnl"] <= 0:
            lesson = (" - stopped out: entry timing/volatility; check if market was choppy"
                      if t["reason"] == "stop-loss" else " - signal flipped before profit; strategy may be whipsawing")
        L.append(f"- {tag} {t['sym']} {t['pnl']:+.0f} ({t['reason']}){lesson}")
    if not recent:
        L.append("- No closed trades yet")

    L += ["", "## Strategy scoreboard (what the bot learns from)"]
    for s, d in stats.items():
        L.append(f"- {s}: {d['n']} trades, win rate {d['wins']/d['n']:.0%}, pnl {d['pnl']:+.0f}"
                 + (" - PAUSED" if s in st["paused"] else ""))
    if not stats:
        L.append("- No data yet")

    if ranking:
        L += ["", "## New strategy search (avg out-of-sample Sharpe, share of symbols beating buy-and-hold)"]
        for spec, sh, w in ranking[:5]:
            L.append(f"- {label(spec)}: Sharpe {sh:.2f}, beat B&H on {w:.0%}")

    L += ["", "## Stock research shortlist (fundamentals rank, NOT a buy recommendation)",
          "```", shortlist, "```", "",
          "_Not financial advice. Paper results do not guarantee live results._"]

    os.makedirs("reports", exist_ok=True)
    text = "\n".join(L)
    for path in (f"reports/{today}.md", "reports/latest.md"):
        with open(path, "w") as f:
            f.write(text)
    print(text)

if __name__ == "__main__":
    run()
