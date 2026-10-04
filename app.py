import streamlit as st
import pandas as pd
from trading_agent import get_data, STRATEGIES, backtest, screen_df

st.set_page_config(page_title="Trading Research Agent", layout="wide")
st.title("Trading Research Agent")
st.caption("Backtesting and screening only. Not financial advice.")

tab1, tab2 = st.tabs(["Backtest", "Stock screener"])

with tab1:
    c1, c2 = st.columns(2)
    symbol = c1.text_input("Symbol (e.g. ITC.NS, BTC-USD)", "ITC.NS")
    strat = c2.selectbox("Strategy", list(STRATEGIES.keys()))
    if st.button("Run backtest"):
        with st.spinner("Running..."):
            df = get_data(symbol)
            if df.empty:
                st.error("No data found. Check the symbol.")
            else:
                pos = STRATEGIES[strat](df)
                split = int(len(df) * 0.8)
                ins = backtest(df.iloc[:split], pos.iloc[:split])
                oos = backtest(df.iloc[split:], pos.iloc[split:])
                st.subheader("In-sample (history the strategy was tuned on)")
                st.dataframe(pd.DataFrame([ins]).round(3))
                st.subheader("Out-of-sample (last 20%, the honest test)")
                st.dataframe(pd.DataFrame([oos]).round(3))
                ret = df["Close"].pct_change().fillna(0)
                eq = (1 + pos.shift(1).fillna(0) * ret).cumprod()
                bh = (1 + ret).cumprod()
                st.line_chart(pd.DataFrame({"Strategy": eq, "Buy & hold": bh}))

with tab2:
    st.write("Ranks a starter NSE list on ROE, growth, PE and debt. A shortlist for research, not buy signals.")
    if st.button("Run screener"):
        with st.spinner("Fetching fundamentals..."):
            st.dataframe(screen_df())
