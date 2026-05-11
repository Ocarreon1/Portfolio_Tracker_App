import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- APP CONFIG ---
st.set_page_config(page_title="Oscar's Barbell Strategy", layout="wide")

ASSET_UNIVERSE = {
    'QQQ (US Tech)': 'QQQ', 
    'KWEB (China)': 'KWEB',
    'URA (Uranium)': 'URA',
    'IAU (Gold)': 'IAU'
}

# --- CALCULATOR ENGINE ---
@st.cache_data(ttl=3600)
def fetch_data(tickers, start, end):
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True)['Close']
    return data

def run_inflow_simulation(initial_usd, monthly_usd, assets, start, end):
    tickers = [ASSET_UNIVERSE[a] for a in assets]
    prices = fetch_data(tickers, start, end).ffill().bfill()
    if prices.empty: return pd.DataFrame(), 0
    
    # 1. Initialize Portfolio (Equally weighted)
    shares = {t: (initial_usd * 0.25) / prices.iloc[0][t] for t in tickers}
    total_invested = initial_usd
    history = []
    
    # Get monthly contribution dates
    contribution_dates = prices.groupby([prices.index.year, prices.index.month]).head(1).index

    for date, row in prices.iterrows():
        # 2. Check for Monthly Contribution
        if date in contribution_dates and date != prices.index[0]:
            # Current Portfolio Value before contribution
            current_val = sum(shares[t] * row[t] for t in tickers)
            new_total_val = current_val + monthly_usd
            total_invested += monthly_usd
            
            # Find the "Laggard" (Asset furthest from 25% target)
            deviations = {}
            for t in tickers:
                actual_weight = (shares[t] * row[t]) / current_val
                deviations[t] = actual_weight - 0.25
            
            # Identify the ticker with the most negative deviation
            laggard_ticker = min(deviations, key=deviations.get)
            
            # Buy shares of ONLY the laggard
            shares[laggard_ticker] += monthly_usd / row[laggard_ticker]

        # 3. Daily Value Tracking
        daily_val = sum(shares[t] * row[t] for t in tickers)
        history.append({
            'date': date, 
            'total_value': daily_val, 
            'invested': total_invested,
            'pnl': daily_val - total_invested
        })
        
    df = pd.DataFrame(history).set_index('date')
    return df, total_invested

# --- UI ---
st.title("📈 The Barbell Strategy: Inflow Rebalancing")
st.markdown("This tracker simulates **monthly contributions** to buy the underrepresented asset.")

with st.sidebar:
    st.header("Financial Inputs")
    initial_mxn = st.number_input("Initial Deposit (MXN)", value=10000)
    monthly_mxn = st.number_input("Monthly Contribution (MXN)", value=1500)
    fx_rate = st.number_input("Est. USD/MXN", value=17.5)
    
    # Convert to USD for YFinance compatibility
    initial_usd = initial_mxn / fx_rate
    monthly_usd = monthly_mxn / fx_rate
    
    st.divider()
    start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    end_date = st.date_input("End Date", value=datetime.now())

if st.button("Calculate My Real Journey"):
    results, total_inv = run_inflow_simulation(initial_usd, monthly_usd, list(ASSET_UNIVERSE.keys()), start_date, end_date)
    
    if not results.empty:
        final_val = results['total_value'].iloc[-1]
        pnl_pct = (final_val / total_inv) - 1
        
        # Display Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Balance (USD)", f"${final_val:,.2f}")
        c2.metric("Total Invested (USD)", f"${total_inv:,.2f}")
        c3.metric("Total ROI", f"{pnl_pct:.2%}")

        # Chart 1: Value vs. Invested (The "Gap" is your profit)
        st.subheader("Portfolio Growth vs. Total Contributions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.index, y=results['total_value'], name="Portfolio Value", fill='tonexty'))
        fig.add_trace(go.Scatter(x=results.index, y=results['invested'], name="Total Principal", line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"By the end of this period, your 1,500 MXN monthly contribution has built a principal of **${total_inv * fx_rate:,.0f} MXN**.")
