import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Oscar's Barbell Tracker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .footer { text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# --- EXPANDED ASSET UNIVERSE ---
ASSET_UNIVERSE = {
    # US & Growth
    'QQQ (Nasdaq 100)': 'QQQ', 
    'SPY (S&P 500)': 'SPY',
    'VTI (Total US Market)': 'VTI',
    
    # China & Emerging Markets
    'KWEB (China Internet)': 'KWEB',
    'MCHI (MSCI China)': 'MCHI',
    'VWO (Emerging Markets)': 'VWO',
    
    # Thematic / Future
    'URA (Uranium Miners)': 'URA',
    'COPX (Copper Miners)': 'COPX',
    'SMH (Semiconductors)': 'SMH',
    
    # Global Fixed Income (No MX exposure)
    'BNDW (Total World Bond)': 'BNDW',
    'EMLC (EM Local Currency Bond)': 'EMLC',
    'BWX (Intl Treasury)': 'BWX',
    
    # Commodities / Ballast
    'IAU (Gold MiniShares)': 'IAU',
    'GLD (Gold Shares)': 'GLD',
    'SLV (Silver)': 'SLV',
    
    # Crypto Proxy
    'BITO (Bitcoin Strategy)': 'BITO'
}

def fetch_asset_data(ticker, asset_key, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        if df.empty: return pd.DataFrame()
        return df[['Close']].rename(columns={'Close': asset_key})
    except Exception as e:
        st.error(f"Error fetching {asset_key}: {e}")
        return pd.DataFrame()

def run_inflow_simulation(initial_mxn, monthly_mxn, fx_rate, assets, weights, start_date, end_date):
    """Simulation: Contributions buy the under-target asset (No Selling)"""
    # Convert inputs to USD
    initial_usd = initial_mxn / fx_rate
    monthly_usd = monthly_mxn / fx_rate

    # Fetch Data
    all_dfs = []
    for asset in assets:
        ticker = ASSET_UNIVERSE[asset]
        df = fetch_asset_data(ticker, asset, start_date, end_date)
        if not df.empty: all_dfs.append(df)
    
    if not all_dfs: return pd.DataFrame()
    prices_df = pd.concat(all_dfs, axis=1, join='outer').ffill().bfill()
    
    # Trackers
    shares = {asset: (initial_usd * w) / prices_df.iloc[0][asset] for asset, w in zip(assets, weights)}
    total_invested_usd = initial_usd
    portfolio_history = []
    
    # Monthly contribution dates
    contribution_dates = prices_df.groupby([prices_df.index.year, prices_df.index.month]).head(1).index

    for date, prices in prices_df.iterrows():
        # Handle Monthly Inflow
        if date in contribution_dates and date != prices_df.index[0]:
            current_vals = {a: shares[a] * prices[a] for a in assets}
            total_current_val = sum(current_vals.values())
            
            # Identify "Laggard" (Asset furthest below target weight)
            deviations = {a: (current_vals[a] / total_current_val) - w for a, w in zip(assets, weights)}
            laggard = min(deviations, key=deviations.get)
            
            # Buy ONLY the laggard
            shares[laggard] += monthly_usd / prices[laggard]
            total_invested_usd += monthly_usd

        # Record Daily Snapshot
        daily_val = sum(shares[a] * prices[a] for a in assets)
        portfolio_history.append({
            'date': date,
            'total_value': daily_val,
            'invested': total_invested_usd
        })
    
    res = pd.DataFrame(portfolio_history).set_index('date')
    res['daily_return'] = res['total_value'].pct_change()
    res['cumulative_return'] = (res['total_value'] / res['invested']) - 1
    return res

# --- MAIN APP ---
def main():
    st.markdown('<h1 class="main-header">📈 Oscar\'s Barbell Strategy Tracker</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🎯 Configuration")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
        with col2: end_date = st.date_input("End Date", value=datetime.now())
        
        # Financial Premise
        st.subheader("Financial Premise")
        initial_mxn = st.number_input("Initial Investment (MXN)", value=10000, step=1000)
        monthly_mxn = st.number_input("Monthly Contribution (MXN)", value=1500, step=100)
        fx_rate = st.number_input("USD/MXN Exchange Rate", value=18.0, step=0.1)
        
        st.markdown("---")
        num_assets = st.slider("Number of Securities", 1, 5, 4)
        
        assets = []
        weights = []
        asset_options = list(ASSET_UNIVERSE.keys())
        
        for i in range(num_assets):
            c1, c2 = st.columns([3, 1])
            with c1:
                asset = st.selectbox(f"Asset {i+1}", asset_options, index=i % len(asset_options), key=f"a_{i}")
                assets.append(asset)
            with c2:
                weight = st.number_input("%", 0, 100, value=int(100/num_assets), key=f"w_{i}")
                weights.append(weight / 100.0)
        
        if abs(sum(weights) - 1.0) > 0.01:
            st.error(f"⚠️ Weights must sum to 100% (Current: {sum(weights)*100:.1f}%)")
            st.stop()
        
        run_sim = st.button("🚀 Run Portfolio Simulation", type="primary", use_container_width=True)

    if run_sim:
        with st.spinner("Calculating Inflow Rebalancing..."):
            res = run_inflow_simulation(initial_mxn, monthly_mxn, fx_rate, assets, weights, 
                                        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if not res.empty:
            st.success("✅ Journey Simulated!")
            
            # Metrics
            final_val_usd = res['total_value'].iloc[-1]
            total_inv_usd = res['invested'].iloc[-1]
            roi = (final_val_usd / total_inv_usd) - 1
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Balance (MXN)", f"${final_val_usd * fx_rate:,.2f}")
            m2.metric("Total Invested (MXN)", f"${total_inv_usd * fx_rate:,.2f}")
            m3.metric("Money-Weighted ROI", f"{roi:.2%}")

            # Charts
            st.subheader("The Wealth Gap")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.index, y=res['total_value'] * fx_rate, name="Portfolio Value (MXN)", fill='tonexty'))
            fig.add_trace(go.Scatter(x=res.index, y=res['invested'] * fx_rate, name="Principal Invested (MXN)", line=dict(dash='dash', color='gray')))
            fig.update_layout(template="plotly_white", height=500, yaxis_title="MXN")
            st.plotly_chart(fig, use_container_width=True)

            # Cumulative Return Chart
            st.subheader("Performance (%)")
            fig_pct = px.line(res, y='cumulative_return', title="Portfolio Performance vs. Weighted Principal")
            st.plotly_chart(fig_pct, use_container_width=True)
            
            with st.expander("📋 View Raw Simulation Data"):
                st.dataframe(res.tail(10))
        else:
            st.error("Simulation failed. Check your ticker availability for these dates.")

if __name__ == "__main__":
    main()
