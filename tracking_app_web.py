import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Aggressive Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Expanded Asset Universe
ASSET_UNIVERSE = {
    # US Equities & Growth
    'SPY (S&P 500)': 'SPY',
    'QQQ (Nasdaq 100)': 'QQQ', 
    'VUG (US Growth)': 'VUG',
    'IWM (Small Cap)': 'IWM',
    
    # Aggressive / Thematic (The Barbell)
    'KWEB (China Internet)': 'KWEB',
    'URA (Uranium Miners)': 'URA',
    'COPX (Copper Miners)': 'COPX',
    'PAVE (US Infrastructure)': 'PAVE',
    
    # International & EM
    'VWO (Emerging Markets)': 'VWO',
    'IEUR (Europe Stocks)': 'IEUR',
    'EFA (International Stocks)': 'EFA',
    
    # Fixed Income & Credit
    'BND (Total Bond)': 'BND',
    'ANGL (Fallen Angel High Yield)': 'ANGL',
    'EMLC (EM Local Currency Bonds)': 'EMLC',
    
    # Commodities & Hard Assets
    'GLD (Gold)': 'GLD',
    'IAU (Gold MiniShares)': 'IAU',
    'VNQ (Real Estate)': 'VNQ',
    
    # Crypto
    'BTC (Bitcoin)': 'BTC-USD',
    'ETH (Ethereum)': 'ETH-USD', 
}

def fetch_asset_data(ticker, asset_key, start_date, end_date):
    try:
        # Use period to ensure we get enough data for the date range
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        if df.empty:
            return pd.DataFrame()
        df = df[['Close']].rename(columns={'Close': asset_key})
        return df
    except Exception as e:
        st.error(f"Error fetching {asset_key}: {e}")
        return pd.DataFrame()

def run_portfolio_simulation(initial_investment, assets, weights, start_date, end_date):
    all_dfs = []
    for asset in assets:
        ticker = ASSET_UNIVERSE[asset]
        df = fetch_asset_data(ticker, asset, start_date, end_date)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    prices_df = pd.concat(all_dfs, axis=1, join='outer')
    prices_df = prices_df.ffill().bfill()
    
    rebalance_dates = []
    for year in prices_df.index.year.unique():
        for month in prices_df.index.month.unique():
            month_data = prices_df[(prices_df.index.year == year) & (prices_df.index.month == month)]
            if not month_data.empty:
                rebalance_dates.append(month_data.index[0])
    
    current_shares = {asset: 0.0 for asset in assets}
    portfolio_history = []
    
    for i, (date, prices) in enumerate(prices_df.iterrows()):
        if date in rebalance_dates:
            if portfolio_history:
                portfolio_value = sum(current_shares[asset] * prices[asset] for asset in assets)
            else:
                portfolio_value = initial_investment
            
            for asset, weight in zip(assets, weights):
                target_value = portfolio_value * weight
                current_shares[asset] = target_value / prices[asset]
        
        daily_values = {'date': date}
        total_value = 0
        for asset in assets:
            asset_value = current_shares[asset] * prices[asset]
            daily_values[f'{asset}_value'] = asset_value
            total_value += asset_value
        
        daily_values['total_value'] = total_value
        portfolio_history.append(daily_values)
    
    result_df = pd.DataFrame(portfolio_history).set_index('date')
    result_df['daily_return'] = result_df['total_value'].pct_change()
    result_df['cumulative_return'] = (result_df['total_value'] / initial_investment) - 1
    return result_df

def calculate_metrics(portfolio_df, initial_investment):
    if portfolio_df.empty: return {}
    final_value = portfolio_df['total_value'].iloc[-1]
    total_return = (final_value - initial_investment) / initial_investment
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = max(days / 365.25, 0.001)
    annualized_return = (1 + total_return) ** (1 / years) - 1
    volatility = portfolio_df['daily_return'].std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
    max_drawdown = ((portfolio_df['total_value'] - portfolio_df['total_value'].cummax()) / portfolio_df['total_value'].cummax()).min()
    
    return {
        'initial_value': initial_investment, 'final_value': final_value,
        'total_return': total_return, 'annualized_return': annualized_return,
        'volatility': volatility, 'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown, 'total_days': days
    }

def main():
    st.markdown('<h1 class="main-header">📈 Aggressive Portfolio Tracker</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🎯 Configuration")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=1095))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000)
        
        st.markdown("---")
        st.subheader("Asset Allocation")
        
        # Select number of assets
        num_assets = st.slider("Number of assets to include", 1, 5, 4)
        
        assets = []
        weights = []
        asset_options = list(ASSET_UNIVERSE.keys())
        
        # Dynamic Weight Defaults for convenience (4 assets = 25% each)
        default_w = round(100 / num_assets)
        
        for i in range(num_assets):
            c1, c2 = st.columns([3, 1])
            with c1:
                asset = st.selectbox(f"Asset {i+1}", options=asset_options, index=i % len(asset_options), key=f"a_{i}")
                assets.append(asset)
            with c2:
                weight = st.number_input("%", min_value=0, max_value=100, value=default_w, key=f"w_{i}")
                weights.append(weight / 100.0)
        
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            st.error(f"⚠️ Weights must sum to 100% (Current: {total_weight*100:.1f}%)")
            st.stop()
        
        run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    if run_sim:
        with st.spinner("Analyzing macro trends and fetching data..."):
            portfolio_df = run_portfolio_simulation(initial_investment, assets, weights, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if not portfolio_df.empty:
            metrics = calculate_metrics(portfolio_df, initial_investment)
            st.success("✅ Analysis Complete")
            
            # Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Value", f"${metrics['final_value']:,.2f}")
            c2.metric("Total Return", f"{metrics['total_return']:.2%}")
            c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            c4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

            # Tabs for Charts
            tab1, tab2, tab3 = st.tabs(["Performance", "Relative Returns", "Final Allocation"])
            with tab1:
                fig = px.line(portfolio_df, y='total_value', title="Growth of $10,000 (Monthly Rebalanced)")
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                fig = px.line(portfolio_df, y='cumulative_return', title="Cumulative Portfolio Return (%)")
                st.plotly_chart(fig, use_container_width=True)
            with tab3:
                last_vals = [portfolio_df[f'{a}_value'].iloc[-1] for a in assets]
                fig = px.pie(values=last_vals, names=assets, hole=0.4, title="Terminal Allocation")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data found for the selected tickers or date range.")

if __name__ == "__main__":
    main()
