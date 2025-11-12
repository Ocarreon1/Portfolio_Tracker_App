

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00cc96;
    }
    .negative {
        color: #ef553b;
    }
</style>
""", unsafe_allow_html=True)

# Asset universe
ASSET_UNIVERSE = {
    # US Equities
    'SPY (S&P 500)': 'SPY',
    'QQQ (Nasdaq 100)': 'QQQ', 
    'IWM (Russell 2000)': 'IWM',
    'DIA (Dow Jones)': 'DIA',
    'VTI (Total Stock Market)': 'VTI',
    
    # International Equities
    'EFA (International Stocks)': 'EFA',
    'VEA (Developed Markets)': 'VEA',
    'IEUR (Europe)': 'IEUR',
    
    # Bonds
    'BND (Total Bond)': 'BND',
    'AGG (Aggregate Bond)': 'AGG',
    
    # Commodities
    'GLD (Gold)': 'GLD',
    'SLV (Silver)': 'SLV',
    
    # Crypto
    'BTC (Bitcoin)': 'BTC-USD',
    'ETH (Ethereum)': 'ETH-USD', 
    'SOL (Solana)': 'SOL-USD',
    'ADA (Cardano)': 'ADA-USD',
}

# Your existing portfolio functions (slightly modified for Streamlit)
def fetch_asset_data(ticker: str, asset_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data for a single asset"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df[['Close']].rename(columns={'Close': asset_key})
        return df
        
    except Exception as e:
        st.error(f"Error fetching {asset_key}: {e}")
        return pd.DataFrame()

def fetch_all_data(asset_keys: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data for all assets"""
    all_dfs = []
    
    for key in asset_keys:
        ticker = ASSET_UNIVERSE[key]
        df = fetch_asset_data(ticker, key, start_date, end_date)
        
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        st.error("No data could be fetched for any asset")
        return pd.DataFrame()
    
    master_df = pd.concat(all_dfs, axis=1, join='outer')
    master_df = master_df.ffill().bfill()
    
    return master_df

def run_portfolio_simulation(initial_investment: float, asset_keys: list[str], 
                           weights: list[float], start_date: str, end_date: str) -> pd.DataFrame:
    """Run portfolio simulation with monthly rebalancing"""
    prices_df = fetch_all_data(asset_keys, start_date, end_date)
    
    if prices_df.empty:
        return pd.DataFrame()
    
    # Get rebalance dates (first of each month)
    rebalance_dates = []
    for year in prices_df.index.year.unique():
        for month in prices_df.index.month.unique():
            month_data = prices_df[(prices_df.index.year == year) & (prices_df.index.month == month)]
            if not month_data.empty:
                rebalance_dates.append(month_data.index[0])
    
    # Run simulation
    current_shares = {key: 0.0 for key in asset_keys}
    portfolio_history = []
    
    for i, (date, prices) in enumerate(prices_df.iterrows()):
        current_rebalance = date in rebalance_dates
        
        if current_rebalance:
            if portfolio_history:
                portfolio_value = sum(current_shares[key] * prices[key] for key in asset_keys)
            else:
                portfolio_value = initial_investment
            
            for key, weight in zip(asset_keys, weights):
                target_value = portfolio_value * weight
                current_shares[key] = target_value / prices[key]
        
        daily_values = {'date': date}
        total_value = 0
        
        for key in asset_keys:
            asset_value = current_shares[key] * prices[key]
            daily_values[f'{key}_value'] = asset_value
            total_value += asset_value
        
        daily_values['total_value'] = total_value
        portfolio_history.append(daily_values)
    
    result_df = pd.DataFrame(portfolio_history)
    result_df = result_df.set_index('date')
    result_df['daily_return'] = result_df['total_value'].pct_change()
    result_df['cumulative_return'] = (result_df['total_value'] / initial_investment) - 1
    
    return result_df

def calculate_metrics(portfolio_df: pd.DataFrame, initial_investment: float) -> dict:
    """Calculate portfolio performance metrics"""
    final_value = portfolio_df['total_value'].iloc[-1]
    total_return = (final_value - initial_investment) / initial_investment
    
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    daily_returns = portfolio_df['daily_return'].dropna()
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    cumulative_max = portfolio_df['total_value'].cummax()
    drawdown = (portfolio_df['total_value'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return {
        'initial_value': initial_investment,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_days': days
    }

# Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Portfolio Performance Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Portfolio Configuration")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime(2024, 1, 1))
        
        # Initial investment
        initial_investment = st.number_input(
            "Initial Investment ($)", 
            min_value=1000, 
            max_value=1000000, 
            value=10000,
            step=1000
        )
        
        st.markdown("---")
        st.subheader("Asset Allocation")
        
        # Asset selection and weights
        assets = []
        weights = []
        
        for i in range(3):
            col1, col2 = st.columns([3, 1])
            with col1:
                asset = st.selectbox(
                    f"Asset {i+1}",
                    options=list(ASSET_UNIVERSE.keys()),
                    key=f"asset_{i}"
                )
                assets.append(asset)
            with col2:
                weight = st.number_input(
                    "Weight %",
                    min_value=0,
                    max_value=100,
                    value=33 if i < 2 else 34,
                    key=f"weight_{i}"
                )
                weights.append(weight / 100)
        
        # Validate weights
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            st.error(f"⚠️ Weights must sum to 100% (current: {total_weight*100:.1f}%)")
            st.stop()
        
        # Run simulation button
        run_simulation = st.button("🚀 Run Portfolio Simulation", type="primary")
    
    # Main content area
    if run_simulation:
        with st.spinner("Running portfolio simulation..."):
            # Run simulation
            portfolio_df = run_portfolio_simulation(
                initial_investment, assets, weights, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if portfolio_df.empty:
                st.error("Failed to run simulation. Please check your inputs.")
                return
            
            # Calculate metrics
            metrics = calculate_metrics(portfolio_df, initial_investment)
            
            # Display metrics
            st.markdown("## 📊 Portfolio Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Initial Investment", 
                    f"${metrics['initial_value']:,.0f}"
                )
                st.metric(
                    "Final Value", 
                    f"${metrics['final_value']:,.0f}",
                    f"${metrics['final_value'] - metrics['initial_value']:,.0f}"
                )
            
            with col2:
                return_color = "positive" if metrics['total_return'] >= 0 else "negative"
                st.metric(
                    "Total Return", 
                    f"{metrics['total_return']:.2%}",
                    delta=f"{metrics['annualized_return']:.2%} annualized"
                )
                st.metric("Volatility", f"{metrics['volatility']:.2%}")
            
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            
            with col4:
                st.metric("Period", f"{metrics['total_days']} days")
                st.metric("Best Day", f"{portfolio_df['daily_return'].max():.2%}")
            
            # Charts
            st.markdown("## 📈 Portfolio Charts")
            
            tab1, tab2, tab3 = st.tabs(["Portfolio Value", "Performance", "Asset Allocation"])
            
            with tab1:
                # Portfolio value over time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['total_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Cumulative returns
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['cumulative_return'] * 100,
                    mode='lines',
                    name='Cumulative Return',
                    line=dict(color='#00cc96', width=3)
                ))
                fig.update_layout(
                    title="Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Asset allocation pie chart
                asset_values = [portfolio_df[f'{asset}_value'].iloc[-1] for asset in assets]
                fig = go.Figure(data=[go.Pie(
                    labels=assets,
                    values=asset_values,
                    hole=.3,
                    marker_colors=px.colors.qualitative.Set3
                )])
                fig.update_layout(
                    title="Current Asset Allocation",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Raw data
            with st.expander("View Raw Data"):
                st.dataframe(portfolio_df.tail(10))
                
            # Download button
            csv = portfolio_df.to_csv()
            st.download_button(
                label="📥 Download Portfolio Data",
                data=csv,
                file_name="portfolio_data.csv",
                mime="text/csv"
            )

    else:
        # Welcome message when no simulation has been run
        st.markdown("""
        ## Welcome to the Portfolio Performance Tracker! 🎯
        
        This tool helps you analyze how your investment portfolio would have performed
        with monthly rebalancing.
        
        ### How to use:
        1. **Configure** your portfolio in the sidebar
        2. **Select** 3 assets and set their weights (must sum to 100%)
        3. **Choose** your initial investment and date range
        4. **Click** "Run Portfolio Simulation" to see the results!
        
        ### Features:
        - 📊 Performance metrics and analytics
        - 📈 Interactive charts and visualizations  
        - 💰 Portfolio value tracking
        - 📉 Risk analysis (volatility, drawdowns)
        - 📥 Downloadable results
        
        Get started by configuring your portfolio in the sidebar! →
        """)
        
        # Sample portfolio preview
        st.markdown("### 💡 Sample Portfolio Ideas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Conservative**")
            st.write("60% SPY\n\n30% BND\n\n10% GLD")
        
        with col2:
            st.markdown("**Balanced**") 
            st.write("50% SPY\n\n30% QQQ\n\n20% BTC")
        
        with col3:
            st.markdown("**Growth**")
            st.write("40% QQQ\n\n40% BTC\n\n20% ETH")

if __name__ == "__main__":
    main()