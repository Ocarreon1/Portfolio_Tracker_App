import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Portfolio Performance Tracker",
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

# Asset universe with clear names
ASSET_UNIVERSE = {
    # US Equities
    'SPY (S&P 500)': 'SPY',
    'QQQ (Nasdaq 100)': 'QQQ', 
    'VTI (Total Stock Market)': 'VTI',
    
    # International
    'EFA (International Stocks)': 'EFA',
    'IEUR (Europe Stocks)': 'IEUR',
    
    # Bonds
    'BND (Total Bond)': 'BND',
    
    # Commodities
    'GLD (Gold)': 'GLD',
    
    # Crypto
    'BTC (Bitcoin)': 'BTC-USD',
    'ETH (Ethereum)': 'ETH-USD', 
    'SOL (Solana)': 'SOL-USD',
}

def fetch_asset_data(ticker, asset_key, start_date, end_date):
    """Fetch data for a single asset"""
    try:
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
    """Run portfolio simulation with monthly rebalancing"""
    # Fetch data for all assets
    all_dfs = []
    for asset in assets:
        ticker = ASSET_UNIVERSE[asset]
        df = fetch_asset_data(ticker, asset, start_date, end_date)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        st.error("No data could be fetched for any asset")
        return pd.DataFrame()
    
    # Merge data
    prices_df = pd.concat(all_dfs, axis=1, join='outer')
    prices_df = prices_df.ffill().bfill()
    
    if prices_df.empty:
        st.error("No valid data after merging")
        return pd.DataFrame()
    
    # Get rebalance dates (first trading day of each month)
    rebalance_dates = []
    for year in prices_df.index.year.unique():
        for month in prices_df.index.month.unique():
            month_data = prices_df[(prices_df.index.year == year) & (prices_df.index.month == month)]
            if not month_data.empty:
                rebalance_dates.append(month_data.index[0])
    
    # Run simulation
    current_shares = {asset: 0.0 for asset in assets}
    portfolio_history = []
    
    for i, (date, prices) in enumerate(prices_df.iterrows()):
        current_rebalance = date in rebalance_dates
        
        if current_rebalance:
            # Calculate current portfolio value
            if portfolio_history:
                portfolio_value = sum(current_shares[asset] * prices[asset] for asset in assets)
            else:
                portfolio_value = initial_investment
            
            # Rebalance
            for asset, weight in zip(assets, weights):
                target_value = portfolio_value * weight
                current_shares[asset] = target_value / prices[asset]
        
        # Calculate daily values
        daily_values = {'date': date}
        total_value = 0
        
        for asset in assets:
            asset_value = current_shares[asset] * prices[asset]
            daily_values[f'{asset}_value'] = asset_value
            total_value += asset_value
        
        daily_values['total_value'] = total_value
        portfolio_history.append(daily_values)
    
    # Create results
    result_df = pd.DataFrame(portfolio_history)
    if result_df.empty:
        return result_df
        
    result_df = result_df.set_index('date')
    result_df['daily_return'] = result_df['total_value'].pct_change()
    result_df['cumulative_return'] = (result_df['total_value'] / initial_investment) - 1
    
    return result_df

def calculate_metrics(portfolio_df, initial_investment):
    """Calculate portfolio performance metrics"""
    if portfolio_df.empty:
        return {}
        
    final_value = portfolio_df['total_value'].iloc[-1]
    total_return = (final_value - initial_investment) / initial_investment
    
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = max(days / 365.25, 0.001)  # Avoid division by zero
    
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    daily_returns = portfolio_df['daily_return'].dropna()
    if len(daily_returns) > 0:
        annualized_volatility = daily_returns.std() * np.sqrt(252)
    else:
        annualized_volatility = 0
    
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

# Main App
def main():
    st.markdown('<h1 class="main-header">📈 Portfolio Performance Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Portfolio Configuration")
        
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
            value=10000,
            step=1000
        )
        
        st.markdown("---")
        st.subheader("Asset Allocation")
        
        # Asset selection - simplified for 3 assets
        assets = []
        weights = []
        
        asset_options = list(ASSET_UNIVERSE.keys())
        
        for i in range(3):
            col1, col2 = st.columns([3, 1])
            with col1:
                asset = st.selectbox(
                    f"Asset {i+1}",
                    options=asset_options,
                    index=i if i < len(asset_options) else 0,
                    key=f"asset_{i}"
                )
                assets.append(asset)
            with col2:
                # Default weights: 40%, 40%, 20%
                default_weight = 40 if i < 2 else 20
                weight = st.number_input(
                    "Weight %",
                    min_value=0,
                    max_value=100,
                    value=default_weight,
                    key=f"weight_{i}"
                )
                weights.append(weight / 100.0)
        
        # Validate weights
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            st.error(f"⚠️ Weights must sum to 100% (current: {total_weight*100:.1f}%)")
            st.stop()
        
        if st.button("🚀 Run Portfolio Simulation", type="primary", use_container_width=True):
            st.session_state.run_simulation = True
        else:
            st.session_state.run_simulation = False

    # Main content
    if st.session_state.get('run_simulation', False):
        with st.spinner("Running portfolio simulation... This may take a few seconds."):
            portfolio_df = run_portfolio_simulation(
                initial_investment, assets, weights, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
        
        if portfolio_df.empty:
            st.error("❌ Simulation failed. Please try different assets or date range.")
            return
        
        # Calculate metrics
        metrics = calculate_metrics(portfolio_df, initial_investment)
        
        if not metrics:
            st.error("❌ Could not calculate metrics.")
            return
        
        # Display results
        st.success("✅ Simulation completed successfully!")
        
        # Performance Metrics
        st.markdown("## 📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Initial Investment", f"${metrics['initial_value']:,.0f}")
            st.metric("Final Value", f"${metrics['final_value']:,.0f}")
        
        with col2:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
            st.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
        
        with col3:
            st.metric("Volatility", f"{metrics['volatility']:.2%}")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            st.metric("Period", f"{metrics['total_days']} days")
        
        # Charts
        st.markdown("## 📈 Portfolio Charts")
        
        tab1, tab2, tab3 = st.tabs(["Portfolio Value", "Performance", "Asset Allocation"])
        
        with tab1:
            # Portfolio value chart
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
                height=500,
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        xref="paper",
                        yref="paper",
                        text="Source: Yahoo! Finance",
                        showarrow=False,
                        font=dict(size=12, color="gray"),
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Cumulative returns chart
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
                height=500,
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        xref="paper",
                        yref="paper",
                        text="Source: Yahoo! Finance",
                        showarrow=False,
                        font=dict(size=12, color="gray"),
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Asset allocation pie chart
            asset_values = [portfolio_df[f'{asset}_value'].iloc[-1] for asset in assets]
            fig = go.Figure(data=[go.Pie(
                labels=assets,
                values=asset_values,
                hole=0.3,
                marker_colors=px.colors.qualitative.Set3
            )])
            fig.update_layout(
                title="Current Asset Allocation",
                height=500,
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.1,
                        xref="paper",
                        yref="paper",
                        text="Source: Yahoo! Finance",
                        showarrow=False,
                        font=dict(size=12, color="gray"),
                    )
                ] + [  # Keep the pie chart annotations
                    dict(
                        text="Allocation",
                        x=0.5,
                        y=0.5,
                        font_size=20,
                        showarrow=False,
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Global footer
        st.markdown("---")
        st.markdown(
            '<div class="footer">'
            'Data Source: Yahoo! Finance | '
            'Built with Streamlit | '
            'For educational purposes only'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Raw data
        with st.expander("📋 View Raw Data"):
            st.dataframe(portfolio_df.tail(10))
            
            # Download button
            csv = portfolio_df.to_csv()
            st.download_button(
                label="📥 Download Portfolio Data as CSV",
                data=csv,
                file_name="portfolio_data.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Portfolio Performance Tracker! 🎯
        
        This tool simulates how your investment portfolio would have performed with **monthly rebalancing**.
        
        ### 🚀 How to use:
        1. **Configure** your portfolio in the sidebar
        2. **Select** 3 assets and set their weights (must sum to 100%)
        3. **Choose** your initial investment and date range
        4. **Click** "Run Portfolio Simulation" to see the magic!
        
        ### 📊 What you'll see:
        - Performance metrics and analytics
        - Interactive charts and visualizations  
        - Portfolio value tracking over time
        - Risk analysis (volatility, drawdowns)
        - Downloadable results
        
        **Ready to get started?** Configure your portfolio in the sidebar! →
        """)
        
        # Sample portfolios
        st.markdown("### 💡 Sample Portfolio Ideas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🛡️ Conservative**")
            st.write("""
            - 60% SPY (S&P 500)
            - 30% BND (Total Bond)  
            - 10% GLD (Gold)
            """)
        
        with col2:
            st.markdown("**⚖️ Balanced**")
            st.write("""
            - 40% SPY (S&P 500)
            - 40% QQQ (Nasdaq 100)
            - 20% BTC (Bitcoin)
            """)
        
        with col3:
            st.markdown("**🚀 Growth**")
            st.write("""
            - 50% QQQ (Nasdaq 100)
            - 30% BTC (Bitcoin)
            - 20% ETH (Ethereum)
            """)
        
        # Footer on welcome page too
        st.markdown("---")
        st.markdown(
            '<div class="footer">'
            'Data Source: Yahoo! Finance | '
            'Built with Streamlit | '
            'For educational purposes only'
            '</div>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
