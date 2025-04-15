import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trading_env import StockTradingEnv
from train_model import train_model
from stable_baselines3 import PPO
from typing import List, Dict, Any
import os

# Configuration
st.set_page_config(layout="wide")
st.title("🧩 RL Trading Platform for Stocks & ETFs")

def get_ticker_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Get historical data for a single ticker"""
    return yf.download(ticker, period=period)

def plot_portfolio_history(history: Dict[str, Any]) -> None:
    """Plot portfolio performance and actions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Portfolio value
    ax1.plot(history['portfolio_values'], label="Portfolio Value", color='blue')
    ax1.set_title("Portfolio Performance")
    ax1.set_ylabel("Value (₹)")
    ax1.grid(True)
    
    # Actions
    actions = np.array(history['actions'])
    for i in range(actions.shape[1]):
        ax2.plot(actions[:, i], label=f"Asset {i+1}")
    ax2.set_title("Model Actions Over Time")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Action Value")
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)

def main():
    """Main Streamlit application"""
    # Sidebar configuration
    st.sidebar.header("Configuration")
    training_steps = st.sidebar.slider(
        "Training Steps", 
        min_value=10000, 
        max_value=500000, 
        value=50000, 
        step=10000
    )
    
    # Stock and ETF lists
    stocks = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
        "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
        "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
        "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
        "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS"
    ]
    
    etfs = [
        "MAFANG.NS", "FMCGIETF.NS", "MOGSEC.NS", "TATAGOLD.NS",
        "GOLDIETF.NS", "GOLDCASE.NS", "HDFCGOLD.NS", "GOLD1.NS",
        "AXISGOLD.NS", "GOLD360.NS", "ABGSEC.NS", "SETFGOLD.NS",
        "GOLDBEES.NS", "LICMFGOLD.NS"
    ]
    
    # Asset selection
    selected_stocks = st.multiselect("Select Stocks", stocks, default=["ASIANPAINT.NS", "HDFCBANK.NS"])
    selected_etfs = st.multiselect("Select ETFs", etfs, default=["GOLDBEES.NS"])
    tickers = selected_stocks + selected_etfs
    
    # Main content
    if st.button("Train and Evaluate Model"):
        if not tickers:
            st.error("Please select at least one stock or ETF")
        else:
            with st.spinner("Training model. This may take several minutes..."):
                try:
                    # Train model
                    model = train_model(tickers, training_steps)
                    st.success("Model training completed!")
                    
                    # Test the trained model
                    env = StockTradingEnv(tickers)
                    obs = env.reset()
                    done = False
                    history = {
                        'portfolio_values': [],
                        'actions': [],
                        'prices': []
                    }
                    
                    while not done:
                        action, _ = model.predict(obs)
                        obs, reward, done, info = env.step(action)
                        history['portfolio_values'].append(info["portfolio_value"])
                        history['actions'].append(info["actions"])
                        history['prices'].append(info["current_prices"])
                    
                    # Display results
                    st.subheader("Training Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Final Portfolio Value", 
                            f"₹{history['portfolio_values'][-1]:,.2f}",
                            delta=f"₹{history['portfolio_values'][-1] - env.initial_balance:,.2f}"
                        )
                    with col2:
                        returns = (history['portfolio_values'][-1] - env.initial_balance) / env.initial_balance * 100
                        st.metric(
                            "Total Return", 
                            f"{returns:.2f}%"
                        )
                    
                    # Plot performance
                    plot_portfolio_history(history)
                    
                    # Show sample recommendations
                    st.subheader("Sample Recommendations")
                    last_actions = history['actions'][-1]
                    recommendations = pd.DataFrame({
                        "Ticker": tickers,
                        "Action": last_actions,
                        "Recommendation": np.select([
                            last_actions > 0.33,
                            last_actions < -0.33,
                            (last_actions >= -0.33) & (last_actions <= 0.33)
                        ], [
                            "Strong Buy",
                            "Strong Sell",
                            "Hold"
                        ]),
                        "Current Price": [f"₹{p:,.2f}" for p in history['prices'][-1]]
                    })
                    st.dataframe(recommendations)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check the console for details")
    
    # Market data display
    if tickers:
        st.sidebar.header("Market Data")
        selected_ticker = st.sidebar.selectbox("View Ticker Data", tickers)
        period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
        
        data = get_ticker_data(selected_ticker, period)
        st.subheader(f"Market Data for {selected_ticker}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.line_chart(data['Close'])
        with col2:
            st.metric("Current Price", f"₹{data['Close'].iloc[-1]:,.2f}")
            st.metric("52W High", f"₹{data['High'].max():,.2f}")
            st.metric("52W Low", f"₹{data['Low'].min():,.2f}")
        
        st.write("Recent Prices")
        st.dataframe(data.tail(10))

if __name__ == "__main__":
    main()
