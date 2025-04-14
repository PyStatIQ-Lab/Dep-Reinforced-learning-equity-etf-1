import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from trading_env import StockTradingEnv
from train_model import train_model
from stable_baselines3 import PPO

# Configuration
st.set_page_config(layout="wide")
st.title("🧩 RL Trading Environment for Stocks & ETFs")

# Sidebar controls
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Model",
    ["PPO", "A2C", "DQN"]
)
training_steps = st.sidebar.slider(
    "Training Steps",
    min_value=10000,
    max_value=500000,
    value=100000,
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

selected_stocks = st.multiselect("Select Stocks", stocks, default=stocks[:5])
selected_etfs = st.multiselect("Select ETFs", etfs, default=etfs[:5])
tickers = selected_stocks + selected_etfs

if st.button("Train Model"):
    if not tickers:
        st.error("Please select at least one stock or ETF")
    else:
        with st.spinner("Training model..."):
            env = StockTradingEnv(tickers)
            model = train_model(tickers, training_steps)
            st.success("Model training completed!")
            
            # Test the trained model
            obs = env.reset()
            done = False
            portfolio_values = []
            
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                portfolio_values.append(info["portfolio_value"])
            
            # Plot results
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(portfolio_values, label="Portfolio Value")
            ax.set_title("Portfolio Performance During Testing")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            st.pyplot(fig)

# Display market data
if tickers:
    st.header("Market Data Overview")
    selected_ticker = st.selectbox("Select ticker to view", tickers)
    
    data = yf.download(selected_ticker, period="1y")
    st.line_chart(data['Close'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Prices")
        st.dataframe(data.tail(10))
    with col2:
        st.subheader("Statistics")
        st.json({
            "Current Price": data['Close'].iloc[-1],
            "52 Week High": data['High'].max(),
            "52 Week Low": data['Low'].min(),
            "Average Volume": int(data['Volume'].mean())
        })
