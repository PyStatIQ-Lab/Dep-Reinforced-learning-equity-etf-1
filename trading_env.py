import gym
from gym import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from gym.utils import seeding
from typing import Tuple, Dict, Any

class StockTradingEnv(gym.Env):
    """
    Custom RL Environment for Stock/ETF Trading
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, tickers: list, initial_balance: float = 100000, lookback: int = 30):
        super(StockTradingEnv, self).__init__()
        
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_balance = initial_balance
        self.lookback = lookback
        
        # Action space: [-1, 1] for each asset (continuous)
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # Observation space: OHLCV + indicators + portfolio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_assets * (5 + 3) * lookback + self.n_assets + 1,),
            dtype=np.float32
        )
        
        self.data = self._download_data()
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _download_data(self) -> Dict[str, pd.DataFrame]:
        """Download historical data using yfinance"""
        data = {}
        for ticker in self.tickers:
            df = yf.download(ticker, period="5y", interval="1d")
            df = self._add_technical_indicators(df)
            data[ticker] = df
        return data

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df.dropna()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.holdings = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = [self.initial_balance]
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Flatten all observations into single vector"""
        price_data = np.zeros((self.n_assets, self.lookback, 5))
        indicator_data = np.zeros((self.n_assets, self.lookback, 3))
        
        for i, ticker in enumerate(self.tickers):
            df = self.data[ticker]
            start_idx = self.current_step - self.lookback
            end_idx = self.current_step
            
            price_data[i] = df.iloc[start_idx:end_idx][['Open', 'High', 'Low', 'Close', 'Volume']].values
            indicator_data[i] = df.iloc[start_idx:end_idx][['RSI', 'MACD', 'ATR']].values
        
        # Flatten everything
        obs = np.concatenate([
            price_data.flatten(),
            indicator_data.flatten(),
            np.array([self.holdings[ticker] for ticker in self.tickers]),
            np.array([self.balance])
        ])
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        assert len(action) == self.n_assets, f"Expected {self.n_assets} actions, got {len(action)}"
        
        # Get current prices
        current_prices = np.array([
            self.data[ticker].iloc[self.current_step]['Close'] 
            for ticker in self.tickers
        ])
        
        # Execute trades
        for i, ticker in enumerate(self.tickers):
            action_val = action[i]
            price = current_prices[i]
            
            if action_val > 0.33:  # Buy
                max_buy = self.balance / price
                buy_amount = min(max_buy * action_val, max_buy)
                self.holdings[ticker] += buy_amount
                self.balance -= buy_amount * price
            elif action_val < -0.33:  # Sell
                sell_amount = min(self.holdings[ticker], -action_val * self.holdings[ticker])
                self.holdings[ticker] -= sell_amount
                self.balance += sell_amount * price
        
        # Update step
        self.current_step += 1
        done = self.current_step >= min(len(df) for df in self.data.values()) - 1
        
        # Calculate portfolio value
        portfolio_value = self.balance + sum(
            self.holdings[ticker] * self.data[ticker].iloc[self.current_step]['Close']
            for ticker in self.tickers
        )
        reward = portfolio_value - self.portfolio_value[-1]
        self.portfolio_value.append(portfolio_value)
        
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, done, {
            "portfolio_value": portfolio_value,
            "current_prices": current_prices,
            "actions": action
        }

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance:.2f}")
            for ticker in self.tickers:
                print(f"{ticker}: {self.holdings[ticker]:.2f} shares")
            print(f"Portfolio Value: {self.portfolio_value[-1]:.2f}")
