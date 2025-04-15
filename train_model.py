import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from trading_env import StockTradingEnv
from typing import List, Optional

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values to Tensorboard
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.portfolio_values = []

    def _on_step(self) -> bool:
        # Log portfolio value
        for env in self.training_env.envs:
            if hasattr(env, 'portfolio_value'):
                self.logger.record('portfolio/value', env.portfolio_value[-1])
        return True

def train_model(
    tickers: List[str],
    total_timesteps: int = 100000,
    save_path: str = "ppo_trading_model",
    tensorboard_log: Optional[str] = None
) -> PPO:
    """
    Train PPO model on stock trading environment
    
    Args:
        tickers: List of stock/ETF tickers
        total_timesteps: Number of training timesteps
        save_path: Path to save trained model
        tensorboard_log: Path to save tensorboard logs
        
    Returns:
        Trained PPO model
    """
    # Create environment
    env = make_vec_env(
        lambda: StockTradingEnv(tickers),
        n_envs=1,  # Start with single environment
        seed=42
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=tensorboard_log or "./tensorboard_logs"
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardCallback()
    )
    model.save(save_path)
    
    return model

if __name__ == "__main__":
    # Example usage
    stocks = ["ASIANPAINT.NS", "HDFCBANK.NS"]
    etfs = ["GOLDBEES.NS"]
    tickers = stocks + etfs
    
    trained_model = train_model(
        tickers,
        total_timesteps=50000,
        save_path="models/ppo_stock_trading",
        tensorboard_log="logs/stock_trading"
    )
