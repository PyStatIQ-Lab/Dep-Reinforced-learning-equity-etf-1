from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from trading_env import StockTradingEnv

def train_model(tickers, save_path="ppo_trading_model"):
    # Create environment
    env = make_vec_env(
        lambda: StockTradingEnv(tickers),
        n_envs=4  # Parallel environments
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Train model
    model.learn(total_timesteps=100000)
    model.save(save_path)
    
    return model

if __name__ == "__main__":
    stocks = ["ADANIENT.NS", "ADANIPORTS.NS", ...]  # Your stock list
    etfs = ["MAFANG.NS", "FMCGIETF.NS", ...]       # Your ETF list
    tickers = stocks + etfs
    train_model(tickers)
