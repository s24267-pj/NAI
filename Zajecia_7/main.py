"""
BeamRider AI agent using PPO (Proximal Policy Optimization).
The script initializes the environment, loads a pre-trained model if available,
or trains a new one from scratch. It also saves periodic checkpoints and
allows resuming training from the latest checkpoint.

Features:
- Loads the latest available checkpoint if present.
- Saves model checkpoints every 10,000 training steps.
- Evaluates and tests the trained model.
"""

import gymnasium as gym
from stable_baselines3 import PPO
import os
import glob
import ale_py
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

# Model file name
MODEL_PATH = "beamrider_ppo_model.zip"
CHECKPOINT_DIR = "./checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_latest_checkpoint(checkpoint_dir, prefix="ppo_beamrider"):
    """Finds the latest checkpoint file in the given directory."""
    checkpoints = glob.glob(f"{checkpoint_dir}/{prefix}_*.zip")
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)  # Latest file
    return None


# 1. Initialize the BeamRider environment
env = gym.make("ALE/BeamRider-v5", render_mode="human")

# 2. Check if a saved model or checkpoint exists
checkpoint_path = get_latest_checkpoint(CHECKPOINT_DIR)
if checkpoint_path:
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env)
elif os.path.exists(MODEL_PATH):
    print("Loaded saved model.")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("No saved model found. Creating a new one...")
    model = PPO("CnnPolicy", env, verbose=1)

# 3. Configure checkpoints
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix="ppo_beamrider")

# 4. Train the model
train_steps = 100_000
print("Starting agent training...")
model.learn(total_timesteps=train_steps, callback=checkpoint_callback)

# 5. Save the trained model
model.save("beamrider_ppo_model")
print("Model saved as 'beamrider_ppo_model'.")

# 6. Evaluate the model
print("Evaluating the trained agent...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# 7. Test the trained agent
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    if done:
        obs, _ = env.reset()

env.close()
