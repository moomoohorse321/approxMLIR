import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple


# Create the environment
env = gym.make("CartPole-v1")

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

num_actions = env.action_space.n  # 2
num_hidden_units = 128


def env_step(action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns state, reward and done flag given an action."""
    
    @tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
    def _env_step_numpy(action):
        state, reward, done, truncated, info = env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))
    
    # Call the numpy function
    state, reward, done = _env_step_numpy(action)
    
    # Explicitly set the shapes to match what's expected
    state.set_shape([num_hidden_units])
    reward.set_shape([])
    done.set_shape([])
    
    return state, reward, done
