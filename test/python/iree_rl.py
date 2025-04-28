# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb
import iree.compiler.tf
import iree.runtime
import tensorflow as tf
from typing import Any, List, Sequence, Tuple
from iree_rl_impl import env, num_actions, num_hidden_units
from iree_rl_impl import env_step

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
        self,
        num_actions: int,
        num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
    
    @tf.function(input_signature=[
        tf.TensorSpec([128]),  # inputs
        tf.TensorSpec([], tf.int32)  # labels
    ])
    def run_episode(self,
                    initial_state: tf.Tensor,
                    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs a single episode to collect training data without dynamic arrays."""
        
        # Pre-allocate fixed-size tensors with maximum possible size
        # Use padding and masks to handle variable episode lengths
        action_probs = tf.zeros([max_steps], dtype=tf.float32)
        values = tf.zeros([max_steps], dtype=tf.float32)
        rewards = tf.zeros([max_steps], dtype=tf.int32)
        
        initial_state_shape = initial_state.shape
        state = initial_state
        
        # Initialize a counter to keep track of actual steps
        episode_length = 0
        
        # Boolean to track if episode is done
        done = tf.constant(False)
        
        for t in tf.range(max_steps):
            # Only execute if not done
            if not done:
                # Update the counter
                episode_length = t
                
                # Convert state into a batched tensor (batch size = 1)
                _state = tf.expand_dims(state, 0)
                
                # Run the model to get action probabilities and critic value
                action_logits_t, value = self.call(_state)
                
                # Sample next action from the action probability distribution
                action = tf.random.categorical(action_logits_t, 1)[0, 0]
                action_probs_t = tf.nn.softmax(action_logits_t)
                
                # Store critic values
                values = tf.tensor_scatter_nd_update(
                    values, [[t]], [tf.squeeze(value)]
                )
                
                # Store log probability of the action chosen
                action_probs = tf.tensor_scatter_nd_update(
                    action_probs, [[t]], [action_probs_t[0, action]]
                )
                
                # Apply action to the environment to get next state and reward
                # Uncomment and modify this according to your environment
                state, reward, done_step = env_step(action)
                # state.set_shape(initial_state_shape)
                
                # # Store reward (uncomment when environment step is implemented)
                # rewards = tf.tensor_scatter_nd_update(
                #     rewards, [[t]], [reward]
                # )
                
                # Update done condition (uncomment when environment step is implemented)
                # done = tf.cast(done_step, tf.bool)
        
        # Create a mask for valid steps
        mask = tf.range(max_steps) <= episode_length
        mask = tf.cast(mask, tf.float32)
        
        # Apply mask to get only valid entries
        action_probs = action_probs * mask
        values = values * mask
        rewards = tf.cast(rewards * tf.cast(mask, tf.int32), tf.int32)
        
        # For now, just return the last action, probabilities, and value like in your example
        # If you need the full arrays, uncomment the following:
        # return action_probs[:episode_length+1], values[:episode_length+1], rewards[:episode_length+1]
        
        return action_probs, values, rewards


model = ActorCritic(num_actions, num_hidden_units)

exported_names = ["run_episode"]

vm_flatbuffer = iree.compiler.tf.compile_module(
    ActorCritic(num_actions, num_hidden_units),
    saved_model_dir="/home/hao/cs521/iree/workloads",
    exported_names=exported_names,
    import_only=True,
    output_file="/home/hao/cs521/iree/workloads/iree_rl.mlir")