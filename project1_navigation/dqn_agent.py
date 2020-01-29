import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import tensorflow as tf

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, update_type='dqn', seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.update_type = update_type

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    @tf.function
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        q_values = self.qnetwork_local(state)
    
        # Epsilon-greedy action selection
        batch_size = (q_values.shape)[0]
        random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype= tf.float32)
        random_action = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.action_size, dtype=tf.int64)
        best_action = tf.argmax(q_values, axis=1)
        action = tf.where(tf.less(random, eps), random_action, best_action)
        return action

    @tf.function
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"

        with tf.GradientTape() as tape:

            target_q = self.compute_target(next_states)
            target_q = tf.multiply(tf.cast((1-float(dones)),tf.float32), target_q)
            target_qv = tf.stop_gradient(tf.cast(rewards,tf.float32) + GAMMA * target_q)
            q_v = tf.reduce_sum(tf.multiply(self.qnetwork_local(states),tf.one_hot(actions,self.action_size)),1)
            td_error = q_v - target_qv
            loss = self.huber_loss(td_error)

        grads = tape.gradient(loss, self.qnetwork_local.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.qnetwork_local.trainable_variables))

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def compute_target(self, next_states):

        if self.update_type=='ddqn':
            max_acts = tf.argmax(self.qnetwork_local(next_states), 1)
            target_q = tf.reduce_sum(tf.multiply(self.qnetwork_target(next_states),
                                                tf.one_hot(max_acts, self.action_size)), 1)
        else:
            target_q = tf.reduce_max(self.qnetwork_target(next_states),1)
        return target_q

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (tf.keras model): weights will be copied from
            target_model (tf.keras model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.trainable_variables,\
            local_model.trainable_variables):

            target_param.assign(tau*local_param+ (1.0-tau)*target_param)

    def huber_loss(self, x, delta=1.0):
        loss = tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x)-(0.5*delta))
        )
        return loss


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = []
        self.maxsize = buffer_size
        self.next_idx = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        if self.next_idx >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self.next_idx] = e

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def _encode_sample(self, idxes):

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in idxes:
            e = self.memory[i]
            states.append(np.array(e.state, copy=False, dtype='float32'))
            actions.append(np.array(e.action, copy=False))
            rewards.append(e.reward)
            next_states.append(np.array(e.next_state, copy=False, dtype='float32'))
            dones.append(e.done)

        return np.array(states), np.array(actions), np.array(rewards, dtype='float32'), np.array(next_states), np.array(dones)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        idxes = [random.randint(0, len(self.memory) - 1) for _ in range(self.batch_size)]
        return self._encode_sample(idxes)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
