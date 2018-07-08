from collections import namedtuple, deque
import numpy as np
import random

'''
===============================================================
Experience Replay
===============================================================
In general, training and evaluating your policy and/or value function with thousands of temporally-correlated simulated trajectories leads to the introduction of enormous amounts of variance in your approximation of the true Q-function (the critic). The TD error signal is excellent at compounding the variance introduced by your bad predictions over time. It is highly suggested to use a replay buffer to store the experiences of the agent during training, and then randomly sample experiences to use for learning in order to break up the temporal correlations within different training episodes. This technique is known as experience replay. DDPG uses this.

===============================================================
Reference
===============================================================
[http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html]
'''

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(106)
        
    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        if len(self.buffer) >= self.buffer_size: 
            self.buffer.popleft()
        self.buffer.append(e)
        
    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size, action_size, state_size):
        experiences = []
        
        if len(self.buffer) < batch_size:
            experiences = random.sample(self.buffer, len(self.buffer))
        else:
            experiences = random.sample(self.buffer, batch_size)
            
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)        
        dones = np.array([e.done for e in experiences if e is not None]).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        return states, actions, rewards, dones, next_states