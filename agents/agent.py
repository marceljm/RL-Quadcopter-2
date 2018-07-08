from agents.actor import Actor
from agents.critic import Critic
from agents.experience import ReplayBuffer
from agents.noise import OUNoise
import numpy as np

'''
===============================================================
Intro
===============================================================
Google DeepMind has devised a solid algorithm for tackling the continuous action space problem. They have produced a policy-gradient actor-critic algorithm called Deep Deterministic Policy Gradients (DDPG) that is off-policy and model-free, and that uses some of the deep learning tricks that were introduced along with Deep Q-Networks (hence the "deep"-ness of DDPG).

===============================================================
DDPG
===============================================================
At its core, DDPG is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn. Policy gradient algorithms utilize a form of policy iteration: they evaluate the policy, and then follow the policy gradient to maximize performance. Since DDPG is off-policy and uses a deterministic target policy, this allows for the use of the Deterministic Policy Gradient theorem (which will be derived shortly). DDPG is an actor-critic algorithm as well; it primarily uses two neural networks, one for the actor and one for the critic. These networks compute action predictions for the current state and generate a temporal-difference (TD) error signal each time step. The input of the actor network is the current state, and the output is a single real value representing an action chosen from a continuous action space (whoa!). The critic’s output is simply the estimated Q-value of the current state and of the action given by the actor. The deterministic policy gradient theorem provides the update rule for the weights of the actor network. The critic network is updated from the gradients obtained from the TD error signal.

===============================================================
Policy-Gradient Methods
===============================================================
Policy-Gradient (PG) algorithms optimize a policy end-to-end by computing noisy estimates of the gradient of the expected reward of the policy and then updating the policy in the gradient direction. Ideally, the algorithm sees lots of training examples of high rewards from good actions and negative rewards from bad actions. Then, it can increase the probability of the good actions.

===============================================================
Model-free
===============================================================
Model-free RL algorithms are those that make no effort to learn the underlying dynamics that govern how an agent interacts with the environment. Its stochastic matrix gives all of the probabilities for arriving at a desired state given the current state and action.

===============================================================
Off-policy
===============================================================
Reinforcement Learning algorithms which are characterized as off-policy generally employ a separate behavior policy that is independent of the policy being improved upon; the behavior policy is used to simulate trajectories. A key benefit of this separation is that the behavior policy can operate by sampling all actions, whereas the estimation policy can be deterministic (e.g., greedy) [1]. Q-learning is an off-policy algorithm, since it updates the Q values without making any assumptions about the actual policy being followed. Rather, the Q-learning algorithm simply states that the Q-value corresponding to state s(t) and action a(t) is updated using the Q-value of the next state s(t+1) and the action a(t+1) that maximizes the Q-value at state s(t+1). On-policy algorithms directly use the policy that is being estimated to sample trajectories during training.

===============================================================
Reference
===============================================================
[http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html]

'''
    
class DDPG():
    
    def __init__(self, task):        
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        # Actor
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())    
        
        # Critic
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())        
        
        # Exploration noise
        self.exploration_mu = 0.1
        self.exploration_sigma = 0.1
        self.exploration_theta = 0.1
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        # Experience
        self.buffer_size = 100000000
        self.batch_size = 64
        self.buffer = ReplayBuffer(self.buffer_size)

        # Parameters
        self.gamma = 0.99
        self.tau = 0.001
        
    def act(self, states):
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())

    def learn(self):
        # Sample
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size, self.action_size, self.state_size)

        # Predict
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        # Train Critic
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train Actor
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Update weights
        self.update_target_weights(self.critic_local.model, self.critic_target.model)
        self.update_target_weights(self.actor_local.model, self.actor_target.model)    
    
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.buffer.add(self.last_state, action, reward, next_state, done)
        self.learn()
        self.last_state = next_state

    def update_target_weights(self, local_model, target_model):
        target_model.set_weights(self.tau * np.array(local_model.get_weights()) + 
                                 (1 - self.tau) * np.array(target_model.get_weights()))