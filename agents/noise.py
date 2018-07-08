import numpy as np

'''
===============================================================
Ornstein-Uhlenbeck process
===============================================================
it is simply a stochastic process which has mean-reverting properties.
- theta (θ): means the how "fast" the variable reverts towards to the mean;
- mu (μ) represents the equilibrium or mean value;
- sigma (σ) is the degree of volatility of the process.

===============================================================
Random noise
===============================================================
Random noise is crucial for getting DNNs to work well:
- allows neural nets to produce multiple outputs given the same instance of input.
- limits the amount of information flowing through the network, forcing the network to learn meaningful representations of data. 
- provides "exploration energy" for finding better optimization solutions during gradient descent.

===============================================================
Reference
===============================================================
[https://blog.evjang.com/2016/07/randomness-deep-learning.html]
[https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html]
'''

class OUNoise:

    def __init__(self, size, mu, theta, sigma):
        self.size = size
        self.mu = mu        
        self.theta = theta
        self.sigma = sigma
        self.reset()
        np.random.seed(106)

    def reset(self):
        self.state = np.ones(self.size) * self.mu        

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state = self.state + dx        
        return self.state