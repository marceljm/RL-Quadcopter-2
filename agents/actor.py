from keras import backend as K
from keras import layers, models, optimizers

'''
===============================================================
Actor
===============================================================
The Actor-Critic learning algorithm is used to represent the policy function independently of the value function. The policy function structure is known as the actor, and the value function structure is referred to as the critic. The actor produces an action given the current state of the environment, and the critic produces a TD (Temporal-Difference) error signal given the state and resultant reward. If the critic is estimating the action-value function Q(s,a), it will also need the output of the actor. The output of the critic drives learning in both the actor and the critic. In Deep Reinforcement Learning, neural networks can be used to represent the actor and critic structures.

===============================================================
Reference
===============================================================
[http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html]
'''

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()

    # Maps states to actions
    def build_model(self):
        # Input layer
        states = layers.Input(shape=(self.state_size,), name='states')

        # Hidden layers
        L2 = 0.01
        net = layers.Dense(units=512, kernel_regularizer=layers.regularizers.l2(L2))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(L2))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net) 

        # Output layer
        out = layers.Dense(units=self.action_size, 
                           activation='sigmoid', 
                           name='out', 
                           kernel_initializer=layers.initializers.RandomUniform(minval=-0.03, maxval=0.03))(net)

        # Scale [0, 1]
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(out)

        # Model
        self.model = models.Model(inputs=states, outputs=actions)

        # Loss
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Optimizer and training function
        optimizer = optimizers.Adam(lr=L2)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], 
                                   outputs=[],
                                   updates=updates_op)