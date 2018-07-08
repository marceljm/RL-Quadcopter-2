from keras import backend as K
from keras import layers, models, optimizers

'''
===============================================================
Critic
===============================================================
The Actor-Critic learning algorithm is used to represent the policy function independently of the value function. The policy function structure is known as the actor, and the value function structure is referred to as the critic. The actor produces an action given the current state of the environment, and the critic produces a TD (Temporal-Difference) error signal given the state and resultant reward. If the critic is estimating the action-value function Q(s,a), it will also need the output of the actor. The output of the critic drives learning in both the actor and the critic. In Deep Reinforcement Learning, neural networks can be used to represent the actor and critic structures.

===============================================================
Reference
===============================================================
[http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html]
'''

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    # Maps [state, action]'s to actions (Q-values)
    def build_model(self):
        # Input layer
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Hidden layers (state)
        L2 = 0.1
        net_states = layers.Dense(units=512, kernel_regularizer=layers.regularizers.l2(L2))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        net_states = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(L2))(net_states)         

        # Hidden layers (action)
        net_actions = layers.Dense(units=256,kernel_regularizer=layers.regularizers.l2(L2))(actions)

        # Hidden layers (both)
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Output layer
        out = layers.Dense(units=1, 
                           name='q_values',
                           kernel_initializer=layers.initializers.RandomUniform(minval=-0.3, maxval=0.3))(net)

        # Model
        self.model = models.Model(inputs=[states, actions], outputs=out)

        # Compile
        self.model.compile(optimizer=optimizers.Adam(lr=L2), loss='mse')

        # Action gradients
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], 
                                               outputs=K.gradients(out, actions))