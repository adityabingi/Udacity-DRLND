import tensorflow as tf

W_INIT = tf.keras.initializers.he_normal(seed=0)
B_INIT = tf.constant_initializer(0)

def linear(units):

    return tf.keras.layers.Dense(units,
                kernel_initializer = W_INIT,
                bias_initializer = B_INIT)

class QNetwork(tf.keras.Model):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__(name='Q_Network')

        "*** YOUR CODE HERE ***"
        
        self.fc1 = linear(fc1_units)
        self.fc2 = linear(fc2_units)
        self.fc3 = linear(action_size)

    def call(self, state):
        """Build a network that maps state -> action values."""

        out = self.fc1(state)
        out = tf.nn.relu(out)
        out = self.fc2(out)
        out = tf.nn.relu(out)
        out = self.fc3(out)
        
        return out