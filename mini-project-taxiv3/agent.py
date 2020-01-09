import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha = 0.1, gamma=0.999):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1.0
        self.episode = 1
        self.num_episodes = 20000

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.eps = 1.0/self.episode
        if state in self.Q:
            best_action = np.argmax(self.Q[state])
            if np.random.uniform() > self.eps:
                return best_action
        return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.episode += 1
        q_t = self.Q[state][action] 
        best_q_tp1 = np.max(self.Q[next_state])
        #exp_q = ((1-self.eps)*best_q_tp1) + ((self.eps/self.nA)*(sum(self.Q[next_state])))
        target = reward + (self.gamma*(1-done)*best_q_tp1)
        self.Q[state][action] = q_t + (self.alpha *(target - q_t))