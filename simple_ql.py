from __future__ import division
import sys
sys.path.append('gym/')
import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import tflearn

# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.optimizers import Adam

################################################################################
# Define game parameters
################################################################################
GAME = 'Breakout-v0'
BUFFER_SIZE = 4
INPUT_SHAPE = (BUFFER_SIZE, 84,84)
NUM_EPISODES = 1
DISCOUNT = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

################################################################################
# Define hyperparameters
################################################################################
NUM_HIDDEN_UNITS=100
LR = 0.001 # learning rate
NUM_HIDDEN_UNITS = 10

################################################################################
# Define Environment wrapper
################################################################################
class EnvWrapper(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size buffer_size from which environment state is constructed.
    """

    def __init__(self, gym_env, buffer_size):
        self.env = gym_env
        self.buffer_size = buffer_size
        self.num_actions = gym_env.action_space.n

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(gym_env.action_space.n)
        # Screen buffer of size buffer_size to be able to build
        # state arrays of size [1, buffer_size, 84, 84]
        self.state_buffer = deque()

    def start_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack([x_t for i in range(self.buffer_size)], axis=0)

        for i in range(self.buffer_size-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        0) Atari frames: 210 x 160
        1) Get image grayscale
        2) Rescale image 110 x 84
        3) Crop center 84 x 84 (you can crop top/bottom according to the game)
        """
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]

    def step(self, action_index):
        """
        Executes an action in the gym environment.
        Builds current state (concatenation of buffer_size-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        #x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1, r_t, terminal, info = self.env.step(action_index)
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.buffer_size, 84, 84))
        s_t1[:self.buffer_size-1, :] = previous_frames
        s_t1[self.buffer_size-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info


################################################################################
# Define network
################################################################################
class QlWorker(object):

    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.inputs, self.qvals = self.create_model()
        self.target_inputs, self.target_qvals = self.create_model()
        self.epsilon = EPSILON_START
        sess.run(tf.initialize_all_variables())

    def create_model(self):
        inputs = tf.placeholder(tf.float32, [None] + list(INPUT_SHAPE))
        net = tflearn.flatten(inputs)
        net = tflearn.fully_connected(net,NUM_HIDDEN_UNITS,activation='relu')
        qvals = tflearn.fully_connected(net, self.env.num_actions)
        return inputs, qvals

    def pred_qvals(self, state):
        return self.qvals.eval(session=self.sess, feed_dict={self.inputs: [state]})

    def choose_action(self, state):
        self.epsilon *= EPSILON_DECAY
        self.epsilon = min(EPSILON_MIN, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.rangrange(self.env.num_actions)
        else:
            return np.argmax(pred_qvals(state))





################################################################################
# Train agent
################################################################################
def train(sess):
    """Trains the RL Agent"""

    env = EnvWrapper(gym.make(GAME), BUFFER_SIZE)
    dqn_agent = QlWorker(env, sess)

    for episode in range(NUM_EPISODES):
        cur_state = env.start_state()
        done = False

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while not done:
            # Forward the deep q network, get Q(s,a) values
            if np.random.random() < self.epsilon:
                action_idx = random.rangrange(self.env.num_actions)
            else:
                pred_qvals = dqn_agent.pred_qvals(cur_state)
                action_idx = np.argmax(pred_qvals)
                Qopt = np.max(pred_qvals)

            # Get new state and reward from environment
            new_state, reward, done, info = env.step(action_idx)

            target = reward
            if not done:
                target_state = dqn_agent.pred_qvals(new_state)
                target += DISCOUNT * np.max(target_state)



            done =True


 


if __name__ == "__main__":


    with tf.Session() as sess:
        train(sess)












# for i in range(num_episodes):
    # s = env.reset() # Restart game
    # rAll = 0 # counts the reward
    # done = False
    # step = 0
    # while step < 99:
        # step += 1
        # # Choose action greedily, with e chance of random action
        # a, allQ = predict_using_model(state) # get predicted action, and weights
        # if np.random.rand(1) < e:
            # a[0] = env.action_space.sample()

        # #Get new state and reward from environment
        # s1,r,done,_ = env.step(a[0])

        # #Obtain the Q' values by feeding the new state through our network
        # Q1 = predict_using_model(s1)

        # #Obtain maxQ' and set our target value for chosen action.
        # maxQ1 = np.max(Q1)
        # targetQ = allQ
        # targetQ[0,a[0]] = r + y*maxQ1

        # #Train our network using target and predicted Q values
        # # Update weights :)

        # rAll += r
        # s = s1
        # if d == True:
            # #Reduce chance of random action as we train the model.
            # # Epislon greedy reducing the epsilon :)
            # e = 1./((i/50) + 10)
            # break
    # stepList.append(step)
    # rList.append(rAll)





















