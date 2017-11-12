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
NUM_FRAMES = 50000000
DISCOUNT = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_FRAME = 1000000
EPSILON_TRAINING_PERIOD = 50000
TARGET_UPDATE_RATE = 10000
ACTION_REPEAT = 4

################################################################################
# Define hyperparameters
################################################################################
NUM_HIDDEN_UNITS = 100
LR = 0.00025 # learning rate
NUM_HIDDEN_UNITS = 10
MOMENTUM = 0.95

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

        self.build_graph()
        sess.run(tf.global_variables_initializer())

    def build_qn(self):
        inputs = tf.placeholder(tf.float32, [None] + list(INPUT_SHAPE))
        net = tflearn.flatten(inputs)
        net = tflearn.fully_connected(net,NUM_HIDDEN_UNITS,activation='relu')
        qvals = tflearn.fully_connected(net, self.env.num_actions)
        return inputs, qvals

    def build_graph(self):
        # Define prediction model
        self.pred_inputs, self.pred_qvals = self.build_qn()
        network_params = tf.trainable_variables()

        # Define target model
        self.target_inputs, self.target_qvals = self.build_qn()
        target_network_params = tf.trainable_variables()[len(network_params):]

        # Define model updater (SGD)
        self.actions = tf.placeholder(tf.float32, [None, self.env.num_actions])
        self.targets = tf.placeholder(tf.float32, [None])
        Qopts = tf.reduce_sum(tf.multiply(self.pred_qvals, self.actions), reduction_indices=1)
        cost = tflearn.mean_square(Qopts, self.targets)
        optimizer = tf.train.RMSPropOptimizer(LR, decay = MOMENTUM) # change different optimizer
        self.updateModel = optimizer.minimize(cost, var_list=network_params)

        # Define updating target model with prediction model
        self.update_target_network_params = \
                [target_network_params[i].assign(network_params[i])
                        for i in range(len(target_network_params))]

        return

    def pred_qvals(self, state, is_target):
        if is_target:
            return self.target_qvals.eval(session=self.sess, feed_dict={self.inputs: [state]})
        else:
            return self.qvals.eval(session=self.sess, feed_dict={self.inputs: [state]})

    def update_model(self, action, target):
        self.sess.run(self.updateModel, feed_dict={self.actions: [action], self.targets: [target]})





################################################################################
# Train agent
################################################################################
def train(sess):
    """Trains the RL Agent"""

    env = EnvWrapper(gym.make(GAME), BUFFER_SIZE)
    dqn_agent = QlWorker(env, sess)
    epsilon = EPSILON_START
    frame = 0
    episode = 0

    while frame < NUM_FRAMES:
        cur_state = env.start_state()
        done = False

        # Set up per-episode counters
        ep_reward = 0
        ep_ave_max_q = 0
        step = 0

        while not done:
            # Forward the deep q network, get Q(s,a) values
            # pred_qvals = dqn_agent.pred_qvals(cur_state, False)
            pred_qvals = sess.run(dqn_agent.pred_qvals, feed_dict={dqn_agent.pred_inputs: [cur_state]})
            # Encode the action in a one hot vector
            action = np.zeros([env.num_actions])
            if np.random.random() < epsilon:
                action_idx = random.randrange(env.num_actions)
                # Qopt = pred_qvals[action_idx]
            else:
                action_idx = np.argmax(pred_qvals)
                # Qopt = np.max(pred_qvals)
            action[action_idx]=1

            target = 0
            for i in range(ACTION_REPEAT):
                # Get new state and reward from environment
                new_state, reward, done, info = env.step(action_idx)
                ep_reward += reward
                ep_ave_max_q += np.max(pred_qvals)
                target += reward

            if not done:
                # target_state = dqn_agent.pred_qvals(new_state, True)
                target_qvals = sess.run(dqn_agent.target_qvals, \
                        feed_dict={dqn_agent.target_inputs: [new_state]})
                target += DISCOUNT * np.max(target_qvals)

            # Update network weights
            sess.run(dqn_agent.updateModel, feed_dict={dqn_agent.actions: [action], dqn_agent.targets: [target], dqn_agent.pred_inputs: [cur_state]})
            # dqn_agent.update_model(action, target)

            # Update target network
            if frame % TARGET_UPDATE_RATE == 0:
                sess.run(dqn_agent.update_target_network_params)

            if epsilon > EPSILON_MIN and frame > EPSILON_TRAINING_PERIOD:
                epsilon -= (EPSILON_START - EPSILON_MIN)/EPSILON_FRAME

            cur_state = new_state
            step+=1

        # Episode finished, print stats
        episode += 1
        frame += step
        print "Episode: {}  Step: {}  Frame: {}  Reward: {}  Qmax: {}  Epsilon: {}".\
            format(episode, step, frame, ep_reward, ep_ave_max_q/float(step), epsilon)

if __name__ == "__main__":
    with tf.Session() as sess:
        train(sess)

