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
import threading
import time

# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.optimizers import Adam

################################################################################
# Define parameters
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
PRED_UPDATE_RATE = 32
TARGET_UPDATE_RATE = 10000
ACTION_REPEAT = 4
NUM_THREADS = 2

################################################################################
# Define hyperparameters
################################################################################
NUM_HIDDEN_UNITS = 100
LR = 0.00025 # learning rate
NUM_HIDDEN_UNITS = 10
MOMENTUM = 0.95

################################################################################
# Define helper functions
################################################################################
def copy_vars(sess, from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = [to_var.assign(from_var) for from_var, to_var in zip(from_vars, to_vars)]

    def run_op():
        sess.run(op_holder)

    return run_op

def increment_global_frame_op(sess, global_frame):
    op = tf.assign(global_frame, global_frame+1)

    def run_op():
        _, step = sess.run([op, global_frame])
        return step

    return run_op


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
class DQN(object):

    def __init__(self, num_actions, scope):
        self.num_actions = num_actions
        self.scope = scope

        self.build_dqn()
        if scope == 'pred':
            self.build_updater()

    def build_dqn(self):
        '''Defines the value function model'''

        with tf.variable_scope(self.scope):
            self.inputs = tf.placeholder(tf.float32, [None] + list(INPUT_SHAPE))
            net = tflearn.flatten(self.inputs)
            net = tflearn.fully_connected(net,NUM_HIDDEN_UNITS,activation='relu')
            self.qvals = tflearn.fully_connected(net, self.num_actions)
        return

    def build_updater(self):
        '''Defines the model cost, optimizer and updater'''

        # Define model updater (SGD)
        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.targets = tf.placeholder(tf.float32, [None])
        Qopts = tf.reduce_sum(tf.multiply(self.qvals, self.actions), reduction_indices=1)
        cost = tflearn.mean_square(Qopts, self.targets)
        optimizer = tf.train.RMSPropOptimizer(LR, decay = MOMENTUM) # change different optimizer
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.updateModel = optimizer.minimize(cost, var_list=local_vars)

        return
    
    def predict(self, sess, state):
        return self.qvals.eval(session=sess, feed_dict={self.inputs: [state]})

    def update(self, sess, states, actions, targets):
        sess.run(self.updateModel, feed_dict={self.inputs: states, self.actions: actions, self.targets: targets})


################################################################################
# Define worker
################################################################################
class Worker(object):
    
    def __init__(self, sess, thread_id, coord, global_frame, env, pred_network, target_network):
        self.sess = sess
        # print sess._closed
        self.thread_id = thread_id
        self.coord = coord
        self.env = env
        self.pred_network = pred_network
        self.target_network = target_network
        self.target_update = copy_vars(sess, 'pred', 'target')
        self.global_frame = global_frame
        self.increment_global_frame = increment_global_frame_op(sess, global_frame)
        self.global_frame_lock = threading.Lock()
    
    
    def work(self):
        """Trains the RL Agent"""

        epsilon = EPSILON_START
        episode = 0

        print 'Starting worker {} with final epsilon {}'.format(self.thread_id,EPSILON_MIN)

        while not self.coord.should_stop():

            cur_state = self.env.start_state()
            done = False

            # Set up per-episode counters
            ep_reward = 0
            ep_ave_max_q = 0
            step = 0
            
            state_batch = []
            action_batch = []
            target_batch = []

            while not done:
                with self.global_frame_lock:
                    global_frame = self.increment_global_frame()

                # Forward the deep q network, get Q(s,a) values
                pred_qvals = self.pred_network.predict(self.sess, cur_state)

                # Encode the action in a one hot vector
                action = np.zeros([self.env.num_actions])
                if np.random.random() < epsilon:
                    action_idx = random.randrange(self.env.num_actions)
                else:
                    action_idx = np.argmax(pred_qvals)
                action[action_idx]=1

                target = 0
                for i in range(ACTION_REPEAT):
                    # Get new state and reward from environment
                    new_state, reward, done, info = self.env.step(action_idx)
                    ep_reward += reward
                    ep_ave_max_q += np.max(pred_qvals)
                    target += reward

                if not done:
                    target_qvals = self.target_network.predict(self.sess, new_state)
                    target += DISCOUNT * np.max(target_qvals)

                # Add to batch
                state_batch.append(cur_state)
                action_batch.append(action)
                target_batch.append(target)

                # Update pred network
                if step % PRED_UPDATE_RATE == 0 or done:
                    # In case we are done without adding anything new to batch
                    # if state_batch:
                    self.pred_network.update(self.sess, state_batch, action_batch, target_batch)
                    state_batch, action_batch, target_batch = [], [] ,[]

                # Update target network
                if global_frame % TARGET_UPDATE_RATE == 0:
                    print "Updating target network..."
                    self.target_update()

                # Update epsilon
                if epsilon > EPSILON_MIN and global_frame > EPSILON_TRAINING_PERIOD:
                    epsilon -= (EPSILON_START - EPSILON_MIN)/EPSILON_FRAME

                # If max step is reached, request all threads to stop
                if global_frame >= NUM_FRAMES:
                    self.coord.request_stop()

                cur_state = new_state
                step+=1

            # Episode finished, print stats
            episode += 1
            print "Worker: {}  Episode: {}  Step: {}  Frame: {}  Reward: {}  Qmax: {}  Epsilon: {}".\
                format(self.thread_id, episode, step, global_frame, ep_reward, ep_ave_max_q/float(step), epsilon)


################################################################################
# Train agent
################################################################################

def train(sess):
    '''Launches the training by creating parallel threads, launching agents in each thread and starting each agent learning'''

    # Create global step counter
    global_frame = tf.Variable(name='global_frame', initial_value=0, trainable=False, dtype=tf.int32)

    # Get num actions
    env = gym.make(GAME)
    num_actions = env.action_space.n
    env.close()

    # Create shared networks
    pred_network = DQN(num_actions, 'pred')
    target_network = DQN(num_actions, 'target')

    # Init variables
    sess.run(tf.global_variables_initializer())
    first_update = copy_vars(sess, 'pred','target')
    first_update()

    # Create thread coordinator
    coord = tf.train.Coordinator()

    # Create environment for each thread
    env = [EnvWrapper(gym.make(GAME), BUFFER_SIZE) for i in range(NUM_THREADS)]

    # Create workers for each thread
    workers = []
    threads = []
    for thread_id in range(NUM_THREADS):
        worker = Worker(sess, thread_id, coord, global_frame, env[thread_id], pred_network, target_network)
        workers.append(worker)
        worker_work = lambda: worker.work()
        t = threading.Thread(target=worker_work)
        threads.append(t)
        # t.daemon=True # WHAT
        t.start()
        time.sleep(0.01) # needed?

    coord.join(threads)


if __name__ == "__main__":
    with tf.Session() as sess:
        train(sess)

