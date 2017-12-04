from __future__ import division
import sys
sys.path.append('gym/')
import gym
from gym.wrappers import Monitor
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
import os

################################################################################
# Define parameters
################################################################################

GAME = 'Breakout-v0'
BUFFER_SIZE = 4
INPUT_SHAPE = (BUFFER_SIZE, 84,84)
BATCH_INPUT_SHAPE = (None, BUFFER_SIZE, 84, 84)
NUM_FRAMES_GREEDY = 10 #250000
NUM_FRAMES_IM = 1000 #50000000
DISCOUNT = 0.99
EPSILON_START = 1.0
EPSILON_FRAME = 200000 #1000000
EPSILON_TRAINING_PERIOD = 10000 #50000
PRED_UPDATE_RATE = 32
TARGET_UPDATE_RATE = 2000 #10000
TEACHER_UPDATE_RATE = 5
CHECKPOINT_UPDATE_RATE = 10000
INT_REWARD_DECAY = 10000
NUM_THREADS = 2
NUM_EPISODES_EVAL = 1
MAX_TO_KEEP = 1  # For the saved models
TEST_MODE = False
PLAY_RANDOM_MODE = False
# TEST_PATH = "./trained/qlearning.tflearn.ckpt"
LOG_PATH = "./trained/"
NUM_HIDDEN_ENCODER = 512

################################################################################
# Define hyperparameters
################################################################################

LR = 0.00025 # learning rate
MOMENTUM = 0.95
BETA = 0.1
LEARNING_RATE_ENC = 0.001

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

def sample_final_epsilon():
    final_epsilons = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

################################################################################
# Define Environment wrapper
################################################################################

class EnvWrapper(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size buffer_size from which environment state is constructed.
    """

    def __init__(self, gym_env, buffer_size, video_dir = None):
        self.env = gym_env
        if video_dir is not None:
            self.env = Monitor(env = self.env, directory = videodir, resume = True)
        self.buffer_size = buffer_size
        self.num_actions = gym_env.action_space.n
        # TBD: Workaround for pong and breakout actions
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
        return resize(rgb2gray(observation), (110, 84), mode='constant')[13:110 - 13, :]

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
            net = tf.transpose(self.inputs, [0,2,3,1])
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
            net = tflearn.conv_2d(net, 64, 3, strides=1, activation='relu')
            net = tflearn.fully_connected(net, 512, activation='relu')
            self.qvals = tflearn.fully_connected(net, self.num_actions, activation='linear')
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
# Define Encoder
################################################################################
class Encoder(object):

    def __init__(self, num_actions):#, encoder):
        self.num_actions = num_actions
        self.build_model()

    def build_encoder(self, states):
        encoder = tflearn.conv_2d(states, 32, 8, strides=4, padding="same", activation='relu')
        encoder = tflearn.conv_2d(encoder, 64, 4, strides=2, padding="same", activation='relu')
        encoder = tflearn.conv_2d(encoder, 64, 3, strides=2, padding="same", activation='relu')
        encoder = tflearn.fully_connected(encoder, NUM_HIDDEN_ENCODER, activation='relu')
        return encoder


    def build_model(self):
        self.states = tf.placeholder(tf.float32, [None] + list(INPUT_SHAPE))
        self.new_states = tf.placeholder(tf.float32, [None] + list(INPUT_SHAPE))
        self.actions = tf.placeholder(tf.int32, [None, self.num_actions])
        action_idxs = tf.argmax(self.actions, axis=1)
        # Define encoded state
        with tf.variable_scope('encoder'):
            self.phis = self.build_encoder(self.states)

        # Define encoded new state. Reuse so that encoder network not redefined
        with tf.variable_scope('encoder',reuse=True):
            new_phis = self.build_encoder(self.new_states)

        with tf.variable_scope('encode_updater'):
            action_predictor = tf.concat([self.phis, new_phis], axis=1)
            action_predictor = tflearn.fully_connected(action_predictor, 256, activation='relu')
            logits = tflearn.fully_connected(action_predictor, self.num_actions, activation='linear')
            encoder_cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=action_idxs), name="encoder_cost")
            self.ainvprobs = tf.nn.softmax(logits, dim=1)
            optimizer = tf.train.RMSPropOptimizer(LR, decay=MOMENTUM)
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')
            local_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encode_updater')
            self.updateModel = optimizer.minimize(encoder_cost, var_list=local_vars)

    def encode_state(self, sess, state):
        return sess.run(self.phis, {self.states: [state]})[0,:]

    def encode_states(self, sess, states):
        return sess.run(self.phis, {self.states: states})

    def predict_action(self, sess, state, new_state):
        return sess.run(self.ainvprobs, {self.new_state: [state], self.new_states: [new_state]})[0,:]

    def update(self, sess, states, new_states, actions):
        sess.run(self.updateModel, feed_dict={self.states: states, self.new_states: new_states, self.actions: actions})
        return


################################################################################
# Define state predictor
################################################################################

class StatePredictor(object):

    def __init__(self, num_actions, encoder):
        self.num_actions = num_actions
        self.encoder = encoder
        self.max_error = 0.0
        self.build_model()

    def build_model(self):
        self.phis = tf.placeholder(tf.float32, [None, NUM_HIDDEN_ENCODER])
        self.new_phis = tf.placeholder(tf.float32, [None, NUM_HIDDEN_ENCODER])
        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])

        with tf.variable_scope('state_predictor'):
            state_actions = tf.concat([self.phis, self.actions], axis=1)

            state_predictor = tflearn.fully_connected(state_actions, 256, activation='relu')
            self.state_predictor = tflearn.fully_connected(state_predictor, NUM_HIDDEN_ENCODER, activation='linear')

            self.state_predictor_cost = tf.nn.l2_loss(tf.subtract(self.state_predictor, self.new_phis), name = 'state_predictor_cost')
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'state_predictor')
            optimizer = tf.train.RMSPropOptimizer(LR, decay=MOMENTUM)
            self.updateModel = optimizer.minimize(self.state_predictor_cost, var_list=local_vars)

    def predict_new_phi(self, sess, phi, action):
        return sess.run(self.state_predictor, {self.phis: [phi], self.actions: [action]})[0:,]

    def get_normalized_error(self, sess, state, new_state, action):
        phi = self.encoder.encode_state(sess, state)
        new_phi = self.encoder.encode_state(sess, new_state)
        error = sess.run(self.state_predictor_cost, {self.phis: [phi], self.new_phis: [new_phi], self.actions: [action]})
        self.max_error = max(error, self.max_error)
        return error / self.max_error

    def update(self, sess, states, new_states, actions):
        phis = self.encoder.encode_states(sess, states)
        new_phis = self.encoder.encode_states(sess, new_states)
        sess.run(self.updateModel, {self.phis: phis, self.new_phis: phis, self.actions: actions})
        return

################################################################################
# Define greedy worker
################################################################################
class GreedyWorker(object):

    def __init__(self, sess, saver, thread_id, coord, global_frame, global_state_batch, global_new_state_batch, global_action_batch, env, pred_network, target_network):
        self.sess = sess
        self.saver = saver
        self.thread_id = thread_id
        self.coord = coord
        self.env = env
        self.pred_network = pred_network
        self.target_network = target_network
        self.target_update = copy_vars(sess, 'pred', 'target')
        self.final_epsilon = sample_final_epsilon()
        self.global_frame = global_frame
        self.global_state_batch = global_state_batch
        self.global_action_batch = global_action_batch
        self.global_new_state_batch = global_new_state_batch
        self.increment_global_frame = increment_global_frame_op(sess, global_frame)
        self.global_frame_lock = threading.Lock()
        self.global_batches_lock = threading.Lock()

    def work(self):

        epsilon = EPSILON_START
        episode = 0
        last_target_update = 0
        last_checkpoint_update = 0

        print 'Starting greedy worker {} with final epsilon {}'.format(self.thread_id,self.final_epsilon)
        thread_file = './data/greedy_worker_{}.csv'.format(self.thread_id)
        with open(thread_file, 'w') as f:
            f.write('reward, epsilon, ave_max_q, global_frame\n')

        state_batch_M = []
        action_batch_M = []
        new_state_batch_M = []

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

                # Get new state and reward from environment
                new_state, reward, done, info = self.env.step(action_idx)
                ep_reward += reward
                ep_ave_max_q += np.max(pred_qvals)
                target = np.clip(reward, -1, 1)

                if not done:
                    target_qvals = self.target_network.predict(self.sess, new_state)
                    target += DISCOUNT * np.max(target_qvals)

                # Add to batch
                state_batch.append(cur_state)
                action_batch.append(action)
                target_batch.append(target)

                state_batch_M.append(cur_state)
                new_state_batch_M.append(new_state)
                action_batch_M.append(action)

                # Update pred network
                if step % PRED_UPDATE_RATE == 0 or done:
                    # In case we are done without adding anything new to batch
                    self.pred_network.update(self.sess, state_batch, action_batch, target_batch)
                    state_batch, action_batch, target_batch = [], [] ,[]

                # Update epsilon
                if epsilon > self.final_epsilon and global_frame > EPSILON_TRAINING_PERIOD:
                    epsilon -= (EPSILON_START - self.final_epsilon)/EPSILON_FRAME

                if self.thread_id == 0:
                    # Update target network
                    if global_frame - last_target_update >= TARGET_UPDATE_RATE:
                        print "Updating target network..."
                        print "global frame {}, last_target_update {}, target {}".format(global_frame, last_target_update, TARGET_UPDATE_RATE)
                        self.target_update()
                        last_target_update = global_frame

                    if global_frame - last_checkpoint_update >= CHECKPOINT_UPDATE_RATE:
                        saver.save(sess, os.path.join(LOG_PATH, '{}-sess.ckpt'.format(GAME)), global_step=global_frame)
                        print "Session saved to {}".format(LOG_PATH)
                        last_checkpoint_update = global_frame

                # If max step is reached, request all threads to stop
                if global_frame >= NUM_FRAMES_GREEDY:
                    self.coord.request_stop()

                cur_state = new_state
                step+=1

            # Episode finished, print stats
            episode += 1
            print "Greedy Worker: {}  Episode: {}  Step: {}  Frame: {}  Reward: {}  Qmax: {}  Epsilon: {}".\
                format(self.thread_id, episode, step, global_frame, ep_reward, ep_ave_max_q/float(step), epsilon)
            with open(thread_file,'a') as f:
                write_string = "{}, {}, {}, {}\n".format(ep_reward, epsilon, ep_ave_max_q/float(step), global_frame)
                f.write(write_string)


        with self.global_batches_lock:
            # print global_frame, self.thread_id
            # print self.global_frame.eval(session=self.sess), self.thread_id
            # print "Thread {} has local batch size {}".format(self.thread_id, len(state_batch_M))
            self.global_state_batch.extend(state_batch_M)
            self.global_action_batch.extend(action_batch_M)
            self.global_new_state_batch.extend(new_state_batch_M)

################################################################################
# Define IM worker
################################################################################
class IMWorker(object):

    def __init__(self, sess, saver, thread_id, coord, global_frame, env, pred_network, target_network, encoder_network, teacher_network):
        self.sess = sess
        self.saver = saver
        self.thread_id = thread_id
        self.coord = coord
        self.env = env
        self.pred_network = pred_network
        self.target_network = target_network
        self.encoder_network = encoder_network
        self.teacher_network = teacher_network
        self.target_update = copy_vars(sess, 'pred', 'target')
        self.global_frame = global_frame
        self.increment_global_frame = increment_global_frame_op(sess, global_frame)
        self.global_frame_lock = threading.Lock()

    def work(self):
        """Trains the RL Agent using intrinsic motivation"""

        episode = 0
        last_target_update = 0
        last_checkpoint_update = 0

        print 'Starting IM worker {}'.format(self.thread_id)
        thread_file = './data/IM_worker_{}.csv'.format(self.thread_id)
        with open(thread_file, 'w') as f:
            f.write('total reward, intrinsic reward, extrinsic reward, ave_max_q, global_frame\n')

        while not self.coord.should_stop():

            cur_state = self.env.start_state()
            done = False

            # Set up per-episode counters
            ep_total_reward = 0
            ep_ext_reward = 0
            ep_int_reward = 0
            ep_ave_max_q = 0
            step = 0

            state_batch_P = []
            action_batch_P = []
            target_batch_P = []

            state_batch_M = []
            action_batch_M = []
            new_state_batch_M = []

            while not done:
                with self.global_frame_lock:
                    global_frame = self.increment_global_frame()

                # Forward the deep q network, get Q(s,a) values
                pred_qvals = self.pred_network.predict(self.sess, cur_state)

                # Encode the action in a one hot vector
                action = np.zeros([self.env.num_actions])
                action_idx = np.argmax(pred_qvals)
                action[action_idx]=1

                # Get new state and extrinsic reward from environment
                new_state, ext_reward, done, info = self.env.step(action_idx)
                ep_ext_reward += ext_reward
                ep_ave_max_q += np.max(pred_qvals)
                ext_reward = np.clip(ext_reward, -1, 1)

                # Get intrinsic reward
                err = self.teacher_network.get_normalized_error(sess, cur_state, new_state, action)
                int_reward = err / (global_frame / INT_REWARD_DECAY)
                ep_int_reward += int_reward

                total_reward = ext_reward + BETA * int_reward
                ep_total_reward += total_reward
                target = total_reward

                if not done:
                    target_qvals = self.target_network.predict(self.sess, new_state)
                    target += DISCOUNT * np.max(target_qvals)

                # Add to predictive model batch
                state_batch_P.append(cur_state)
                action_batch_P.append(action)
                target_batch_P.append(target)

                # Add to teacher model batch
                state_batch_M.append(cur_state)
                new_state_batch_M.append(new_state)
                action_batch_M.append(action)

                # Update pred network
                if step % PRED_UPDATE_RATE == 0 or done:
                    # In case we are done without adding anything new to batch
                    self.pred_network.update(self.sess, state_batch_P, action_batch_P, target_batch_P)
                    state_batch_P, action_batch_P, target_batch_P = [], [] ,[]

                # Update teacher network
                if step % TEACHER_UPDATE_RATE == 0 or done:
                    self.teacher_network.update(self.sess, state_batch_M, new_state_batch_M, action_batch_M)
                    self.encoder_network.update(self.sess, state_batch_M, new_state_batch_M, action_batch_M)
                    state_batch_M, action_batch_M, new_state_batch_M = [], [] ,[]

                if self.thread_id == 0:
                    # Update target network
                    if global_frame - last_target_update >= TARGET_UPDATE_RATE:
                        print "Updating target network..."
                        print "still running, global_frame {}, last target {}, rate {}".format(global_frame, last_target_update, TARGET_UPDATE_RATE)
                        self.target_update()
                        last_target_update = global_frame

                    if global_frame - last_checkpoint_update >= CHECKPOINT_UPDATE_RATE:
                        saver.save(sess, os.path.join(LOG_PATH, 'IM-{}-sess.ckpt'.format(GAME)), global_step=global_frame)
                        print "Session saved to {}".format(LOG_PATH)
                        last_checkpoint_update = global_frame

                # If max step is reached, request all threads to stop
                if global_frame >= NUM_FRAMES_IM:
                    self.coord.request_stop()

                cur_state = new_state
                step+=1

            # Episode finished, print stats
            episode += 1
            print "IM Worker: {}  Episode: {}  Step: {}  Frame: {}  Total Reward: {}  Int Reward: {}  Ext Reward: {}  Qmax: {}".\
                format(self.thread_id, episode, step, global_frame, ep_total_reward, ep_int_reward, ep_ext_reward, ep_ave_max_q/float(step))
            with open(thread_file,'a') as f:
                write_string = "{}, {}, {}, {}, {}\n".format(ep_total_reward, ep_int_reward, ep_ext_reward, ep_ave_max_q/float(step), global_frame)
                f.write(write_string)



################################################################################
# Train agent
################################################################################

def train(sess, saver):
    '''Launches the training by creating parallel threads, launching agents in each thread and starting each agent learning'''

    # Create global step counter
    global_frame_eps = tf.Variable(name='global_frame_eps', initial_value=0, trainable=False, dtype=tf.int32)
    global_frame_im = tf.Variable(name = 'global_frame_im', initial_value = 0, trainable = False, dtype = tf.int32)

    global_state_batch = []
    global_action_batch = []
    global_new_state_batch = []

    # Get num actions
    env = gym.make(GAME)
    num_actions = env.action_space.n
    env.close()

    # Create shared networks
    pred_network = DQN(num_actions, 'pred')
    target_network = DQN(num_actions, 'target')
    encoder_network = Encoder(num_actions)
    teacher_network = StatePredictor(num_actions, encoder_network)

    # Init variables
    sess.run(tf.global_variables_initializer())
    first_update = copy_vars(sess, 'pred','target')
    first_update()

    # Create thread coordinator
    coord = tf.train.Coordinator()

    # Create environment for each thread
    env = [EnvWrapper(gym.make(GAME), BUFFER_SIZE) for i in range(NUM_THREADS)]

    # Create workers for each thread
    # greedy_workers = []
    greedy_threads = []

    # im_workers = []
    im_threads = []

    # Initialize with an epsilon-greedy strategy to collect states and actions for initial training of encoder
    for thread_id in range(NUM_THREADS):
        worker = GreedyWorker(sess, saver, thread_id, coord, global_frame_eps, global_state_batch, global_new_state_batch, global_action_batch, env[thread_id], pred_network, target_network)
        # workers.append(worker)
        worker_work = lambda: worker.work()
        t = threading.Thread(target=worker_work)
        greedy_threads.append(t)
        #t.daemon=True # WHAT
        t.start()
        time.sleep(0.01) # needed?

    coord.join(greedy_threads)

    global_state_batch = np.array(global_state_batch)
    global_new_state_batch = np.array(global_new_state_batch)
    global_action_batch = np.array(global_action_batch)

    # Train the auto encoder with data collected with epsilon-greedy
    encoder_network.update(sess, global_state_batch, global_new_state_batch, global_action_batch)
    teacher_network.update(sess, global_state_batch, global_new_state_batch, global_action_batch)
    global_state_batch, global_new_state_batch, global_action_batch = None, None, None

    # Create thread coordinator
    im_coord = tf.train.Coordinator()

    # Train with intrinsic motivation approach
    for thread_id in range(NUM_THREADS):
        worker = IMWorker(sess, saver, thread_id, im_coord, global_frame_im, env[thread_id], pred_network, target_network, encoder_network, teacher_network)
        # workers.append(worker)
        worker_work = lambda: worker.work()
        t = threading.Thread(target=worker_work)
        im_threads.append(t)
        #t.daemon=True # WHAT
        t.start()
        time.sleep(0.01) # needed?

    print "finished here"
    im_coord.join(im_threads)
    saver.save(sess, os.path.join(LOG_PATH, 'IM-{}-sess.ckpt'.format(GAME)), global_step=global_frame_im)
    print "Final session saved to {}".format(LOG_PATH)

    # Evaluate straight away
    evaluate_model(sess, pred_network)

################################################################################
# Evaluate trained model
################################################################################

def evaluate(sess, saver):

    trained_eval = './data/trained_eval.csv'
    open(trained_eval, 'w').close()

    monitor_env = gym.make(GAME)
    env = EnvWrapper(monitor_env, BUFFER_SIZE)
    num_actions = env.num_actions
    pred_network = DQN(num_actions, 'pred')
    target_network = DQN(num_actions, 'target')

    # Init variables
    sess.run(tf.global_variables_initializer())
    # first_update = copy_vars(sess, 'pred','target')
    # first_update()

    print [v.name for v in tf.trainable_variables()]

    new_saver = tf.train.import_meta_graph(TEST_PATH+'.meta')
    new_saver.restore(sess, TEST_PATH)
    # var = tf.get_default_graph().get_tensor_by_name("pred/Conv2D/b:0")
    # print "var: ", var.eval()
    # saver.restore(sess, TEST_PATH)
    print [v.name for v in tf.trainable_variables()]
    tf_vars = [v for v in tf.trainable_variables()]
    mid = len(tf_vars)//2
    for i in range(mid):
        v1 = tf_vars[i]
        v2 = tf_vars[mid+i]
        assign_op = v1.assign(v2)
        sess.run(assign_op)
        # v2.assign(v1)

    print("Restored model weights")
    # print "var: ", var.eval()

    # monitor_env.monitor.start("qlearning/eval")


    for episode in range(NUM_EPISODES_EVAL):
        monitor_env.reset()
        cur_state = env.start_state()
        ep_reward = 0
        done = False
        while not done:
            monitor_env.render()

            pred_qvals = pred_network.predict(sess, cur_state)
            action_idx = np.argmax(pred_qvals)
            new_state, reward, done, info = env.step(action_idx)
            ep_reward += reward
            cur_state = new_state

        print(ep_reward)
        with open(test_eval,'a') as f:
            write_string = "{}\n".format(ep_reward)
            f.write(write_string)

    monitor_env.monitor.close()

def evaluate_model(sess, pred_network):

    monitor_env = gym.make(GAME)
    env = EnvWrapper(monitor_env, BUFFER_SIZE)
    trained_eval = './data/IM-{}-trained_eval.csv'.format(GAME)
    with open(trained_eval, 'w') as f:
        f.write('reward\n')

    for episode in range(NUM_EPISODES_EVAL):
        monitor_env.reset()
        cur_state = env.start_state()
        ep_reward = 0
        done = False
        while not done:
            # monitor_env.render()

            pred_qvals = pred_network.predict(sess, cur_state)
            action_idx = np.argmax(pred_qvals)
            new_state, reward, done, info = env.step(action_idx)
            ep_reward += reward
            cur_state = new_state

        print(ep_reward)
        with open(trained_eval,'a') as f:
            write_string = "{}\n".format(ep_reward)
            f.write(write_string)


################################################################################
# Play at random
################################################################################

def evaluate_random():
    random_eval = './data/random_eval.csv'
    open(random_eval, 'w').close()

    monitor_env = gym.make(GAME)
    env = EnvWrapper(monitor_env, BUFFER_SIZE)
    for episode in range(NUM_EPISODES_EVAL):
        monitor_env.reset()
        cur_state = env.start_state()
        ep_reward = 0
        done = False
        while not done:
            monitor_env.render()
            action_idx = random.randrange(env.num_actions)
            new_state, reward, done, info = env.step(action_idx)
            ep_reward += reward
            cur_state = new_state

        print(ep_reward)
        with open(random_eval,'a') as f:
            write_string = "{}\n".format(ep_reward)
            f.write(write_string)

################################################################################
# Main script
################################################################################
if __name__ == "__main__":
    start = time.time()
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep = MAX_TO_KEEP)
        if TEST_MODE:
            evaluate(sess, saver)
        elif PLAY_RANDOM_MODE:
            evaluate_random()
        else:
            train(sess, saver)

    end = time.time()
    print "Time taken: {}".format(end-start)

