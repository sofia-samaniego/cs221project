from __future__ import division
import sys
sys.path.append('../../gym')

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

################################################################################
# Define model environment
################################################################################
game = sys.argv[1]
env = gym.make(game)
INPUT_SHAPE = (84,84)
WINDOW_LENGTH = 4
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE



################################################################################
# Define hyperparameters
################################################################################
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network


################################################################################
# Define network
################################################################################

class QN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.model = self.create_model()
        self.target_model = self.create_model()

        def create_model(self):
            model = Sequential()
            state_shape = self.env.observation_space.shape
            model.add(Flatten())
            model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
            model.add(Dense(self.env.action_space.n))

def preprocess(s):
    '''Preprocesses the atari screen'''
    pass
