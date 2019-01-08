import gym
from gym import wrappers
import random
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from collections import deque
import datetime

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam 

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def downsample(img):
    return cv2.resize(img, (84, 105))

def preprocess(img):
    im = grayscale(downsample(img))
    im = im[14:98, :]
    im = im.reshape((84, 84, 1))
    return im


class DQN():
    def __init__(self, state_size, action_size, test=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 
        if test:
            self.epsilon = 0.1
        else:
            self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self.build_model()
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.state_size, data_format="channels_last", filters=16, kernel_size=(8,8), strides=(4, 4), activation="relu"))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    env = wrappers.Monitor(env, './saves/video/{}/'.format(str(datetime.datetime.now().time())))
    state = preprocess(env.reset())
    state_size = state.shape
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size, test=True) 
    dqn.load('./saves/breakout-dqn-235.h5')
    score = 0
    for t in range(10000):
        env.render()
        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        score += reward
        next_state = preprocess(next_state)
        state = next_state
        if done:
            print('Score: ', score)
            break