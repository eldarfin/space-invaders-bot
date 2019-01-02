import gym
import random
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam 

def grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return cv2.resize(img, (86, 110))

def preprocess(img):
    im = grayscale(downsample(img))
    return im[16:102, :]


class DQN():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.state_size, filters=16, kernel_size=(8,8), strides=(4, 4), activation="relu"))
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
        act_values = self.model.predict(state)
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
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    episodes = 3
    batch_size = 32
    env = gym.make('SpaceInvaders-v0')
    state_size = (86, 86)
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size)
    for episode in range(episodes):
        state = env.reset()
        state = preprocess(state)
        score = 0
        for t in range(1000):
            #env.render()
            action = dqn.act(np.array(state))
            next_state, reward, done, info = env.step(env.action_space.sample())
            next_state = preprocess(next_state)
            if reward > 0:
                score += reward
            dqn.remember(state, action, reward, next_state, done)            
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, episodes, score, dqn.epsilon))
                break
            if len(dqn.memory) > batch_size:
                dqn.replay(batch_size)

    '''img = preprocess(state)
    print(img.shape)
    plt.imshow(img, cmap='Greys')
    plt.show()'''