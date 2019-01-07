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
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 1000
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
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
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

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    episodes = 500
    batch_size = 32
    env = gym.make('Breakout-v0')
    state = preprocess(env.reset())
    state_size = state.shape
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size)
    scores = 0
    avgs = []
    for episode in range(episodes):
        state = env.reset()
        state = preprocess(state)
        score = 0
        lives = 5
        prev_lives = 5
        if episode % 5 == 0 and episode != 0:
            print('Interim save, episode: ', episode)
            dqn.save("./saves/breakout-dqn-{}.h5".format(episode))
            print('Average score of last 5 games: ', scores / 5)
            avgs.append(scores/5)
            scores = 0
        for t in range(100000):
            #env.render()
            action = dqn.act(state)
            next_state, reward, done, info = env.step(action)
            lives = info['ale.lives']
            if lives < prev_lives:
                reward = -10
            next_state = preprocess(next_state)
            score += reward
            dqn.remember(np.array([state]), action, reward, np.array([next_state]), done)            
            state = next_state
            prev_lives = lives
            last_action = action
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode+1, episodes, score, dqn.epsilon))
                break
            if len(dqn.memory) > batch_size:
                dqn.replay(batch_size)
        scores += score
    
    dqn.save('./saves/breakout-dqn.h5')
    print(avgs)
    rng = np.arange(5, len(avgs) * 5 + 1, 5)
    
    plt.plot(rng, avgs)
    plt.show()

    '''state = env.reset()
    state = preprocess(state)
    score = 0
    for t in range(5000):
        env.render()
        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = preprocess(next_state)
        score += reward
        state = next_state
        if done:
            break'''
    

    '''img = preprocess(state)
    print(img.shape)
    plt.imshow(img, cmap='Greys')
    plt.show()'''