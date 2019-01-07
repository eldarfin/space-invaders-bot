import gym
import random
import numpy as np 
import cv2
import matplotlib.pyplot as plt

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def downsample(img):
    return cv2.resize(img, (84, 105))

def preprocess(img):
    im = grayscale(downsample(img))
    im = im[14:98, :]
    #im = im.reshape((84, 84, 1))
    return im

env = gym.make('Breakout-v0')
state = env.reset()
state = preprocess(state)
print(env.env.get_action_meanings())
for t in range(10000):
    env.render()
    next_state, reward, done, info = env.step(env.action_space.sample())
    #print(reward, info['ale.lives'])
    state = preprocess(next_state)
    if done: 
        break
plt.imshow(state, cmap='Greys')
plt.show()




