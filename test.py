import gym
import matplotlib.pyplot as plt
import cv2
import numpy as np

env = gym.make('SpaceInvaders-v0')
env.reset()

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def downsample(img):
    return cv2.resize(img, (80, 105))

def preprocess(img):
    im = grayscale(downsample(img))
    im = im.reshape((105, 80))
    return im

for t in range(5000):
    #env.render()
    next_state, reward, done, info = env.step(5)
    print(next_state.shape)
    if done:
        break

plt.imshow(preprocess(next_state), cmap='Greys')
plt.show()