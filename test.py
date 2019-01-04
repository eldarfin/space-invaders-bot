import gym

env = gym.make('SpaceInvaders-v0')
env.reset()

for t in range(5000):
    env.render()
    next_state, reward, done, info = env.step(6)
    print(reward, info['ale.lives'])
    if done:
        break