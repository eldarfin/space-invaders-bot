import gym

env = gym.make('BreakoutDeterministic-v4')
state = env.reset()
count = 0
done = False
for t in range(10000) or not done:
    env.render()
    next_state, reward, done, info = env.step(env.action_space.sample())
    count += 1
    if done:
        break
print(count)