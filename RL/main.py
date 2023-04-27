from env.env import Env

env = Env()

observation = env.start()

while True:
    action = env.sample()
    observation, reward, done = env.step(action)
    print(observation)
    if done:
        break
    
env.stop()
