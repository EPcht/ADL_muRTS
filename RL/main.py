from env.env import Env

env = Env()

observation = env.start()

while True:
    action = env.sample()
    observation, reward, done = env.step(action)
    if done:
        break
    
env.stop()
