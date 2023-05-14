from tqdm import tqdm
from env.env import Env

env = Env("../MicroRTS/maps/16x16/basesWorkers16x16.xml")

observation = env.start()

for step in tqdm(range(10000)):
    action = env.sample()
    mask = env.getMask()
    observation, reward, done = env.step(action)
    if done:
        env.reset()

env.stop()
