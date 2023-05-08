from env.env import Env

env = Env("../MicroRTS/maps/3x3/bases3x3.xml")

observation = env.start()

reset = 0
i = 0
while i < 1000:
    action = env.sample()
    observation, reward, done = env.step(action)
    print(f"REWARD {i+1} / {reset+1}: " + str(reward))
    reset += 1
    if done or reset == 100:
        reset = 0
        if i+1 < 1000:
            env.reset()
        i+=1
env.stop()
