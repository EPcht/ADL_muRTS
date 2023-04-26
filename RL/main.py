from env.env import Env
from env.bot import AI

ai = AI()
ai.start()

# env = Env()

# observation = env.reset()
# for _ in range(200):
#     action = env.sample()
#     observation, reward, done = env.step(action)
#     if done:
#         break
