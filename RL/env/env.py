class Env:

    # Create an environnement for the RL
    def __init__(self):
        self.observation = []
        self.reward = 0

    # Reset the environnement and return the observation features
    def reset(self):
        self.observation = []
        return self.observation
    
    # Return a random available action
    def sample(self):
        action = []
        return action
    
    # Play the action and return the new observation features, the reward, if the environnement is done
    def step(self, action):
        done = False
        return self.observation, self.reward, done