import numpy as np
from utils.util import *
# The Evaluator class simulates running the policy in the environment for multiple episodes and records how well it performs. 
# It does this by tracking the rewards earned and the distance between the generated images and the target images.
class Evaluator(object):

    def __init__(self, args, writer):    
        self.validate_episodes = args.validate_episodes
        self.max_step = args.max_step
        self.env_batch = args.env_batch # number of environments to run in parallel
        self.writer = writer
        self.log = 0

    def __call__(self, env, policy, debug=False):        
        observation = None
        for episode in range(self.validate_episodes):
            # reset at the start of episode
            observation = env.reset(test=True, episode=episode)
            episode_steps = 0
            episode_reward = 0.     
            assert observation is not None            
            # start episode
            episode_reward = np.zeros(self.env_batch)
            while (episode_steps < self.max_step or not self.max_step):
                action = policy(observation)    #selects an action using policy based on current state/observation
                observation, reward, done, (step_num) = env.step(action) # takes the selected action in the environment and receives the new state, reward, and done flag
                episode_reward += reward   #reward for each episode
                episode_steps += 1
                env.save_image(self.log, episode_steps)
            dist = env.get_dist()   # distance (measure of similarity) between the generated and target image
            self.log += 1
        return episode_reward, dist
