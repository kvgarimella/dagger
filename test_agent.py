from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
import torch.nn as nn
import argparse
from model import Model
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch Device:", device)

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0
    state = env.reset()

    while True:
        # preprocessing 
        gray = np.dot(state[...,:3], [0.2125, 0.7154, 0.0721])[:84,...]
        pred = agent(torch.from_numpy(gray[np.newaxis, np.newaxis,...]).type(torch.FloatTensor))
        a    = pred.detach().numpy().flatten()

        # take action, receive new state & reward
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", required=True, type=str, help="Path to PyTorch model")
    args = parser.parse_args()

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = Model()
    print("Loading model {}:".format(args.path))
    agent.load(args.path)
    # agent.load("models/agent.ckpt")
    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
