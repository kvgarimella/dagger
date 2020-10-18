from __future__ import print_function
import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
from model import Model
import train_agent
from utils import *

import torch
NUM_ITS = 20
beta_i  = 0.9
T       = 4000
s = """  ____    _                         
 |  _ \  / \   __ _  __ _  ___ _ __ 
 | | | |/ _ \ / _` |/ _` |/ _ \ '__|
 | |_| / ___ \ (_| | (_| |  __/ |   
 |____/_/   \_\__, |\__, |\___|_|   
              |___/ |___/           

"""

def wait():
    _ = input()

def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.2

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0.0
    if k == key.UP:    a[1] = 0.0
    if k == key.DOWN:  a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_dagger.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


if __name__ == "__main__":

    print(s)
    print("Welcome to the DAgger Algorithm for CarRacing-v0!")
    print("Drive the car using the arrow keys.")
    print("After every {} timesteps, the game will freeze and train the agent using the collected data.".format(T))
    print("After each training loop, the previously trained network will have more control of the action taken.")
    print("Trained models will be saved in the dagger_models directory.")
    print("Press any key to begin driving!")
    wait()


    if not os.path.exists("dagger_models"):
        os.mkdir("dagger_models")
 

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    env = gym.make('CarRacing-v0').unwrapped

    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    
    episode_rewards = []
    steps = 0
    agent = Model()
    agent.save("dagger_models/model_0.pth")
    model_number = 0
    old_model_number = 0


    for iteration in range(NUM_ITS):
        agent = Model()
        agent.load("dagger_models/model_{}.pth".format(model_number))
        curr_beta = beta_i ** model_number 

        if model_number != old_model_number:
            print("Using model : {}".format(model_number))
            print("Beta value: {}".format(curr_beta))
            old_model_number = model_number

        episode_reward = 0
        state = env.reset() 
        # pi: input to the environment
        # a : expert input
        pi = np.array([0.0, 0.0, 0.0]).astype('float32')
        a = np.zeros_like(pi)

        while True:

            next_state, r, done, info = env.step(pi)

            # preprocess image and find prediction ->  policy(state)
            gray = np.dot(next_state[...,:3], [0.2125, 0.7154, 0.0721])[:84,...]
            prediction = agent(torch.from_numpy(gray[np.newaxis,np.newaxis,...]).type(torch.FloatTensor))

            # calculate linear combination of expert and network policy
            pi = curr_beta * a + (1 - curr_beta) * prediction.detach().numpy().flatten()
            episode_reward += r

            samples["state"].append(state)            # state has shape (96, 96, 3)
            samples["action"].append(np.array(a))     # action has shape (1, 3), STORE THE EXPERT ACTION
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)
            
            state = next_state
            steps += 1

            # train after T steps
            if steps % T == 0:
                print('... saving data')
                store_data(samples, "./data")
                save_results(episode_rewards, "./results")
                X_train, y_train, X_valid, y_valid = train_agent.read_data("./data", "data_dagger.pkl.gzip")
                X_train, y_train, X_valid, y_valid = train_agent.preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)
                train_agent.train_model(X_train, y_train, X_valid, y_valid, "dagger_models/model_{}.pth".format(model_number+1), num_epochs=10)
                model_number += 1
                print("Training complete. Press return to continue to the next iteration")
                wait()
                break

            env.render()
            if done: 
                
                break
        
        episode_rewards.append(episode_reward)

    env.close()

      
