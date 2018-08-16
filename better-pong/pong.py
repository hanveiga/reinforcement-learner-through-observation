#!/usr/bin/env python
"""
Train a Pong AI using policy gradient-based reinforcement learning.

Based on Andrej Karpathy's "Deep Reinforcement Learning: Pong from Pixels"
http://karpathy.github.io/2016/05/31/rl/
and the associated code
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

from __future__ import print_function

import argparse
import pickle
import numpy as np
import gym

import tensorflow as tf

from pythonosc import dispatcher, osc_server
import threading

from policy_network import Network

lamb = 0

def set_volume_handler_oc(unused_addr, args):
    global lamb
    lamb = args
    print(lamb)

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer_size', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--batch_size_episodes', type=int, default=1)
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=10)
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--discount_factor', type=int, default=0.99)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

# Action values to send to gym environment to move paddle up/down
UP_ACTION = 2
DOWN_ACTION = 3
# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}


# From Andrej's code
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards

env = gym.make('Pong-v0')

with tf.Graph().as_default() as net1_graph:
    network = Network(
    args.hidden_layer_size, args.learning_rate, net1_graph, checkpoints_dir='checkpoints')
    
    if args.load_checkpoint:
        network.load_checkpoint()

batch_state_action_reward_tuples = []
smoothed_reward = None
episode_n = 1

train = 0
evaluation = 1

if evaluation:
    #network2 = Network(
    #    args.hidden_layer_size, args.learning_rate, checkpoints_dir='checkpoints_good')
    #if args.load_checkpoint:
    #    network2.load_checkpoint()

    with tf.Graph().as_default() as net2_graph:
        network2 = Network(
        args.hidden_layer_size, args.learning_rate, net2_graph, checkpoints_dir='checkpoints_bad')
        
        if args.load_checkpoint:
            network2.load_checkpoint()

print(network)
print(network2)

if train:
    while True:
        print("Starting episode %d" % episode_n)

        episode_done = False
        episode_reward_sum = 0

        round_n = 1

        last_observation = env.reset()
        last_observation = prepro(last_observation)
        action = env.action_space.sample()
        observation, _, _, _ = env.step(action)
        observation = prepro(observation)
        n_steps = 1

        while not episode_done:
            if args.render:
                env.render()

            observation_delta = observation - last_observation
            last_observation = observation
            up_probability = network.forward_pass(observation_delta)[0]
            if np.random.uniform() < up_probability:
                action = UP_ACTION
            else:
                action = DOWN_ACTION

            observation, reward, episode_done, info = env.step(action)
            observation = prepro(observation)
            episode_reward_sum += reward
            n_steps += 1

            tup = (observation_delta, action_dict[action], reward)
            batch_state_action_reward_tuples.append(tup)

            if reward == -1:
                print("Round %d: %d time steps; lost..." % (round_n, n_steps))
            elif reward == +1:
                print("Round %d: %d time steps; won!" % (round_n, n_steps))
            if reward != 0:
                round_n += 1
                n_steps = 0

        print("Episode %d finished after %d rounds" % (episode_n, round_n))

        # exponentially smoothed version of reward
        if smoothed_reward is None:
            smoothed_reward = episode_reward_sum
        else:
            smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
        print("Reward total was %.3f; discounted moving average of reward is %.3f" \
            % (episode_reward_sum, smoothed_reward))

        if episode_n % args.batch_size_episodes == 0:
            states, actions, rewards = zip(*batch_state_action_reward_tuples)
            rewards = discount_rewards(rewards, args.discount_factor)
            rewards -= np.mean(rewards)
            rewards /= np.std(rewards)
            batch_state_action_reward_tuples = list(zip(states, actions, rewards))
            network.train(batch_state_action_reward_tuples)
            batch_state_action_reward_tuples = []

        if episode_n % args.checkpoint_every_n_episodes == 0:
            network.save_checkpoint()

        episode_n += 1

if evaluation:
    # evaluate (stop training)
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/observing", set_volume_handler_oc)

    server = osc_server.ThreadingOSCUDPServer(
        ("127.0.0.1", 8013), dispatcher)
    server_thread = threading.Thread(target=server.serve_forever)
    print("Serving on {}".format(server.server_address))
    server_thread.start()

    while True:
        print("Starting episode %d" % episode_n)

        episode_done = False
        episode_reward_sum = 0

        round_n = 1

        last_observation = env.reset()
        last_observation = prepro(last_observation)
        action = env.action_space.sample()
        observation, _, _, _ = env.step(action)
        observation = prepro(observation)
        n_steps = 1

        while not episode_done:
            if args.render:
                env.render()

            observation_delta = observation - last_observation
            last_observation = observation

            up_probability1 = network.forward_pass(observation_delta)[0]
            print(observation_delta)
            up_probability2 = network2.forward_pass(observation_delta)[0]
            print(observation_delta)
            print(up_probability1,up_probability2)
            #lamb = 0.5

            up_probability = lamb*up_probability1 + (1-lamb)*up_probability2


            if np.random.uniform() < up_probability:
                action = UP_ACTION
            else:
                action = DOWN_ACTION

            observation, reward, episode_done, info = env.step(action)
            observation = prepro(observation)
            episode_reward_sum += reward
            n_steps += 1

            tup = (observation_delta, action_dict[action], reward)
            batch_state_action_reward_tuples.append(tup)

            if reward == -1:
                print("Round %d: %d time steps; lost..." % (round_n, n_steps))
            elif reward == +1:
                print("Round %d: %d time steps; won!" % (round_n, n_steps))
            if reward != 0:
                round_n += 1
                n_steps = 0

        print("Episode %d finished after %d rounds" % (episode_n, round_n))

        episode_n += 1
