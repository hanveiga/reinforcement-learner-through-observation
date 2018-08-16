# in this we run the two competing strategies
# no training exists

import argparse
import atexit
import gym
import numpy as np
import random
import utils
from DQAgent import DQAgent, DQAgent_split
from evaluation import evaluate
from Logger import Logger
from pythonosc import dispatcher, osc_server
import threading


import parameters as p


def print_volume_handler_oc(unused_addr, args):
    print(args)
    DQA.set_interaction_level(args)


def exit_handler():
	global DQA
	DQA.quit()

if __name__ == '__main__':
	IMG_SIZE = (84, 110)
	utils.IMG_SIZE = IMG_SIZE

	# I/O
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--train', action='store_true',
						help='train the agent')
	parser.add_argument('-l', '--load', type=str, default=None,
						help='load the neural network weights from the given path')
	parser.add_argument('-v', '--video', action='store_true',
						help='show video output')
	parser.add_argument('-d', '--debug', action='store_true',
						help='run in debug mode (no output files)')
	parser.add_argument('--eval', action='store_true',
						help='evaluate the agent')
	parser.add_argument('-e', '--environment', type=str,
						help='name of the OpenAI Gym environment to use '
							 '(default: MsPacmanDeterministic-v4)',
						default='MsPacmanDeterministic-v4')
	parser.add_argument('--minibatch-size', type=int, default=32,
						help='number of sample to train the DQN at each update')
	parser.add_argument('--replay-memory-size', type=int, default=1e6,
						help='number of samples stored in the replay memory')
	parser.add_argument('--target-network-update-freq', type=int, default=10e3,
						help='frequency (number of DQN updates) with which the '
							 'target DQN is updated')
	parser.add_argument('--avg-val-computation-freq', type=int, default=50e3,
						help='frequency (number of DQN updates) with which the '
							 'average reward and Q value are computed')
	parser.add_argument('--discount-factor', type=float, default=0.99,
						help='discount factor for the environment')
	parser.add_argument('--update-freq', type=int, default=4,
						help='frequency (number of steps) with which to train the '
							 'DQN')
	parser.add_argument('--learning-rate', type=float, default=0.00025,
						help='learning rate for optimizer')
	parser.add_argument('--epsilon', type=float, default=0.9, 
						help='initial exploration rate for the agent')
	parser.add_argument('--min-epsilon', type=float, default=0.1,
						help='final exploration rate for the agent')
	parser.add_argument('--epsilon-decrease', type=float, default=9e-7,
						help='rate at which to linearly decrease epsilon')
	parser.add_argument('--replay-start-size', type=int, default=50e3,
						help='minimum number of transitions (with fully random '
							 'policy) to store in the replay memory before '
							 'starting training')
	parser.add_argument('--initial-random-actions', type=int, default=1,
						help='number of random actions to be performed by the agent'
							 ' at the beginning of each episode')
	parser.add_argument('--dropout', type=float, default=0.,
						help='dropout rate for the DQN')
	parser.add_argument('--max-episodes', type=int, default=np.inf,
						help='maximum number of episodes that the agent can '
							 'experience before quitting')
	parser.add_argument('--max-episode-length', type=int, default=np.inf,
						help='maximum number of steps in an episode')
	parser.add_argument('--max-frames-number', type=int, default=50e6,
						help='maximum number of frames during the whole algorithm')
	parser.add_argument('--test-freq', type=int, default=250000,
						help='frequency (number of frames) with which to test the '
							 'agent\'s performance')
	parser.add_argument('--validation-frames', type=int, default=135000,
						help='number of frames to test the model like in table 3 of'
							 ' the paper')
	parser.add_argument('--test-states', type=int, default=30,
						help='number of states on which to compute the average Q '
							 'value')
	args = parser.parse_args()

	logger = Logger(debug=args.debug, append=args.environment)
	atexit.register(exit_handler)  # Make sure to always save the model when exiting

	# Variables
	test_scores = []
	test_mean_q = []
	test_states = []

	# Setup
	env = gym.make(args.environment)
	network_input_shape = (4, 110, 84)  # Dimension ordering: 'th' (channels first)

	DQA = DQAgent_split(env.action_space.n,
					network_input_shape,
					replay_memory_size=args.replay_memory_size,
					minibatch_size=args.minibatch_size,
					learning_rate=args.learning_rate,
					discount_factor=args.discount_factor,
					dropout_prob=args.dropout,
					epsilon=args.epsilon,
					epsilon_decrease_rate=args.epsilon_decrease,
					min_epsilon=args.min_epsilon,
					load_path=p.trained_models_path,
					logger=logger)

	#DQA.set_model_names(model1=p.model1,model2=p.model2)

	# Initial logging
	logger.log({
		'Action space': env.action_space.n,
		'Observation space': env.observation_space.shape
	})
	logger.log(vars(args))
	training_csv = 'training_info.csv'
	eval_csv = 'evaluation_info.csv'
	test_csv = 'test_score_mean_q_info.csv'
	logger.to_csv(training_csv, 'length,score')
	logger.to_csv(eval_csv, 'length,score')
	logger.to_csv(test_csv, 'avg_score,avg_Q')

	# Set counters
	episode = 0
	frame_counter = 0

	dispatcher = dispatcher.Dispatcher()
	dispatcher.map("/observing", print_volume_handler_oc)

	server = osc_server.ThreadingOSCUDPServer(
		("127.0.0.1", 8013), dispatcher)
	server_thread = threading.Thread(target=server.serve_forever)
	print("Serving on {}".format(server.server_address))
	server_thread.start()

	logger.log(evaluate(DQA, args, logger))
