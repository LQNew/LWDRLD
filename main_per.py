import random
import torch
import argparse
import os
import time
import numpy as np
from collections import deque
# ------------------------------
from DQN_Zoo import DQN_per
from DQN_Zoo import Double_DQN_per
# ------------------------------
from DQN_Zoo import Dueling_DQN_per
from DQN_Zoo import Dueling_Double_DQN_per
# -------------------------------
from environments.wrappers import wrap, wrap_cover, SubprocVecEnv
from utils import replay_buffer
from utils.schedule import LinearSchedule

from spinupUtils.logx import EpochLogger
from spinupUtils.run_utils import setup_logger_kwargs

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="DQN", type=str)         # Policy name
	parser.add_argument("--env", default="Pong", type=str)           # OpenAI gym environment name
	parser.add_argument("--num_envs", default=10, type=int)          # Num of vector-envs paralleled
	parser.add_argument("--seed", default=0, type=int)               # Set seeds for Gym, PyTorch and Numpy
	parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps for initial random policy
	parser.add_argument("--eval_freq", default=1e3, type=int)        # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=2e7, type=int)    # Max timesteps to run environment
	parser.add_argument("--discount", default=0.99, type=float)      # Discount factor
	parser.add_argument("--policy_freq", default=1e3, type=int)      # Frequency of delayed policy updates
	parser.add_argument("--update_freq", default=4, type=int)        # Frequency of updating the Q function
	parser.add_argument("--buffer_size", default=1e6, type=int)      # Size of buffer
	parser.add_argument("--batch_size", default=64, type=int)        # Batch size for Q network training
	parser.add_argument("--gradient_clip", default=10.0, type=float) # Clipping gradient
	parser.add_argument("--alpha_per", default=0.6, type=float)      # Alpha parameter for prioritized replay buffer
	parser.add_argument("--beta0_per", default=0.4, type=float)      # Initial value of beta for prioritized replay buffer
	parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                  # Model-loading file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--exp_name", type=str)       				 # Name for algorithms
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print(f"---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print(f"---------------------------------------")

	# Make envs
	env_name = f"{args.env}NoFrameskip-v4"
	env = SubprocVecEnv([wrap_cover(env_name, args.seed) for i in range(args.num_envs)])
	# Set seeds
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	action_dim = env.action_space.n

	kwargs = {
		"action_dim": action_dim,
		"discount": args.discount,
		"gradient_clip": args.gradient_clip,
	}

	# Initialize policy
	# ----------------------------------------------
	if args.policy == "DQN_per":
		kwargs["policy_freq"] = int(args.policy_freq) // int(args.num_envs)
		kwargs["learning_rate"] = 1e-4
		policy = DQN_per.DQN_PER(**kwargs)
		eps_schedule = LinearSchedule(1.0, 0.01, 1e6)  # annealing epsilon
		beta_schedule = LinearSchedule(args.beta0_per, 1.0, args.max_timesteps - args.start_timesteps)  # annealing beta
		args.batch_size = 64
	elif args.policy == "Double_DQN_per":
		kwargs["policy_freq"] = int(args.policy_freq) // int(args.num_envs)
		kwargs["learning_rate"] = 1e-4
		policy = Double_DQN_per.DoubleDQN_PER(**kwargs)
		eps_schedule = LinearSchedule(1.0, 0.01, 1e6)  # annealing epsilon
		beta_schedule = LinearSchedule(args.beta0_per, 1.0, args.max_timesteps - args.start_timesteps)  # annealing beta
		args.batch_size = 64
	# ----------------------------------------------
	elif args.policy == "Dueling_DQN_per":
		kwargs["policy_freq"] = int(args.policy_freq) // int(args.num_envs)
		kwargs["learning_rate"] = 1e-4
		policy = Dueling_DQN_per.DuelingDQN_PER(**kwargs)
		eps_schedule = LinearSchedule(1.0, 0.01, 1e6)  # annealing epsilon
		beta_schedule = LinearSchedule(args.beta0_per, 1.0, args.max_timesteps - args.start_timesteps)  # annealing beta
	elif args.policy == "Dueling_Double_DQN_per":
		kwargs["policy_freq"] = int(args.policy_freq) // int(args.num_envs)
		kwargs["learning_rate"] = 1e-4
		policy = Dueling_Double_DQN_per.DuelingDoubleDQN_PER(**kwargs)
		eps_schedule = LinearSchedule(1.0, 0.01, 1e6)  # annealing epsilon
		beta_schedule = LinearSchedule(args.beta0_per, 1.0, args.max_timesteps - args.start_timesteps)  # annealing beta
	else:
		raise ValueError(f"Invalid Policy: {args.policy}!")
	
	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		if not os.path.exists(f"./models/{policy_file}"):
			assert f"The loading model path of `../models/{policy_file}` does not exist! "
		policy.load(f"./models/{policy_file}")
	
	# Setup loggers
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=False)
	logger = EpochLogger(**logger_kwargs)

	_replay_buffer = replay_buffer.PrioritizedReplayBuffer(int(args.buffer_size), args.alpha_per)
	
	print("Collecting experience...")
	epinfobuf = deque(maxlen=100)  # episode step for accumulate reward 
	start_time = time.time()  # check learning time

	states = np.array(env.reset())  # env reset, output array of num of `#num_envs` states

	step = 0
	for t in range(1, int(args.max_timesteps) // int(args.num_envs) + 1):
		actions = policy.select_action(states, eps_schedule.value)
		next_states, rewards, dones, infos = env.step(actions)  # take actions and get next states
		next_states = np.array(next_states)
		# log arrange
		for info in infos:
			maybeepinfo = info.get("episode")
			if maybeepinfo: 
				epinfobuf.append(maybeepinfo)

		# Store the transition
		for i in range(args.num_envs):
			_replay_buffer.add(states[i], actions[i], next_states[i], rewards[i], dones[i])
			step += 1
		
		eps_schedule.update(step)  # Annealing the epsilon, for exploration strategy
		states = next_states

		# If memory fill 50K and mod 4 == 0 (for speed issue), update the policy
		if (step >= args.start_timesteps) and (step % args.update_freq == 0):
			policy.train(_replay_buffer, beta=beta_schedule.value, batch_size=args.batch_size)
			beta_schedule.update(step - args.start_timesteps)

		# print log and save model
		if t % args.eval_freq == 0:
			if args.save_model: 
				policy.save(f"./models/{file_name}")
			# check time interval
			time_interval = round(time.time() - start_time, 2)
			mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)  # calculate mean return
			print(f"Used Step: {step} | Epsilon: {round(eps_schedule.value, 3)} "
				  f"| Mean ep 100 return: {mean_100_ep_return} "
				  f"| Used Time: {time_interval}")
			# store the logger
			logger.log_tabular("MeanEpReward", mean_100_ep_return)
			logger.log_tabular("TotalEnvInteracts", step)
			logger.log_tabular("Time", time_interval)
			logger.dump_tabular()

	print("The training is done!")
	