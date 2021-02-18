import gym
from gym import error, spaces, utils
from gym.utils import seeding
# to use generate_signal function
import numpy as np
import statistics
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



class SignalEnv(gym.Env):
	"""
		Environment is a 1D signal
		Agent is a gaussian distribution that moves on the signal.


		Actions:
			7 types of actions:
			- move 2 steps to the right
			- move 1 step to the right
			- stay
 			- move 1 step to the left
			- move 2 steps to the left
			- change standard dev to 1
			- change standard dev to 2

			# TODO: prova a vedere se si riesce a fare due tipi di azioni:
			        - move (left or right) the mean of the gaussian
					- change the std-deviation
					--> USA: spaces.Tuple([spaces.Discrete(n1), spaces.Discrete(n2)])


		An episode ends when:
			- The agent reaches the last sample of the signal

		Reward:
			- every sample of the signal returns a reward, which is the output
			  of the function "_compute_reward":
			  RMS(interval around mean+3dev) - RMS(interval around mean-3dev)
			  #TODO: RMS deve essere ancora implementato, intanto uso i valori
			  dei segnali

	"""
	LEFT_2 = 0
	LEFT_1 = 1
	STAY = 2
	RIGHT_1 = 3
	RIGHT_2 = 4

	DELTA1 = 5
	DELTA2 = 6

	def __init__(self):

		# self.x and self.signal are <class 'numpy.ndarray'>
		[self.x, self.signal] = generate_signal(seed=0)
		self.signal_len = len(self.signal)


		self.observation_space = spaces.Discrete(self.signal_len)
		print('States space:', self.observation_space)
		n_actions_mean = 5
		n_actions_d = 2
		#self.action_space = spaces.Tuple([spaces.Discrete(n_actions_mean), spaces.Discrete(n_actions_d)])
		self.action_space = spaces.Discrete(n_actions_mean + n_actions_d)
		print('Actions space:', self.action_space)
		print('type:', type(self.action_space))

		# a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range
		# so no need to set "self.reward_range = (...)"


		# Initialize the agent position at the beginning of the signal
		self.agent_pos = 0
		self.d = self.DELTA1





	def step(self, action):
		"""
		Run one timestep of the environment's dynamics. When end of episode is
		reached, you are responsible for calling `reset()` to reset this
		environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

		Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further
							step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for
			 		     debugging, and sometimes learning)
        """
		# Compute action--------------------------------------------------------
		if action == self.RIGHT_2:
			self.agent_pos = self.agent_pos + 2
			self.d = self.d
		elif action == self.RIGHT_1:
			self.agent_pos = self.agent_pos + 1
			self.d = self.d
		elif action == self.STAY:
			self.agent_pos = self.agent_pos
			self.d = self.d
		elif action == self.LEFT_1:
			self.agent_pos = self.agent_pos - 1
			self.d = self.d
		elif action == self.LEFT_2:
			self.agent_pos = self.agent_pos - 2
			self.d = self.d
		elif action == self.DELTA1:
			self.agent_pos = self.agent_pos
			self.d = 1
		elif action == self.DELTA2:
			self.agent_pos = self.agent_pos
			self.d = 2
		else:
			raise ValueError("Received invalid action={} which is not part of the action space".format(action))

		# Account for the boundaries of the grid
		self.agent_pos = np.clip(self.agent_pos, 0, self.signal_len - 1)
		observation = (self.agent_pos,self.d)

		# Reward given by the function "compute_reward"-------------------------
		'''
		idx_max = self.agent_pos + self.dev
		idx_min = self.agent_pos - self.dev

		# accont for the boundaries of the signal
		if idx_min < 0: idx_min = 0
		if idx_max > self.signal_len-1: idx_max = self.signal_len - 1

		max = self.signal[idx_max]
		min = self.signal[idx_min]
		reward = max - min
		'''
		idx_max = self.agent_pos + self.d
		idx_min = self.agent_pos - self.d

		# select intervals around borders
		idx_max_interval = [idx_max-2, idx_max-1,idx_max,idx_max+1,idx_max+2]
		idx_min_interval = [idx_min-2, idx_min-1,idx_min,idx_min+1,idx_min+2]

		# account for boundaries
		#(remove from the intervals all the values < 0 and >= max length of the signal)
		idx_max_interval = [item for item in idx_max_interval if (item >= 0) & (item < self.signal_len)]
		idx_min_interval = [item for item in idx_min_interval if (item >= 0) & (item < self.signal_len)]

		# select the value of the signal in the two intervals
		max_signal = self.signal[idx_max_interval]
		min_signal = self.signal[idx_min_interval]

		# calculate RMS of the signal in the two intervals
		rms_max = np.sqrt(np.mean(max_signal**2))
		rms_min = np.sqrt(np.mean(min_signal**2))

		reward = rms_max - rms_min

		# When we make the episode end?-----------------------------------------
		done = bool(self.agent_pos == (self.signal_len - 1))


		# Additional info, not necessary----------------------------------------
		info = {}

		return observation, reward, done, info

	def reset(self):
		"""
		Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
		# Initialize the agent position at the beginning of the signal
		self.agent_pos = 0
		self.d = 1
		return (self.agent_pos, self.d)  # reward, done, info can't be included


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]



def generate_signal(seed=0):
	x = np.linspace(0, 100, 100) #start value of the sequence, end value of the seqence, number of samples to generate. Default is 50.
	mask1 = (x>=10) & (x<=17)
	mask2 = (x>=28) & (x<=35)
	mask3 = (x>=64) & (x<=77)
	mask4 = (x>=88) & (x<=93)

	y = np.where(mask1, 20, 0) + np.where(mask2, 10, 0) + np.where(mask3, 30, 0) + np.where(mask4, 5, 0)
	np.random.seed(seed)
	noise = np.random.normal(0,0.5,100)

	return [x,y + noise]





#check if the env works
if __name__ == "__main__":
	[x, signal] = generate_signal()

	env = SignalEnv()

	obs = env.reset()
	print("initial state:", obs)

	# initialize list to plot of the rewards
	idx = []
	rewards = []

	n_steps = 1000000
	for step in range(n_steps):
		print("Step {}".format(step+1))
		action = np.random.randint(7)
		obs,reward,done,_ = env.step(action)
		print('obs = ', obs, 'reward = ', reward, 'done = ', done)

		#append infos to plot the rewards
		idx.append(obs[0])
		rewards.append(reward)

		if done:
			print('end signal')
			break




# PLOT SIGNAL and REWARDS-------------------------------------------------------


	idx = np.array(idx)
	rewards = np.array(rewards)
	mean_rewards = []
	std_rewards = []
	for state in range(100):
		indexes = np.where(idx==state)
		if len(indexes[0]) != 0: #check if the list is not empty
			mean_reward = np.mean(rewards[indexes])
			std_reward = np.std(rewards[indexes])
		else:
			mean_reward = np.nan
			std_reward = np.nan
		#print(mean_reward)
		#print(std_reward)
		mean_rewards.append(mean_reward)
		std_rewards.append(std_reward)



	# plot the signal and the reward values
	fig, ax = plt.subplots(figsize=(100, 16))
	plt.plot(x,signal,label='signal')
	plt.errorbar(x,mean_rewards,std_rewards, linestyle='None', marker='s', label="rewards(mean $\pm$ devstd)")
	ax.set_xlim(0, 100)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
	# Turn grid on for both major and minor ticks and style minor slightly
	# differently.
	ax.grid(which='major', color='#CCCCCC', linestyle='--')
	ax.grid(which='minor', color='#CCCCCC', linestyle=':')
	ax.legend()
	plt.show()
