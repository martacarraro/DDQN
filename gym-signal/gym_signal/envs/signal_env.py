import gym
from gym import error, spaces, utils
from gym.utils import seeding
# to use generate_signal function
import numpy as np
import matplotlib.pyplot as plt



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

	STDEV1 = 5
	STDEV2 = 6

	def __init__(self):
		#self.seed(0) # a cosa serve? non ottengo gli stessi risultati

		[self.x, self.signal] = generate_signal(seed=0) # self.x and self.signal are <class 'numpy.ndarray'>
		self.signal_len = len(self.signal)


		self.observation_space = spaces.Discrete(self.signal_len)
		print('States space:', self.observation_space)
		n_actions_mean = 5
		n_actions_devstd = 2
		#self.action_space = spaces.Tuple([spaces.Discrete(n_actions_mean), spaces.Discrete(n_actions_devstd)])
		self.action_space = spaces.Discrete(n_actions_mean + n_actions_devstd)
		print('Actions space:', self.action_space)
		print('type:', type(self.action_space))

		# a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range
		# so no need to set "self.reward_range = (...)"


		# Initialize the agent position at the beginning of the signal
		self.agent_pos = 0
		self.dev = self.STDEV1





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
			self.dev = self.dev
		elif action == self.RIGHT_1:
			self.agent_pos = self.agent_pos + 1
			self.dev = self.dev
		elif action == self.STAY:
			self.agent_pos = self.agent_pos
			self.dev = self.dev
		elif action == self.LEFT_1:
			self.agent_pos = self.agent_pos - 1
			self.dev = self.dev
		elif action == self.LEFT_2:
			self.agent_pos = self.agent_pos - 2
			self.dev = self.dev
		elif action == self.STDEV1:
			self.agent_pos = self.agent_pos
			self.dev = 1
		elif action == self.STDEV2:
			self.agent_pos = self.agent_pos
			self.dev = 2
		else:
			raise ValueError("Received invalid action={} which is not part of the action space".format(action))

		# Account for the boundaries of the grid
		self.agent_pos = np.clip(self.agent_pos, 0, self.signal_len - 1)
		observation = (self.agent_pos,self.dev)

		# Reward given by the function "compute_reward"-------------------------
		idx_max = self.agent_pos + 3*self.dev
		idx_min = self.agent_pos - 3*self.dev

		# accont for the boundaries of the signal
		if idx_min < 0: idx_min = 0
		if idx_max > self.signal_len-1: idx_max = self.signal_len - 1

		max = self.signal[idx_max]
		min = self.signal[idx_min]
		reward = max - min

		# Are we at the end of the signal?--------------------------------------
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
		self.dev = 1
		return (self.agent_pos, self.dev)  # reward, done, info can't be included

	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def compute_reward(self):
		"""
		Considering the agent a truncated gaussian in [-3dev +3dev]. The reward
		is the difference between the signal value at mean+3dev - the signal
		value at mean-3dev. Since I'm interested in finiding the activation phase
		of the muscle contraction, the gaussian will not have the contraction on
		the left and will have it on the right --> max - min = positive reward
		"""
		idx_max = self.agent_pos + 3*self.dev
		idx_min = self.agent_pos - 3*self.dev

		# accont for the boundaries of the signal
		if idx_min < 0: idx_min = 0
		if idx_max > self.signal_len-1: idx_max = self.signal_len - 1

		max = self.signal[idx_max]
		min = self.signal[idx_min]
		return max - min








def generate_signal(seed=0):
	x = np.linspace(0, 50, 50) #start value of the sequence, end value of the seqence, number of samples to generate. Default is 50.
	mask1 = (x>=10) & (x<=17)
	mask2 = (x>=28) & (x<=35)

	y = np.where(mask1, 20, 0) + np.where(mask2, 10, 0)
	np.random.seed(seed)
	noise = np.random.normal(0,0.5,50)

	return [x,y + noise]





#check if the env works
if __name__ == "__main__":
	[x, signal] = generate_signal()

	env = SignalEnv()

	obs = env.reset()
	print(obs)

	n_steps = 10000
	for step in range(n_steps):
		print("Step {}".format(step+1))
		action = np.random.randint(7)
		obs,reward,done,_ = env.step(action)
		print('obs = ', obs, 'reward = ', reward, 'done = ', done)
		if done:
			print('end signal')
			break


	plt.plot(x,signal)
	plt.grid()
	plt.show()
