import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SignalEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		print('env initialized')
		
	def step(self):
		print('step success!')
		
	def reset(self):
		print('env reset')


