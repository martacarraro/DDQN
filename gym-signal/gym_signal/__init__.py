from gym.envs.registration import register

register(
    id='Signal-v0',
    entry_point='gym_signal.envs:SignalEnv'
)
