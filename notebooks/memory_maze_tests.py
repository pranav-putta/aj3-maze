import gym
import os

os.environ['MUJOCO_GL'] = 'glfw'

env = gym.make('memory_maze:MemoryMaze-9x9-Oracle-Top-v0')
state = env.reset()
