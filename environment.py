import gym
import torch

import numpy as np
import matplotlib.pyplot as plt
from random import randint

import sys

"""
MAP = [
	[0,0,0,0,0,10,10,10,10,10,0,0,0,0,0],
	[0,0,0,0,0,10,10,10,10,10,0,0,0,0,0],
	[0,0,0,0,0,10,10,10,10,10,0,0,0,0,0],
	[0,0,0,0,0,10,10,10,10,10,0,0,0,0,0],
	[0,0,0,0,0,10,10,10,10,10,0,0,0,0,0],
	[0,0,0,0,0,10,10,10,10,10,0,0,0,0,0],
	[0,0,0,0,0,10,10,10,10,10,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]
"""

class environment():
	def __init__(self, MAP, pos, destination):
		self.terrain = np.array(MAP)
		y_len, x_len = self.terrain.shape
		self.x_max, self.y_max = x_len-1, y_len-1
		
		self.pos = pos
		x, y = self.pos

		self.destination = destination
		x_end, y_end = self.destination

		if pos == destination: #TODO: how to prevent agent from taking step if pos is initialized as dest??
			self.done = True
		else:
			self.done = False

		#self.mask_loc = np.zeros(self.terrain.shape)
		#self.mask_loc[y][x] = 1
		#self.mask_loc[y_end][x_end] = 1

	def step(self, action):
		x, y = self.pos
		x_dest, y_dest = self.destination
		elevation_cur = self.terrain[y][x]
		reward = 0

		#self.mask_loc[y][x] = 0 #set current pos of marker in mask to be 0
		temp = np.sqrt(abs(x_dest - x)**2 + abs(y_dest - y)**2)

		if action == 0: #up
			y = max(0, y-1)
		elif action == 1: #down
			y = min(self.y_max, y+1)
		elif action == 2: #left
			x = max(0, x-1)
		elif action == 3: #right
			x = min(self.x_max, x+1)
		elif action == 4: #up-right
			if y-1 >= 0 and x+1 <= self.x_max:
				y = max(0, y-1)
				x = min(self.x_max, x+1)
		elif action == 5: #up-left
			if y-1 >= 0 and x-1 >= 0:
				y = max(0, y-1)
				x = max(0, x-1)
		elif action == 6: #down-right
			if y+1 <= self.y_max and x+1 <= self.x_max:
				y = min(self.y_max, y+1)
				x = min(self.x_max, x+1)
		elif action == 7: #down-left
			if y+1 <= self.y_max and x-1 >= 0:
				y = min(self.y_max, y+1)
				x = max(0, x-1)

		self.pos = (x, y)
		elevation_next = self.terrain[y][x]

		########## rewards ##########
		if x == x_dest and y == y_dest:
			reward += 10
			self.done = True
		else:
			reward += -1 #penalize for every step taken (to encourage minimizing total distance travelled)
		if np.sqrt(abs(x_dest - x)**2 + abs(y_dest - y)**2) < temp:
			reward += 2
		reward -= abs(elevation_cur-elevation_next) #penalize for going upslope
		########## /rewards ##########

		#self.mask_loc[y][x] = 1 #update current pos of marker in mask to be 1

		return np.array([x, y, self.terrain[y][x]]), reward, self.done

	def reset(self, pos, destination):
		self.pos = pos
		x, y = self.pos

		self.destination = destination
		self.done = False

		reward = None

		return np.array([x, y, self.terrain[y][x]]), reward, self.done

	def close(self):
		self.terrain = None
		self.pos = None
		self.destination = None
		self.x_max = None
		self.y_max = None
		self.done = None

	def render(self):
		x, y = self.pos
		x_end, y_end = self.destination
		plt.figure(figsize=(10,6), dpi=200)
		plt.plot(x, y, 'rx')
		plt.plot(x_end, y_end, 'bx')
		plt.imshow(self.terrain, interpolation="nearest")
		plt.gca().set_aspect('equal', adjustable='box')
		plt.colorbar()
		plt.show()


def main():
	start_pos = (4,0)
	end_pos = (6,2)
	env = environment(MAP, start_pos, end_pos)
	env.render()

	new_pos, reward, completed = env.step(6)
	env.render()
	print (new_pos, reward, completed)

	new_pos, reward, completed = env.step(6)
	env.render()
	print (new_pos, reward, completed)

	new_pos, reward, completed = env.reset(start_pos, end_pos)
	env.render()
	print (new_pos, reward, completed)

	env.close()

if __name__ == '__main__' :
	main()