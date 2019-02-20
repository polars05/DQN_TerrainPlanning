import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from environment import environment
from dqn_agent import Agent

import sys 
import gym

x, y = np.meshgrid(np.linspace(0,100,1000), np.linspace(0,100,1000))
d = np.sqrt((x-50)**2 + 4*(y-50)**2)
sigma, mu = 15.0, 0.0
MAP = 100*np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

def main():
	agent = Agent(state_size=3, action_size=8, seed=0)

	start_pos = (200,600)
	end_pos = (800,375)
	#start_pos = (200,500)
	#end_pos = (800,500)
	env = environment(MAP, start_pos, end_pos)

	"""
	x_end, y_end = end_pos
	plt.figure(figsize=(10,6), dpi=200)
	plt.plot(start_pos[0], start_pos[1], 'rx')
	plt.plot(x_end, y_end, 'bx')
	plt.contourf(np.array(MAP), linestyles='dashed')
	#plt.imshow(np.array(MAP))
	plt.gca().set_aspect('equal', adjustable='box')
	plt.colorbar()
	plt.show()
	sys.exit(())
	"""

	# load the weights from file
	agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

	for i in range(1):
		path_x = [start_pos[0]]
		path_y = [start_pos[1]]

		state, _, _ = env.reset(start_pos, end_pos)
		for j in range(6000):
			action = agent.act(state)
			
			#print (j, action)
			print (j)
			#if j%100 == 0:
			#	env.render()

			state, reward, done = env.step(action)
			
			path_x.append(state[0])
			path_y.append(state[1])
			
			if done:
				break 
		
		print (done)
		x_end, y_end = end_pos
		plt.figure(figsize=(10,6), dpi=200)
		plt.plot(path_x, path_y, 'ro', markevery=20)
		plt.plot(x_end, y_end, 'bx')
		plt.contourf(np.array(MAP), linestyles='dashed')
		#plt.imshow(np.array(MAP))
		plt.gca().set_aspect('equal', adjustable='box')
		plt.colorbar()
		plt.show()

	env.close()



if __name__ == '__main__' :
	main()
