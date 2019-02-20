"""
Adapted from Udacity's exercise on implementing Deep Q-Learning 
to solve OpenAI Gym's LunarLander environment. 
"""

import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from environment import environment
from dqn_agent import Agent


x, y = np.meshgrid(np.linspace(0,100,1000), np.linspace(0,100,1000))
d = np.sqrt((x-50)**2 + 4*(y-50)**2)
sigma, mu = 15.0, 0.0
MAP = 100*np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

""""
print (MAP)
h = plt.contourf(x,y,MAP)
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()
"""

def main():
	"""
	agent = Agent(state_size=3, action_size=8, seed=0)

	start_pos = (4,0)
	end_pos = (6,2)
	env = environment(MAP, start_pos, end_pos)

	# watch an untrained agent
	state, _, _ = env.reset(start_pos, end_pos)

	for j in range(10):
		action = agent.act(state)
		print (action)
		assert action < 8

		env.render()
		state, reward, done = env.step(action)
		print (state, reward, done)
		if done:
			break 
			
	env.close()
	"""

	
	scores = dqn()

	# plot the scores
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(len(scores)), scores)
	plt.ylabel('Score')
	plt.xlabel('Episode #')
	plt.show()
	

	



def dqn(n_episodes=4000, max_t=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
	agent = Agent(state_size=3, action_size=8, seed=0)

	start_pos = (200,600)
	end_pos = (800,375)
	env = environment(MAP, start_pos, end_pos)

	"""Deep Q-Learning.
	
	Params
	======
		n_episodes (int): maximum number of training episodes
		max_t (int): maximum number of timesteps per episode
		eps_start (float): starting value of epsilon, for epsilon-greedy action selection
		eps_end (float): minimum value of epsilon
		eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
	"""
	scores = []                        # list containing scores from each episode
	scores_window = deque(maxlen=100)  # last 100 scores
	eps = eps_start                    # initialize epsilon
	
	for i_episode in range(1, n_episodes+1):
		state, _, _ = env.reset(start_pos, end_pos)
		score = 0

		for t in range(max_t):
			action = agent.act(state, eps)
			next_state, reward, done = env.step(action)
			agent.step(state, action, reward, next_state, done)
			state = next_state
			score += reward
			
			if done:
				#print (state)
				break 
		
		scores_window.append(score)       # save most recent score
		scores.append(score)              # save most recent score
		eps = max(eps_end, eps_decay*eps) # decrease epsilon
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
		
		if i_episode % 100 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
		
		#if np.mean(scores_window)>=200.0:
			#print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
		
		if i_episode % 200 == 0:
			torch.save(agent.qnetwork_local.state_dict(), 'checkpoint' + str(i_episode) + '.pth')

		#torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
			#break
	
	return scores



if __name__ == '__main__' :
	main()
