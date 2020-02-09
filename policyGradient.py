#main file
#Policy Gradient: using REINFORCE Algorithm
import gym
import matplotlib.pyplot as plt #for visualization
import critic
import actor
import random
import time

import numpy as np
NUM_EPOCHS = 2000
GAMMA = 0.99

no_of_actions=2 #find depeding on which environment it is

render_plot=True
index_list=[0]
reward_list = [0]

# reproducible
np.random.seed(1234)

if __name__=="__main__":
	
	if render_plot:	
			fig = plt.figure()
			ax = fig.add_subplot(111)
			li, = ax.plot(index_list, reward_list,'-')	
			# draw and show it
			fig.canvas.draw()
			plt.show(block=False)
	epsilon =0.6
	#initialize gym environment
	env = gym.make("CartPole-v0")
	observation = env.reset()
	print observation.shape
	rewardList = []

	# Monitor Env
	#env.monitor.start('cartpole-hill/', force=True)

	#Initialize Actor
	actor = actor.Actor(4,2)
	actor.createModel()	
	#Initialize Critic
	critic = critic.Critic(4)
	critic.createModel()	
	#for n policies 		
	for i in range(NUM_EPOCHS):
		#epsilon=epsilon-0.4*(float(i)/NUM_EPOCHS)
		if epsilon > 0.1:
			epsilon=epsilon-0.001
		#for each rollout <s,a,r>
		observation_old = env.reset()
		T=[]
		tot_reward = 0
		for k in range(1000):
			#env.render()
			observation_old = np.reshape(observation_old,(1,4))
			action_prob = actor.act(observation_old)
				
			if np.random.uniform() < action_prob[0][0]:
				action = 0
			else:
				action = 1
			print "action prob: ",action_prob,"Action taken: ",action
			observation_new, reward, done, info = env.step(action)
			observation_new = np.reshape(observation_new,(1,4))
			T.append([observation_old,action,reward])
			tot_reward = tot_reward + reward
		        observation_old=observation_new
			if done:
				index_list.append(i)
				reward_list.append(tot_reward)
				if render_plot:
						# set the new data
						li.set_xdata(index_list)
						li.set_ydata(reward_list)
						ax.relim() 
						ax.autoscale_view(True,True,True) 
						fig.canvas.draw()
						time.sleep(0.01)
					
				break
		print "Episode:",i,"rollout length:",k,"Total Reward : ",tot_reward		
		#print T
		rewardList.append(tot_reward)

		# Get Rt then T will become <s,a,r,R>
		#T[len(T)-1].append(T[len(T)-1][2])		
		for u in range(len(T)-1):
			#T[i].append(T[i][2]+GAMMA*T[i+1][3])
			#TD Update : R(s) = r + gamma*V(s+1)
			T[u].append(T[u][2] + GAMMA*critic.Value(T[u+1][0]))
		T[len(T)-1].append(T[len(T)-1][2]) 

		# find bt which is a value from critic, then T becomes <s,a,r,R,b>
		for p in range(len(T)):
			T[p].append(critic.Value(T[p][0]))

		#find A which is Advantage (R-b), then T becomes <s,a,r,R,b,A>
		for w in range(len(T)):
			
			T[w].append(T[w][-2]-T[w][-1])

		#train critic using states and R (actual values from environment)
		states=T[0][0]	
		for n in range(1,len(T)):		
			states=np.vstack((states,T[n][0]))
		values=T[0][3]
		for m in range(1,len(T)):
			values=np.vstack((values,np.array(T[m][3])))		
		#print "states shape:",np.shape(states)
		#print "values shape:",np.shape(values)
		critic.train(states,values)


		#train actor using states actions and advantage (for computing gradients too)
		advantages=T[0][5]	
		for l in range(1,len(T)):		
			advantages=np.vstack((advantages,T[l][5]))
		#advantages = 1.0*(advantages - np.mean(advantages))/(1*np.std(advantages))

		#z=np.zeros((np.shape((1,no_of_actions))))		
		#t=z
		#t[[T[0][1]]]=1		
		#actions=t
		#for i in range(1,len(T)): 
			
		#	t=z
	#		t[[T[i][1]]]=1
	#		actions=np.vstack((actions,t))
		#print np.shape(states),np.shape(actions),np.shape(advantages)

		actions = np.zeros((len(T), 2))
		for k in range(len(T)):
			actions[k][T[k][1]] = 1
		actor.train(states,actions,advantages)
	
	#env.monitor.close()
	plt.plot(rewardList)
	plt.show()

		
