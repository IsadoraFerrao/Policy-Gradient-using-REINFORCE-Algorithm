# Policy Gradient Implementation
# REINFORCE
# Slides from Pieter Abbeel

import tensorflow as tf
import numpy as np

#Define Actor Class
class Actor:

	#Instantiate with state and sction dimensions
	def __init__(self,dim_state, dim_action):
		self.sess = tf.Session()
		self.dim_state = dim_state
		self.dim_action = dim_action

		

	#Creates Neural Network
	def createModel(self):

		n_hidden_1 = 10
		tf.set_random_seed(1234)
		# Weights
		weights = {
    		'h1': tf.Variable(tf.random_normal([self.dim_state, self.dim_action])),
    	
    		#'out': tf.Variable(tf.random_normal([n_hidden_1, self.dim_action]))
		}

		# Biases
		biases = {
    		'b1': tf.Variable(tf.random_normal([self.dim_action])),
    		#'out': tf.Variable(tf.random_normal([self.dim_action]))
		}

		#Input Layer
		self.x = tf.placeholder(tf.float32, [None, self.dim_state])

		# Hidden Layer
		layer_1 = tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])
    		#layer_1 = tf.nn.sigmoid(layer_1)

    		# Output Layer
    		#self.out_layer = tf.nn.softmax(tf.matmul(layer_1, weights['out']) + biases['out'])
    		self.out_layer = tf.nn.softmax(layer_1)

		#Good Probabilities
		self.actions = tf.placeholder(tf.float32, [None, self.dim_action])

		self.advantages = tf.placeholder(tf.float32,[None,1])

		good_probabilities = tf.reduce_sum(tf.multiply(self.out_layer, self.actions),reduction_indices=[1])

		log_probabilities = tf.log(tf.clip_by_value(good_probabilities, 1e-10, 1.0))

		eligibility = log_probabilities * self.advantages

		self.loss = -tf.reduce_sum(eligibility)

		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


		#Initialize Variables
		self.sess.run(tf.initialize_all_variables())


	#Get action for a given state
	def act(self,state):
		"""
		@params : state
		returns : probability of each action
		"""
		
		return self.sess.run(self.out_layer, feed_dict={
            	self.x: state
        	})
		


	# Get Gradients for log probabilties
	def gradient(self,pred_action, actual_action):
		"""
		Get gradients for taken action and state
		"""
		
		pass

	#Train to update weights
	def train(self,states,actions,advantages):
		"""
		Trains neural network. Update parameters
		"""
		epochs = 100
		for _ in range(epochs):
			_,c = self.sess.run([self.optimizer, self.loss], feed_dict={
		    	self.x: states,
			self.actions: actions,
			self.advantages: advantages
			})
		return c

		
		

if __name__=="__main__":
	actor = Actor(4,2)
	actor.createModel()
	s = np.array([[1,2,3,4]])
	a = actor.act(s)
	
	state = s
	action = np.array([[1,0]])
	advantages = np.array([[1]])
	print advantages
	transitions = np.array([s,action,advantages])
	c = actor.train(transitions)
	print c


