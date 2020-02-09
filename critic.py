# Policy Gradients : REINFORCE algorithm
import tensorflow as tf
import numpy as np
#Define Critic Class
class Critic:
	#initialize class with input dimension 
	def __init__(self,dim_state):
		self.sess=tf.Session()
		self.dim_state=dim_state
		self.dim_value=1
		self.learning_rate=0.001
	def createModel(self):

		#declaring no of neurons for each layer
		n_hidden_1=10;
		tf.set_random_seed(1234)

		#weights for each layer
		weights={
			'h1':tf.Variable(tf.random_normal([self.dim_state, n_hidden_1])),
			'out':tf.Variable(tf.random_normal([n_hidden_1,self.dim_value]))
		}

		#biases for each layer
		biases={
			'b1':tf.Variable(tf.random_normal([n_hidden_1])),
			'out':tf.Variable(tf.random_normal([self.dim_value]))
		}

		#Input  layer
		self.x=tf.placeholder(tf.float32,[None,self.dim_state])

		#hidden layer 1
		self.layer1=tf.add(tf.matmul(self.x,weights['h1']),biases['b1'])
		self.layer1=tf.nn.relu(self.layer1)

		#Output layer
		self.y=tf.matmul(self.layer1,weights['out'])+biases['out']
		#self.y = tf.nn.relu(self.y)
		self.T=tf.placeholder(tf.float32,[None,1])

		#loss function
		self.cost=tf.nn.l2_loss(self.y-self.T)
#		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		self.sess.run(tf.initialize_all_variables())
	
	#get value given a state
	def Value(self,state):
		#get the value for a state="
		return self.sess.run(self.y,feed_dict={self.x:state})

	def train(self,state, Reward):
		#train critic neural network
		samples=np.shape(state)[1]
		display_step=1
		training_epochs=100
		for epoch in range(training_epochs):
			avg_cost=0
			_, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: state,self.T: Reward})
            		avg_cost = avg_cost + c/samples
            		#if epoch % display_step == 0:
                		#print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    		#	"{:.9f}".format(avg_cost))
        	#print("Optimization Finished!")

if __name__=="__main__":
	critic=Critic(4)
	critic.createModel()
	s=np.array([[1,2,3,4]])
	print critic.Value(s)
	critic.train(s,[[1]])
