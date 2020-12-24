import tensorflow as tf
from tensorflow.keras.layers import (Dense, InputLayer)
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

class ReplayMemory:	

	def __init__(self,capacity):
		self.capacity = capacity
		self.num_in_memory = 0
		self.n_idx = 0
		self.states = np.empty((capacity,))
		self.actions = np.empty((capacity,))
		self.rewards = np.empty((capacity,))
		self.next_states = np.empty((capacity,))
		self.dones = np.empty((capacity,))

	def store(self,state,action,reward,next_state,done):
		self.states[self.n_idx] = state
		self.actions[self.n_idx] = action
		self.rewards[self.n_idx] = reward
		self.next_states[self.n_idx] = next_states
		self.dones[self.n_idx] = done
		self.n_idx = (self.n_idx + 1) % self.capacity
		self.num_in_memory = min(self.num_in_memory + 1, self.capacity)

	def sample(self,batch_size):
		size = batch_size
		if size > self.num_in_memory:
			size = self.num_in_memory 
		idxs = np.random.choice(self.num_in_memory,size)
		
		return self.states[idxs],self.actions[idxs],self.rewards[idxs],self.next_states[idxs]

class QNetwork(tf.keras.Model):
	
	def __init__(self,input_shape,n_actions):
		self.input_layer = InputLayer(input_shape=input_shape)
		self.fc1 = Dense(256,activation='relu')
		self.fc2 = Dense(256,activation='relu')
		self.fc3 = Dense(n_actions)

	@tf.function
	def call(self,x):
		x = self.input_layer(x)
		x = self.fc1(x)
		x = self.fc2(x)
		q_values = self.fc3(x)
		return q_values


class DQNAgent:
	def __init__(self,env,replay_memory_size,epsilon_min,epsilon_max,discount_factor,epsilon_decay_rate,batch_size,update_target_network_intvl):
		
		self.env = env
		self.n_actions = env.action_space.n
		self.state_shape = env.observation_space.shape
		self.q_net = QNetwork(input_shape=state_shape,n_actions = n_actions)
		self.target_network = QNetwork(input_shape = state_shape, n_actions = n_actions)
		self.replay_memory_size = replay_memory_size
		self.epsilon_min = epsilon_min
		self.epsilon = epsilon_max
		self.discount_factor = discount_factor
		self.replay_memory = ReplayMemory(capacity=replay_memory_size)
		self.epsilon_decay_rate = epsilon_decay_rate
		self.batch_size = batch_size
		self.update_target_network_intvl = update_target_network_intvl
		self.optimizer = Adam()

	def train(self,episodes):
		t = 0
		for episode in range(episodes):
			state = env.reset()

			while True:
				greedy_action = np.argmax(self.q_net(state[None]))
				action = self.get_action(greedy_action)
				next_state,reward,done,_ = env.step(action)
				self.replay_memory.store(state,action,reward,next_state,done)
				self.train_update()
				self.decay_epislon()
				
				if t % self.update_target_network_intvl == 0:
					self.target_network.set_weights(self.q_net.get_weights())
			if done:
				break
			else: 
				next_state = state

	def train_update(self):
		states,actions,rewards,next_states,dones = self.replay_memory.sample(self.batch_size)

		action_value_next = np.max(self.target_network(next_states),axis=1)
		actual_action_values = rewards + self.gamma * action_value_next * (1-dones)

		with tf.GradientTape() as tape:
			selected_action_values = tf.math.reduce_sum(
				self.q_net(states) * tf.one_hot(actoins,self.num_actions),axis=1)
			loss = tf.math.reduce_mean(tf.square(actual_action_values-selected_action_values))
		variables = self.q_net.trainable_variables
		gradients = tape.gradients(zip(loss,variables))

	def decay_epsilon(self):
		
		decayed_epsilon = self.epsilon * self.epsilon_decay_rate
		self.epsilon = max(self.epsilon_min,decayed_epsilon)

	def get_action(self,best_action,epsilon):
		
		if np.random.rand() < epsilon: 
			return np.random.randint(self.n_actions)
		else:
			return best_action

	def evaluate(self,episodes):
		cumulative_reward = Mean('avg_reward')

		for episode in range(episodes):
			state = env.reset()
			episode_reward = 0

			while True:
				action = np.argmax(self.q_net(state[None]))
				next_state,reward,done,_ = env.step(action)
				episode_reward += reward
			
			if done:
				print(f"episode:{episode} reward:{episode_reward}")
				cumulative_reward(episode_reward)
				break
			else:
				state = next_state 

