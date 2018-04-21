import tensorflow as tf
import numpy as np
from collections import deque

def as_binary_array(x:int, length=None):
    length = length if length is not None else x.bit_length
    b = []
    for i in range(length):
        b.append(x & 1)
        x = x >> 1
    b.reverse()
    return np.array(b)

def one_hot(values, n_values:int):
    return np.eye(n_values)[values]

def flatten(x):
    return np.reshape(x, [-1])

class DeepQAgent(object):
    def __init__(self,
                 explore_start=1.0,
                 explore_stop=1e-2,
                 decay_rate=1e-4,
                 memory_size=10000,
                 num_actions=2,
                 batch_size=20,
                 gamma=0.99,
                 memory_prepop=None,
                 *args,
                 **kwargs):
        self.net = DeepQNet(num_actions=num_actions, *args, **kwargs)
        
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.memory = Memory(memory_size)
        self.noise = DecayProcess(explore_start=explore_start, explore_stop=explore_stop, decay_rate=decay_rate)
        
        self.losses = []
        self.total_rewards = []
        self._episode_reward = 0
        
        if memory_prepop is not None:
            for d in memory_prepop:
                self.memory.add(d)
    
    def step(self, sess, state, action, reward, next_state, done):
        self.noise.step()
        self.memory.add((state, action, reward, next_state, done))
        self._episode_reward += reward
        if done:
            self.total_rewards.append(self._episode_reward)
            self._episode_reward = 0
            if self.memory.count() > self.batch_size:
                loss = self.learn(sess, gamma=self.gamma)
                self.losses.append(loss)

    def learn(self, sess, gamma=0.99):
        states, actions, rewards, next_states, episode_ends = zip(*self.memory.sample(self.batch_size))
        targets = self.net.compute_targets(sess, np.array(rewards), np.array(next_states), list(episode_ends), gamma=gamma)
        loss = self.net.learn(sess, states, actions, targets)
        return loss
                
    def act(self, sess, state, train=False):
        if train:
            if self.noise.sample() == 1:
                action = np.random.randint(self.num_actions)
            else:
                action = self.net.act(sess, state)
        else:
            action = self.net.act(sess, state)
        return action
   
    def pretrain(self, env, pretrain_length):
        state = flatten(env.reset())
        for i in range(pretrain_length):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            next_state = flatten(next_state)
            self.memory.add((state, action, reward, next_state, done))
            state = next_state
            if done:
                state = flatten(env.reset())


class DeepQNet(object):
    def __init__(self,
                 learning_rate=1e-2,
                 state_shape=4,
                 num_actions=2,
                 hidden=16,
                 name='DeepQNet'
                ):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_shape], name='state')
            self.action = tf.placeholder(tf.int32, [None], name='action')
            action_one_hot = tf.one_hot(self.action, num_actions)
            
            self.target = tf.placeholder(tf.float32, [None], name='target')
            
            self.hidden0 = tf.contrib.layers.fully_connected(self.state, hidden)
            self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0, hidden)
            
            self.value = tf.contrib.layers.fully_connected(self.hidden1, num_actions,
                                                           activation_fn=None)
            
            self.predicted_reward = tf.reduce_sum(tf.multiply(self.value, action_one_hot), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.target - self.predicted_reward))
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def learn(self, sess, states, actions, targets):
        loss, _ = sess.run([self.loss, self.opt], feed_dict={
            self.state: states,
            self.target: targets,
            self.action: actions,
        })
        return loss
    
    def act(self, sess, state):
        value = sess.run(self.value, feed_dict={
            self.state: state.reshape((1, *state.shape))
        })
        action = np.argmax(value)
        return action
    
    def compute_targets(self, sess, rewards, next_states, episode_ends, gamma=0.9):
        next_values = sess.run(self.value, feed_dict={
            self.state: next_states,
        })
        try:
            next_values[episode_ends] = np.zeros(next_values.shape[1:])
        except:
            print(next_values.shape)
            print(len(episode_ends))
            raise
        targets = rewards + gamma * np.max(next_values, axis=1)
        return targets

    
class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def count(self):
        return len(self.buffer)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]
    

class DecayProcess(object):
    def __init__(self, explore_start=1.0, explore_stop=1e-2, decay_rate=1e-4):
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.explore_range = self.explore_start - self.explore_stop
        self.counter = 0
        
    def step(self):
        self.counter += 1
    
    def sample(self, size=None):
        epsilon = self.explore_stop + self.explore_range * np.exp(-self.decay_rate * self.counter)
        return (epsilon > np.random.random(size=size)).astype(np.int32)
    
    def reset(self):
        self.counter = 0
    
    
def play(env, agent):
    with tf.Session() as sess:
        saver.restore(sess, "checkpoints/cartpole.ckpt")
        state = env.reset()
        action = env.action_space.sample()
        env.render()
        state, reward, done, info = env.step(action)
        env.render()
        done = False
        total_reward = 0
        total_reward += reward
        while not done:
            value = sess.run(agent.value, feed_dict={
                agent.state: [state],
            })
            action = np.argmax(value)
            state, reward, done, info = env.step(action)
            total_reward += reward
            print(state, action, reward)
        print(total_reward)