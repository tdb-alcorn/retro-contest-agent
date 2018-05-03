import tensorflow as tf
import numpy as np
from collections import deque
from agents.utils import flatten, array_as_int, as_binary_array
from agents.agent import Agent
from agents.config import env, deep_q
from functools import reduce

class DeepQAgent(Agent):
    def __init__(self,
                 explore_start=1.0,
                 explore_stop=1e-2,
                 decay_rate=1e-4,
                 gamma=0.99,
                 **kwargs):
        self.net = DeepQNet(
            state_shape=env["state_shape"],
            action_shape=env["action_shape"],
            learning_rate=deep_q["learning_rate"],
            hidden=deep_q["num_hidden"],
            **kwargs)
        
        self.gamma = gamma
        
        self.noise = DecayProcess(explore_start=explore_start, explore_stop=explore_stop, decay_rate=decay_rate)
        
        self.losses = []
        self.total_rewards = []
        self._episode_reward = 0
        
    def step(self,
        sess:tf.Session,
        state:np.array,
        action:np.array,
        reward:float,
        next_state:np.array,
        done:bool,
        ):
        self.noise.step()
        self._episode_reward += reward
        if done:
            self.total_rewards.append(self._episode_reward)
            self._episode_reward = 0
            # TODO make an agent wrapper AgentWithMemory
            # if self.memory.count() > self.batch_size:
            #     loss = self.learn(sess, gamma=self.gamma)
            #     self.losses.append(loss)

    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array,
        ) -> float:
        targets = self.net.compute_targets(sess, np.array(rewards), np.array(next_states), list(episode_ends), gamma=self.gamma)
        loss = self.net.learn(sess, states, actions, targets)
        return loss
                
    def act(self, sess, state, train=False):
        if train:
            if self.noise.sample() == 1:
                action = np.random.randint(self.net.num_actions)
            else:
                action = self.net.act(sess, state)
        else:
            action = self.net.act(sess, state)
        return action
   
    def load(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        train_vars = tf.trainable_variables(scope=self.net.name)
        saver = tf.train.Saver(train_vars)
        try:
            saver.restore(sess, "checkpoints/deep_q_agent.ckpt")
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            print(f"deep_q_agent.load: checkpoint file not found, skipping load")

    def save(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        train_vars = tf.trainable_variables(scope=self.net.name)
        saver = tf.train.Saver(train_vars)
        saver.save(sess, "checkpoints/deep_q_agent.ckpt")

class DeepQNet(object):
    def __init__(self,
                 learning_rate=1e-2,
                 state_shape=[4],
                 action_shape=[2],
                 hidden=16,
                 name='DeepQNet'
                ):
        self.name = name
        self.num_actions = 2**reduce(lambda x,y: x*y, action_shape)
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, *state_shape], name='state')
            self.action = tf.placeholder(tf.int32, [None, *action_shape], name='action')
            action_ints = tf.py_func(array_as_int, [self.action], [tf.int64])[0]
            action_one_hot = tf.one_hot(action_ints, self.num_actions)
            # action_one_hot = tf.one_hot(self.action, num_actions)
            
            self.target = tf.placeholder(tf.float32, [None], name='target')
            
            self.state_flat = tf.layers.flatten(self.state)

            self.hidden0 = tf.contrib.layers.fully_connected(self.state_flat, hidden)
            self.hidden1 = tf.contrib.layers.fully_connected(self.hidden0, hidden)
            
            self.value = tf.contrib.layers.fully_connected(self.hidden1, self.num_actions,
                                                           activation_fn=None)
            
            self.predicted_reward = tf.reduce_sum(tf.multiply(self.value, action_one_hot), axis=1)
            # self.predicted_reward = tf.reduce_sum(tf.multiply(self.value, self.action), axis=1)
            
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
        return as_binary_array(action, length=(self.num_actions-1).bit_length())
    
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
    
    
