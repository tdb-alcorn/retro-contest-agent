from functools import reduce
import tensorflow as tf
import numpy as np
from agents.utils import flatten, array_as_int, as_binary_array
from agents.deep_q.q_net import QNet
from agents.config import env, deep_q


class SimpleDeepQNet(QNet):
    def __init__(self,
                 state_shape=env["state_shape"],
                 action_shape=env["action_shape"],
                 learning_rate=deep_q['simple']['learning_rate'],
                 hidden=deep_q['simple']['num_hidden'],
                 name='SimpleDeepQNet'
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
    
    def compute_targets(self, sess, rewards, next_states, episode_ends, gamma):
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