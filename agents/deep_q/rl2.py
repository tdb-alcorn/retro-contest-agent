from functools import reduce
import tensorflow as tf
import numpy as np
from collections import deque
from agents.utils import flatten, find_action, as_binary_array, make_actions
from agents.deep_q.q_net import QNet
from agents.config import env, deep_q
cfg = deep_q['rl2']


class RL2(QNet):
    def __init__(self,
                 state_shape=env["state_shape"],
                 action_shape=env["action_shape"],
                 learning_rate=cfg['learning_rate'],
                 name='RL2'
                ):
        self.name = name
        self.actions = make_actions()
        self.num_actions = len(self.actions)
        with tf.variable_scope(name):
            # TODO Rescale inputs from (0->255) to (-1,1) in each RGB channel

            # Inputs
            self.state = tf.placeholder(tf.float32, [None, *state_shape], name='state')
            self.action = tf.placeholder(tf.int32, [None, *action_shape], name='action')
            self.target = tf.placeholder(tf.float32, [None], name='target')

            # Dropout config
            self.training = tf.placeholder(tf.bool, name='training')
            self.dropout_rate = cfg['dropout']

            # Reshape recurrent input
            action_ints = tf.py_func(find_action(self.actions), [self.action], [tf.int64])[0]
            action_one_hot = tf.one_hot(action_ints, self.num_actions)
            
            # Convolutional layers
            self.conv0_0 = tf.layers.conv2d(self.state, 16, 4, strides=4, padding='same', activation=tf.nn.relu)
            self.conv0_1 = tf.layers.conv2d(self.conv0_0, 32, 4, strides=4, padding='same', activation=tf.nn.relu)
            self.pool0 = tf.layers.max_pooling2d(self.conv0_1, 2, 2, padding='same')
            self.conv_drop0 = tf.layers.dropout(self.pool0, rate=self.dropout_rate, training=self.training)
            self.conv_out = tf.layers.flatten(self.conv_drop0)

            self.embedding = tf.layers.dense(self.conv_out, cfg['embedding'], activation=tf.nn.relu)

            out_size = reduce(lambda x,y: x*y, self.embedding.get_shape().as_list()[1:])
            self.rnn_input = tf.reshape(self.embedding, (-1, 1, out_size))

            # Recurrent layers
            def cell(size:int):
                inner = tf.nn.rnn_cell.GRUCell(size, activation=tf.nn.relu)
                return tf.nn.rnn_cell.DropoutWrapper(inner, output_keep_prob=(1-self.dropout_rate))

            rnn_layers = [out_size] + cfg['rnn_layers']
            lstm_layers = [cell(s) for s in rnn_layers]
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
            self.rnn_state = None  # initial value to be set the first time the graph is run
            self.rnn_state_feed = self.rnn_cell.zero_state(1, tf.float32)
            self.rnn_output, self.rnn_final_state = tf.nn.dynamic_rnn(self.rnn_cell, self.rnn_input, initial_state=self.rnn_state_feed)
            rnn_output_shape = self.rnn_output.get_shape().as_list()

            # Output
            self.rnn_output_cat = tf.reshape(self.rnn_output, (-1, *rnn_output_shape[2:]))
            self.value = tf.layers.dense(self.rnn_output_cat, self.num_actions,
                                                           activation=None)

            self.predicted_reward = tf.reduce_sum(tf.multiply(self.value, action_one_hot), axis=1)

            # Learn
            self.loss = tf.reduce_mean(tf.square(self.target - self.predicted_reward))
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def _ensure_rnn_state(self, sess):
        if self.rnn_state is None:
            self.rnn_state = sess.run(self.rnn_state_feed)
    
    def learn(self, sess, states, actions, targets):
        self._ensure_rnn_state(sess)
        loss, _, self.rnn_state = sess.run([self.loss, self.opt, self.rnn_final_state], feed_dict={
            self.state: states,
            self.target: targets,
            self.action: actions,
            self.training: True,
            self.rnn_state_feed: self.rnn_state,
        })
        return loss
    
    def act(self, sess, state):
        self._ensure_rnn_state(sess)
        value, self.rnn_state = sess.run([self.value, self.rnn_final_state], feed_dict={
            self.state: [state],
            self.training: False,
            self.rnn_state_feed: self.rnn_state,
        })
        action = np.argmax(np.squeeze(value)[-1])
        action = self.actions[action]
        return action
    
    def compute_targets(self, sess, rewards, next_states, episode_ends, gamma):
        self._ensure_rnn_state(sess)
        next_values = sess.run(self.value, feed_dict={
            self.state: next_states,
            self.training: False,
            self.rnn_state_feed: self.rnn_state,
        })
        try:
            next_values[episode_ends] = np.zeros(next_values.shape[1:])
        except:
            print(next_values.shape)
            print(len(episode_ends))
            raise
        targets = rewards + gamma * np.max(next_values, axis=1)
        return targets