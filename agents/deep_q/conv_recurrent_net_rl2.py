from functools import reduce
import tensorflow as tf
import numpy as np
from collections import deque
from agents.utils import flatten, find_action, as_binary_array, make_actions
from agents.deep_q.q_net import QNet
from agents.config import env, deep_q
cfg = deep_q['conv_recurrent']


class FrameHistory(object):
    def __init__(self, num_frames:int):
        self.buffer = deque(list(), num_frames)
        self.rnn_states = deque(list(), num_frames)
    
    def all(self):
        h = list(zip(*self.buffer))
        h = [np.array(r) for r in h]
        h = [r.reshape(1, *r.shape) for r in h]
        # for i in range(len(h)):
        #     try:
        #         h[i] = np.array(h[i])
        #         h[i] = h[i].reshape(1, *h[i].shape)
        #     except ValueError:
        #         continue
        return h

    def push(self, env_state, action, target, rnn_state):
        self.buffer.append((env_state, action, target))
        self.rnn_states.append(rnn_state)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, i:int):
        return self.buffer[i]



class ConvRecurrentRL2DeepQNet(QNet):
    def __init__(self,
                 state_shape=env["state_shape"],
                 action_shape=env["action_shape"],
                 learning_rate=cfg['learning_rate'],
                 name='ConvRecurrentRL2DeepQNet'
                ):
        self.name = name
        self.num_frames = cfg['num_frames']
        self.actions = make_actions()
        self.num_actions = len(self.actions)
        self.history = FrameHistory(self.num_frames)
        with tf.variable_scope(name):
            # Inputs
            self.state = tf.placeholder(tf.float32, [None, self.num_frames, *state_shape], name='state')
            self.action = tf.placeholder(tf.int32, [None, self.num_frames, *action_shape], name='action')
            self.target = tf.placeholder(tf.float32, [None, self.num_frames], name='target')

            # Dropout config
            self.training = tf.placeholder(tf.bool, name='training')
            self.dropout_rate = cfg['dropout']

            # Reshape recurrent input
            self.state_cat = tf.reshape(self.state, (-1, *state_shape))
            self.action_cat = tf.reshape(self.action, (-1, *action_shape))
            action_ints = tf.py_func(find_action(self.actions), [self.action_cat], [tf.int64])[0]
            action_one_hot_cat = tf.one_hot(action_ints, self.num_actions)
            action_one_hot = tf.reshape(action_one_hot_cat, (-1, self.num_frames, self.num_actions))
            
            # Convolutional layers
            self.conv0_0 = tf.layers.conv2d(self.state_cat, 16, 4, strides=4, padding='same', activation=tf.nn.relu)
            self.conv0_1 = tf.layers.conv2d(self.conv0_0, 32, 4, strides=4, padding='same', activation=tf.nn.relu)
            self.pool0 = tf.layers.max_pooling2d(self.conv0_1, 2, 2, padding='same')
            self.conv_drop0 = tf.layers.dropout(self.pool0, rate=self.dropout_rate, training=self.training)
            self.conv_out = tf.layers.flatten(self.conv_drop0)

            self.embedding = tf.layers.dense(self.conv_out, cfg['embedding'], activation=tf.nn.relu)

            out_size = reduce(lambda x,y: x*y, self.embedding.get_shape().as_list()[1:])
            self.rnn_input = tf.reshape(self.embedding, (-1, self.num_frames, out_size))

            # Recurrent layers
            def cell(size:int):
                inner = tf.nn.rnn_cell.BasicLSTMCell(size)
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
            self.value_cat = tf.layers.dense(self.rnn_output_cat, self.num_actions,
                                                           activation=None)
            self.value = tf.reshape(self.value_cat, (-1, self.num_frames, self.num_actions))

            # reduce_sum is now performed over axis 2
            self.predicted_reward = tf.reduce_sum(tf.multiply(self.value, action_one_hot), axis=2)

            # Learn
            self.loss = tf.reduce_mean(tf.square(self.target - self.predicted_reward))
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def _ensure_rnn_state(self, sess):
        if self.rnn_state is None:
            self.rnn_state = sess.run(self.rnn_state_feed)
    
    def learn(self, sess, inp_states, inp_actions, inp_targets):
        if inp_targets is None:
            # TODO what to put for targets here?
            inp_targets = np.array(0)
        self._ensure_rnn_state(sess)
        self.history.push(np.squeeze(inp_states), np.squeeze(inp_actions), np.squeeze(inp_targets), self.rnn_state)
        if len(self.history) >= self.num_frames:
            states, actions, targets = self.history.all()
            loss, _, self.rnn_state = sess.run([self.loss, self.opt, self.rnn_final_state], feed_dict={
                self.state: states,
                self.target: targets,
                self.action: actions,
                self.training: True,
                self.rnn_state_feed: self.history.rnn_states[0],
            })
            self.history.rnn_states[-1] = self.rnn_state
        else:
            loss = 0
        # import pdb; pdb.set_trace()
        return loss
    
    def act(self, sess, state):
        self._ensure_rnn_state(sess)
        if len(self.history) >= self.num_frames:
            states, _, _ = self.history.all()
            # import pdb; pdb.set_trace()
            # self.rnn_state = rnn_states[0]
            value = sess.run(self.value, feed_dict={
                self.state: states,
                self.training: False,
                self.rnn_state_feed: self.history.rnn_states[0],
            })
            action = np.argmax(np.squeeze(value)[-1])
        else:
            action = 0
        action = self.actions[action]
        return action
    
    def compute_targets(self, sess, rewards, next_states, episode_ends, gamma):
        if len(self.history) >= self.num_frames:
            states, _, _ = self.history.all()
            # Add next_state to history of states, dropping oldest state
            nexts = np.concatenate([np.squeeze(states)[1:], next_states], axis=0)
            next_values = sess.run(self.value, feed_dict={
                self.state: [nexts],
                self.training: False,
                self.rnn_state_feed: self.history.rnn_states[0],
            })
            try:
                next_values[episode_ends] = np.zeros(next_values.shape[1:])
            except:
                print(next_values.shape)
                print(len(episode_ends))
                raise
            targets = rewards + gamma * np.max(next_values, axis=2)[:, -1]
            return targets
        return None