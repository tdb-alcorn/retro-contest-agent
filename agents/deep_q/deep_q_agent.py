from typing import TypeVar, Generic, Type
import tensorflow as tf
import numpy as np
from agents.noise import DecayProcess
from agents.agent import Agent
from agents.config import env, deep_q
from agents.deep_q.q_net import QNet

Net = TypeVar('Net', bound=QNet)

class DeepQAgent(Agent, Generic[Net]):
    # TODO(tom) Return all the params from __call__ to __init__
    # if/when python/typing/557 merges.
    # https://github.com/python/typing/pull/557
    def __init__(self,
                 net_constructor: Type[Net],
                 **kwargs):
        self.net_constructor = net_constructor
    
    def __call__(self,
                 *args,
                 explore_start=1.0,
                 explore_stop=1e-2,
                 decay_rate=1e-4,
                 gamma=0.99,
                 **kwargs):
        self.net:QNet = self.net_constructor(
            *args,
            **kwargs)
        self.gamma = gamma
        self.noise = DecayProcess(explore_start=explore_start, explore_stop=explore_stop, decay_rate=decay_rate)
        self.checkpoint_name = "checkpoints/deep_q_agent_{}.ckpt".format(type(self.net).__name__)
        
        self.losses = []
        self.total_rewards = []
        self._episode_reward = 0
        return self
        
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
        targets = self.net.compute_targets(sess, np.array(rewards), np.array(next_states), list(episode_ends), self.gamma)
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
            saver.restore(sess, self.checkpoint_name)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            print(f"deep_q_agent.load: checkpoint file not found, skipping load")

    def save(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        train_vars = tf.trainable_variables(scope=self.net.name)
        saver = tf.train.Saver(train_vars)
        saver.save(sess, self.checkpoint_name)