from .regimen import Regimen
import tensorflow as tf
from train.utils import random_if_empty


class RL2(Regimen):
    def before_epoch(self):
        tf.reset_default_graph()
        self.agent = self.agent_constructor()
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.agent.load(self.sess)
        self.game, self.level = random_if_empty('', '', validation=False)
    
    def after_epoch(self):
        self.agent.save(self.sess)