import tensorflow as tf
from train.regimen import Regimen
from train.utils import random_if_empty


class RL2(Regimen):
    def before_training(self):
        # initialise game, level
        self.game, self.state = random_if_empty('', '', validation=False)

    def before_epoch(self, epoch:int):
        tf.reset_default_graph()
        self.agent = self.agent_constructor()
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.agent.load(self.sess)
    
    def after_epoch(self, epoch:int):
        # next game, level
        self.game, self.state = random_if_empty('', '', validation=False)
        self.agent.save(self.sess)