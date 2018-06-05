import tensorflow as tf
from agents import all_agents
from objective import Passthrough
from train.regimen import Regimen
from train.plugins import Framerate, ManualOverride
from train.utils import random_if_empty, ensure_directory_exists


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


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train an agent online against a live game.")
    parser.add_argument('agent', type=str, help='name of the agent')
    parser.add_argument('--trials', type=int, dest='num_trials', default=1, metavar='N', help='number of trials to run')
    parser.add_argument('--episodes', type=int, dest='episodes_per_trial', default=1, metavar='N', help='number of episodes to run per trial')
    parser.add_argument('--render', const=True, default=False, action='store_const', dest='render', help='enable rendering of training to video')
    parser.add_argument('--bk2dir', type=str, dest='bk2dir', default=None, help='optional directory to store .bk2 gameplay files')
    parser.add_argument('--output', type=str, dest='out_filename', default='', help='file prefix in which to save training losses and rewards')
    
    args = parser.parse_args()
    
    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]
        objective = Passthrough()

        # Ensure that the bk2 save directory is present
        if args.bk2dir is not None:
            ensure_directory_exists(args.bk2dir)
        
        if args.out_filename[-4:] == '.csv':
            args.out_filename = args.out_filename[:-4]

        regimen = RL2(agent_constructor, objective)
        regimen.use(Framerate())
        regimen.use(ManualOverride())
        # Run training
        regimen.train('', '', args.num_trials, args.episodes_per_trial, render=args.render, bk2dir=args.bk2dir, out_filename=args.out_filename)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))
