from agents.agent import Agent
from agents import all_agents
from train.generate import Memory
from typing import Type
import tensorflow as tf


def train(
    agent_constructor:Type[Agent],
    memory:Memory,
    num_episodes:int,
    batch_size:int
    ):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.WARN)

    agent:Agent = agent_constructor()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        agent.load(sess)

        log_every = max(round(num_episodes/10), 1)
        for episode in range(num_episodes):
            batch = memory.sample(batch_size=batch_size, single_episode=False, sequential=False)
            loss = agent.learn(sess, *zip(*batch))
            if episode % log_every == 0:
                print("Episode %d\tLoss: %.2f" % (episode, loss))

        agent.save(sess)

if __name__ == '__main__':
    import argparse
    import sys
    import glob

    parser = argparse.ArgumentParser(description="Train an agent offline against saved data.")
    parser.add_argument('agent', type=str, help='name of the agent')
    parser.add_argument('--episodes', type=int, dest='num_episodes', default=1, metavar='N', help='number of episodes to run')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=100, metavar='M', help='number of data in each training batch')
    parser.add_argument('--data', type=str, dest='data', default='./**/*_*_*_*.npz', help='glob pattern matching all data files to be loaded in training')

    args = parser.parse_args()

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]

        memory = Memory()
        filenames = glob.glob(args.data, recursive=True)
        memory.load(filenames)

        train(agent_constructor, memory, args.num_episodes, args.batch_size)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))
