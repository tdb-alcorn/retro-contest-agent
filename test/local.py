from retro_contest.local import make
from contextlib import closing
from agents import all_agents, Agent
from train.utils import random_if_empty, ensure_directory_exists
from typing import Type
import tensorflow as tf

def main(
    agent_constructor:Type[Agent],
    game:str,
    state:str,
    render:bool=False,
    bk2dir=None,
    ):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.WARN)

    with closing(make(game=game, state=state, bk2dir=bk2dir)) as env:
        agent = agent_constructor()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            agent.load(sess)

            total_reward = 0
            done = False
            state = env.reset()
            if render:
                env.render()
            while not done:
                action = agent.act(sess, state, False)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                if render:
                    env.render()
                agent.step(sess, state, action, reward, next_state, done)
                state = next_state
                if done:
                    print('Episode complete. Total reward: {}'.format(total_reward))


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run an agent for evaluation locally.")
    parser.add_argument('agent', type=str, help='name of the agent')
    parser.add_argument('--game', type=str, dest='game', default='', help='name of the game')
    parser.add_argument('--level', type=str, dest='level', default='', help='name of the level')
    parser.add_argument('--render', const=True, default=False, action='store_const', dest='render', help='enable rendering of training to video')
    parser.add_argument('--bk2dir', type=str, dest='bk2dir', default=None, help='optional directory to store .bk2 gameplay files')

    args = parser.parse_args()

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]

        # Ensure that the bk2 save directory is present
        if args.bk2dir is not None:
            ensure_directory_exists(args.bk2dir)

        # Choose a random/game level if none is specified
        game, level = random_if_empty(args.game, args.level)
        print("Playing game {} on level {}".format(game, level))

        main(agent_constructor, game, level, args.render, args.bk2dir)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))