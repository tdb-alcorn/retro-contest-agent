from agents import all_agents
from agents.agent import Agent
from train.utils import write_to_csv, get_levels, random_if_empty, ensure_directory_exists
from retro_contest.local import make
from contextlib import closing
import tensorflow as tf
import numpy as np
from typing import Type, Union


def train(
    agent_constructor:Type[Agent],
    num_episodes:int,
    game:str,
    state:str,
    # memory:Union[Memory,None]=None,
    memory=None,
    render:bool=False,
    bk2dir=None,
    loss_filename:str='',
    ):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.WARN)

    with closing(make(game=game, state=state, bk2dir=bk2dir)) as env:
        agent = agent_constructor()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            agent.load(sess)

            log_every = max(round(num_episodes/10), 1)

            try:
                for episode in range(num_episodes):
                    if episode % log_every == 0:
                        print("Episode %d" % episode)
                    run_episode(sess, env, agent, memory=memory, render=render)
            finally:
                print("Saving agent... ", end='')
                agent.save(sess)
                print('Done.')
                if hasattr(agent, 'losses') and loss_filename != '':
                    print("Writing losses to {}... ".format(loss_filename), end='')
                    write_to_csv(loss_filename, ['Epoch', 'Loss'], list(zip(range(len(agent.losses)), agent.losses)))
                    print('Done.')

def train_all(
    agent_constructor:Type[Agent],
    num_episodes:int,
    bk2dir=None,
    loss_filename:str='',
    ):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.WARN)

    agent = agent_constructor()

    all_levels = get_levels()
    log_every = max(round(num_episodes/10), 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        agent.load(sess)

        try:
            for game, level in all_levels:
                print("Playing game {} on level {}".format(game, level))
                with closing(make(game=game, state=level, bk2dir=bk2dir)) as env:
                    for episode in range(num_episodes):
                        if episode % log_every == 0:
                            print("Episode %d" % episode)
                        run_episode(sess, env, agent)
        finally:
            print("Saving agent... ", end='')
            agent.save(sess)
            print('Done.')
            if hasattr(agent, 'losses') and loss_filename != '':
                print("Writing losses to {}... ".format(loss_filename), end='')
                write_to_csv(loss_filename, ['Epoch', 'Loss'], list(zip(range(len(agent.losses)), agent.losses)))
                print('Done.')

def run_episode(sess:tf.Session, env, agent:Agent, memory=None, render:bool=False):
    done = False
    state = env.reset()
    if memory is not None:
        memory.begin_episode(state)
    if render:
        env.render()
    while not done:
        action = agent.act(sess, state, True)
        next_state, reward, done, _ = env.step(action)
        agent.step(sess, state, action, reward, next_state, done)

        if render:
            env.render()

        if memory is not None:
            memory.add((state, action, reward, next_state, done))
        
        state = next_state
        if done:
            state = env.reset()


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train an agent online against a live game.")
    parser.add_argument('agent', type=str, help='name of the agent')
    parser.add_argument('--episodes', type=int, dest='num_episodes', default=1, metavar='N', help='number of episodes to run')
    parser.add_argument('--game', type=str, dest='game', default='', help='name of the game')
    parser.add_argument('--level', type=str, dest='level', default='', help='name of the level')
    parser.add_argument('--render', const=True, default=False, action='store_const', dest='render', help='enable rendering of training to video')
    parser.add_argument('--bk2dir', type=str, dest='bk2dir', default=None, help='optional directory to store .bk2 gameplay files')
    parser.add_argument('--output', type=str, dest='loss_filename', default='', help='file in which to save training loss data')
    parser.add_argument('--all', default=False, action='store_const', const=True, dest='all', help='run through all games and levels in sequence. --episodes is then interpreted as the number of times to repeat each episode')
    
    args = parser.parse_args()

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]

        # Ensure that the bk2 save directory is present
        if args.bk2dir is not None:
            ensure_directory_exists(args.bk2dir)

        if args.all:
            train_all(agent_constructor, args.num_episodes, bk2dir=args.bk2dir, loss_filename=args.loss_filename)
        else:
            # Choose a random/game level if none is specified
            game, level = random_if_empty(args.game, args.level)
            print("Playing game {} on level {}".format(game, level))

            # Run training
            train(agent_constructor, args.num_episodes, game, level, render=args.render, bk2dir=args.bk2dir, loss_filename=args.loss_filename)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))