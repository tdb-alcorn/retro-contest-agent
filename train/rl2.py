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
    num_trials:int,
    episodes_per_trial:int,
    render:bool=False,
    bk2dir=None,
    out_filename:str='',
    ):
    tf.logging.set_verbosity(tf.logging.WARN)

    total_rewards = list()
    losses = list()

    for trial in range(num_trials):
        print("Trial {}".format(trial))
        tf.reset_default_graph()
        agent = agent_constructor()

        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                agent.load(sess)
                try:
                    game, level = random_if_empty('', '', validation=False)
                    print("Playing game {} on level {}".format(game, level))
                    with closing(make(game=game, state=level, bk2dir=bk2dir)) as env:
                        for episode in range(episodes_per_trial):
                            print("Episode {}".format(episode))
                            total_reward = run_episode(sess, env, agent, render=render)
                            total_rewards.append((trial, episode, total_reward))
                finally:
                    print("Saving agent... ", end='')
                    agent.save(sess)
                    print('Done.')
                    if hasattr(agent, 'losses'):
                        numbered_losses = list(zip(range(len(agent.losses))))
                        losses.extend([(trial, *row) for row in numbered_losses])
        finally:
            if out_filename != '':
                loss_filename = "{}_loss.csv".format(out_filename)
                reward_filename = "{}_reward.csv".format(out_filename)
                print("Writing losses to {}... ".format(loss_filename), end='')
                write_to_csv(loss_filename, ['Trial', 'Frame', 'Loss'], losses)
                print('Done.')
                print("Writing rewards to {}... ".format(reward_filename), end='')
                write_to_csv(reward_filename, ['Trial', 'Episode', 'Reward'], total_rewards)
                print('Done.')


def run_episode(sess:tf.Session, env, agent:Agent, render:bool=False) -> float:
    total_reward = 0.0
    frame = 0
    done = False
    state = env.reset()
    if render:
        env.render()
    while not done:
        # TODO Print frame and reward here (overwrite)
        action = agent.act(sess, state, True)
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        total_reward += reward
        agent.step(sess, state, action, reward, next_state, done)
        state = next_state
        print("\033[K", end='\r')
        print("Frame: {:d}\tReward: {:.2f}\tTotal: {:.2f}".format(frame, reward, total_reward), end='\r')
        frame += 1
    print()
    return total_reward


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

        # Ensure that the bk2 save directory is present
        if args.bk2dir is not None:
            ensure_directory_exists(args.bk2dir)
        
        if args.out_filename[-4:] == '.csv':
            args.out_filename = args.out_filename[:-4]

        # Run training
        train(agent_constructor, args.num_trials, args.episodes_per_trial, render=args.render, bk2dir=args.bk2dir, out_filename=args.out_filename)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))