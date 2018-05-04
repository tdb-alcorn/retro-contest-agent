from agents import all_agents
from agents.agent import Agent
from retro_contest.local import make
from contextlib import closing
import tensorflow as tf
from typing import Type, Union


default_game = 'SonicTheHedgehog-Genesis'
default_level = 'LabyrinthZone.Act1'


def train(
    agent_constructor:Type[Agent],
    num_episodes:int,
    game:str=default_game,
    state:str=default_level,
    # memory:Union[Memory,None]=None,
    memory=None,
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

            log_every = max(round(num_episodes/10), 1)
            for episode in range(num_episodes):
                if episode % log_every == 0:
                    print("Episode %d" % episode)
                run_episode(sess, env, agent, memory=memory, render=render)

            agent.save(sess)


def run_episode(sess:tf.Session, env, agent:Agent, memory=None, render:bool=False):
    done = False
    state = env.reset()
    if memory is not None:
        memory.begin_episode(state)
    if render:
        env.render()
    while not done:
        action = agent.act(sess, state)
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
    parser.add_argument('--game', type=str, dest='game', default=default_game, help='name of the game')
    parser.add_argument('--level', type=str, dest='level', default=default_level, help='name of the level')
    parser.add_argument('--render', const=True, default=False, action='store_const', dest='render', help='enable rendering of training to video')
    parser.add_argument('--bk2dir', type=str, dest='bk2dir', default=None, help='optional directory to store .bk2 gameplay files')
    

    args = parser.parse_args()

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]
        
        if args.bk2dir is not None:
            import os
            fq_bk2dir = os.path.join(os.getcwd(), args.bk2dir)
            if not os.path.isdir(fq_bk2dir):
                if not os.path.exists(fq_bk2dir):
                    print("bk2 directory {} doesn't exist, creating it for you".format(fq_bk2dir))
                    os.mkdir(fq_bk2dir)
                else:
                    raise FileExistsError(fq_bk2dir)

        train(agent_constructor, args.num_episodes, game=args.game, state=args.level, render=args.render, bk2dir=args.bk2dir)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))