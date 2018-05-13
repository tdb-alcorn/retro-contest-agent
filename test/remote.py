import gym_remote.exceptions as gre
import gym_remote.client as grc
from agents import all_agents, Agent
import argparse
from typing import Type
import tensorflow as tf


def main(agent_constructor:Type[Agent]):
    print('connecting to remote environment')
    env = grc.RemoteEnv('tmp/sock')

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.WARN)

    agent = agent_constructor()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        agent.load(sess)

        print('starting episode')
        state = env.reset()
        while True:
            action = agent.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            agent.step(sess, state, action, reward, next_state, done)
            state = next_state
            if done:
                print('episode complete')
                state = env.reset()


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run an agent for evaluation.")
    parser.add_argument('agent', type=str, help='name of the agent')

    args = parser.parse_args()

    if args.agent in all_agents:
        agent_constructor = all_agents[args.agent]
        try:
            main(agent_constructor)
        except gre.GymRemoteError as e:
            print('exception', e)
    else:
        sys.stderr.write('Agent {} not found. Available agents are: {}.\n'.format(args.agent, ', '.join(all_agents.keys())))