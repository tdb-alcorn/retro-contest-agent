from agents.agent import Agent
from retro_contest.local import make
import tensorflow as tf
from typing import Type

def train(
    agent_constructor:Type[Agent],
    game:str='SonicTheHedgehog-Genesis',
    state:str='LabyrinthZone.Act1'
    ):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.WARN)

    agent = agent_constructor()

    with tf.Session as sess:
