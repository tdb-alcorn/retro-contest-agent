from typing import Dict, Type
from agents.agent import Agent
from agents.random import Random
from agents.deep_q import DeepQAgent, SimpleDeepQNet, ConvDeepQNet, ConvRecurrentDeepQNet


# all_agents: Dict[str, Agent] = {
all_agents = {
    'random': Random,
    # TODO(tom) Simplify these goofy statements to just <agent>[<net>] 
    # if/when python/typing/557 merges
    # https://github.com/python/typing/pull/557
    'simple_deep_q': DeepQAgent[SimpleDeepQNet](SimpleDeepQNet),
    'conv_deep_q': DeepQAgent[ConvDeepQNet](ConvDeepQNet),
    'conv_recurrent_deep_q': DeepQAgent[ConvRecurrentDeepQNet](ConvRecurrentDeepQNet),
}


# TODO(tom) This doesn't work because python only runs submodules that it
# needs to import at runtime, so the @register calls made by each agent never occur.
# If I every figure out how to get around this without resorting to managing local state,
# re-enable the @register decorator.
# class register(object):
#     def __init__(self, agent_name:str, overwrite:bool=False):
#         self.agent_name = agent_name
#         self.overwrite = overwrite

#     def __call__(self, cls:Type[Agent]) -> Type[Agent]:
#         if not self.overwrite and self.agent_name in all_agents:
#             raise KeyError("{} already exists".format(self.agent_name))
#         all_agents[self.agent_name] = cls
#         return cls