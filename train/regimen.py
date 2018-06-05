import tensorflow as tf
from typing import Type, Union, NamedTuple
from contextlib import closing
from agents import Agent
from objective import Objective
from play import KeyboardController


class Regimen(object):
    # Properties
    # sess: tf.Session -- the tensorflow session
    # teaching: bool -- whether to read input from the controller
    # message: str -- to be logged at each time step
    # plugins: List[Regimen] -- other regimens to be run at each step
    # agent_constructor: Type[Agent] -- agent constructor
    # agent: Agent -- the agent
    # env -- the environment
    # objective -- the objective
    def __init__(self,
        agent_constructor:Type[Agent],
        objective:Objective,
    ):
        self.message = list()
        self.plugins = list()
        self.objective = objective
        self.agent_constructor = agent_constructor
        self.rewards = list()
        self.losses = list()

        # placeholders
        self.sess = None
        self.env = None
        self.agent = None
        self.action = None
        self.game = ''
        self.state = ''
    
    def log(self, message:str):
        self.message.append(message)
    
    def train(self,
        game:str,
        state:str,
        epochs:int,
        episodes_per_epoch:int,
        render:bool=False,
        bk2dir=None,
        out_filename:str='',
    ):
        self.game = game
        self.state = state

        tf.logging.set_verbosity(tf.logging.WARN)
        tf.reset_default_graph()

        for plugin in self.plugins:
            plugin.before_training()
        self.before_training()

        for epoch in epochs:
            self.run_epoch(
                epoch,
                episodes_per_epoch,
                render=render,
                bk2dir=bk2dir,
                out_filename=out_filename
            )
        
        for plugin in self.plugins:
            plugin.after_training()
        self.after_training()

    def run_epoch(self,
        epoch:int,
        episodes:int,
        render:bool=False,
        bk2dir:Union[str, None]=None,
        out_filename:str='',
    ):
        try:
            print("Playing game {} on level {}".format(self.game, self.level))
            with closing(make(game=self.game, state=self.state, bk2dir=bk2dir)) as env:
                self.env = env

                for plugin in self.plugins:
                    plugin.before_epoch(epoch)
                self.before_epoch(epoch)

                for episode in range(episodes):
                    reward = self.run_episode(episode, render=render)
                    self.rewards.append((epoch, self.game, self.state, episode, reward))

                for plugin in self.plugins:
                    plugin.after_epoch(epoch)
                self.after_epoch(epoch)

        finally:
            print("Saving agent... ", end='')
            self.agent.save(self.sess)
            print('Done.')
            if hasattr(self.agent, 'losses'):
                losses = self.agent.losses
                numbered_losses = list(zip(range(len(losses)), losses))
                self.losses.extend([(epoch, *row) for row in numbered_losses])
    
    def run_episode(self,
        episode:int,
        render:bool=False
    ) -> float:
        total_reward = 0

        state = self.env.reset()
        step = Step(state)

        for plugin in self.plugins:
            plugin.before_episode(self, episode)
        self.before_episode(episode)

        while not step.done:
            # Clear log message
            self.message = list()

            step.action = self.agent.act(self.sess, step.state, True)

            for plugin in self.plugins:
                plugin.before_step(self, step)
            self.before_step(step)

            try:
                next_state, reward, done, info = self.step(step, render=render)
                step.update(action, next_state, reward, done, info)
                total_reward += reward
            except e:
                for plugin in self.plugins:
                    if plugin.on_error(self, step, e):
                        raise
                if self.on_error(step, e):
                    raise

            # Add basic message prefix
            self.message = [
                "Frame: {:d}".format(self.frame),
                "Reward: {:.2f}".format(reward),
                "Total: {:.2f}".format(total_reward),
            ] + self.message
            print("\033[K", end='\r')
            print('\t'.join(self.message), end='\r')

            for plugin in self.plugins:
                plugin.after_step(self, step)
            self.after_step(step)
        
        for plugin in self.plugins:
            plugin.after_episode(self, episode)
        self.after_episode(episode)

    def use(self, plugin:Plugin):
        self.plugins.append(plugin)

    def step(self, render:bool=False):
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        agent.step(sess, state, action, reward, next_state, done)

    def before_epoch(self, epoch:int):
        pass
    
    def after_epoch(self, epoch:int):
        pass

    def before_episode(self, episode:int):
        pass
    
    def after_episode(self, episode:int):
        pass
    
    def before_step(self, step:Step):
        pass
    
    def after_step(self, step:Step):
        pass
    
    def before_training(self):
        pass
    
    def after_training(self):
        pass
    
    def on_error(self, step:Step, exception:Exception):
        pass


class Step(object):
    def __init__(self, state):
        self.frame = 0
        self.state = state
        self.action = None
        self.reward = 0.0
        self.done = False
        self.info = None

    def update(self, action, next_state, reward, done, info):
        self.frame += 1
        self.action = action
        self.state = next_state
        self.reward = reward
        self.done = done
        self.info = info

# class Epoch(object):
#     def __init__(self, epoch:int, env):
#         self.epoch = epoch
#         self.env = env

# class Episode(object):
#     def __init__(self, episode:int, render:bool=False):
#         self.episode = episode
#         self.render = render
#         self.total_reward = 0


class Plugin(object):
    def before_epoch(self, regimen:Regimen, epoch:int):
        pass
    
    def after_epoch(self, regimen:Regimen, epoch:int):
        pass

    def before_episode(self, regimen:Regimen, episode:int):
        pass
    
    def after_episode(self, regimen:Regimen, episode:int):
        pass
    
    def before_step(self, regimen:Regimen, step:Step):
        pass
    
    def after_step(self, regimen:Regimen, step:Step):
        pass
    
    def before_training(self, regimen:Regimen):
        pass
    
    def after_training(self, regimen:Regimen):
        pass
    
    def on_error(self, regimen:Regimen, step:Step, exception:Exception):
        pass