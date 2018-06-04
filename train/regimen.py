import tensorflow as tf
from typing import Type, Union, NamedTuple
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
        self.teaching = False
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
            for plugin in self.plugins:
                plugin.before_epoch(epoch)
            self.before_epoch(epoch)

            self.run_epoch(
                epoch,
                episodes_per_epoch,
                render=render,
                bk2dir=bk2dir,
                out_filename=out_filename
            )

            for plugin in self.plugins:
                plugin.after_epoch(epoch)
            self.after_epoch(epoch)
        
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
                for episode in range(episodes):
                    for plugin in self.plugins:
                        plugin.before_episode(episode)
                    self.before_episode(episode)

                    reward = self.run_episode(episode, render=render)
                    self.rewards.append((epoch, self.game, self.state, episode, reward))

                    for plugin in self.plugins:
                        plugin.after_episode(episode)
                    self.after_episode(episode)
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
        done = False
        state = self.env.reset()
        frame = 0
        total_reward = 0
        while not done:
            # Clear log message
            self.message = list()

            for plugin in self.plugins:
                plugin.before_step(frame)
            self.before_step(frame)

            next_state, reward, done = self.step(
                epoch,
                render=render,
            )
            total_reward += reward
            # Add basic message prefix
            self.message = [
                "Frame: {:d}".format(self.frame),
                "Reward: {:.2f}".format(reward),
                "Total: {:.2f}".format(total_reward),
            ] + self.message
            print("\033[K", end='\r')
            print('\t'.join(self.message), end='\r')

            for plugin in self.plugins:
                plugin.after_step(frame, reward, done)
            self.after_step(frame, reward, done)

            state = next_state
            frame += 1


    def use(self, plugin:Plugin):
        self.plugins.append(plugin)

    def step(self, frame:int, render:bool=False):
        if self.teaching:
            action = controller.read_action()
            if controller.done:
                teaching = False
                controller.done = False
                continue
        else:
            # TODO Print frame and reward here (overwrite)
        self.action = agent.act(sess, state, True)
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        reward_buffer.append(reward)
        total_reward += reward
        agent.step(sess, state, action, reward, next_state, done)
        state = next_state
        frame_times.append(time.time())
        fps = len(frame_times)/(frame_times[-1] - frame_times[0])
        print("Frame: {:d}\tReward: {:.2f}\tTotal: {:.2f}\tFramerate: {:.2f}/sec".format(frame, reward, total_reward, fps), end='\r')
        frame += 1


    def before_epoch(self, epoch:int):
        pass
    
    def after_epoch(self, epoch:int):
        pass

    def before_episode(self, episode:int):
        pass
    
    def after_episode(self, episode:int):
        pass
    
    def before_step(self, frame:int):
        pass
    
    def after_step(self, frame:int, reward:float, done:bool):
        pass
    
    def before_training(self):
        pass
    
    def after_training(self):
        pass


class Step(object):
    def __init__(self, state):
        self.frame = 0
        self.state = state
        self.action = None
        self.reward = 0.0
        self.done = False
        self.info = info if info is not None else dict()

    def update(self, action, next_state, reward, done, info):
        self.frame += 1
        self.action = action
        self.state = next_state
        self.reward = reward
        self.done = done
        self.info = info

class Epoch(object):
    def __init__(self, env):
        self.env = env


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