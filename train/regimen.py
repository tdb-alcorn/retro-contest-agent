import tensorflow as tf
from typing import Type, Union
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
        self.message = ""
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
            for plugin in self.plugins:
                plugin.before_step(frame)
            self.before_step(frame)

            reward, done = self.step(
                epoch,
                render=render,
            )
            total_reward += reward

            for plugin in self.plugins:
                plugin.after_step(frame, reward, done)
            self.after_step(frame, reward, done)


    def use(self, other:Regimen):
        self.plugins.append(other)

    def step(self, step:int, render:bool=False):
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
        print("\033[K", end='\r')
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