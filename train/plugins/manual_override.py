import time
from collections import deque
from ..play import KeyboardController
from ..plugin import Plugin
from ..regimen import Regimen, Step


class ManualOverride(Plugin):
    def __init__(self):
        # Framerate running average over the last 10 frames
        self.teaching = False
        self.controller = None

    def before_epoch(self, regimen:Regimen, epoch:int):
        self.controller = KeyboardController(regimen.env.action_space.shape)

    def before_step(self, regimen:Regimen, step:Step):
        # print('manual override', self.teaching)
        if self.teaching:
            action = self.controller.read_action()
            step.action = action
            if self.controller.done:
                self.teaching = False
                self.controller.done = False
    
    def on_error(self, regimen:Regimen, step:Step, exception:Exception) -> bool:
        if isinstance(exception, KeyboardInterrupt):
            if not self.teaching:
                self.teaching = True
                return False
        return True