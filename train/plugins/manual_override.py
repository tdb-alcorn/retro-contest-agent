import time
from collections import deque
from ..regimen import Plugin, Regimen, Step


class ManualOverride(Plugin):
    def __init__(self, , progress_threshold:float=0):
        # Framerate running average over the last 10 frames
        self.teaching = False
        self.controller = None
        controller = KeyboardController()
        self.num_frames_no_progress = num_frames_no_progress
        self.progress_threshold = progress_threshold
        self.reward_buffer = deque(list(), num_frames_no_progress)

    def after_step(self, regimen:Regimen, step:Step):
        self.reward_buffer.append(step.reward)
        if step.frame >= self.num_frames_no_progress and sum(self.reward_buffer) < self.progress_threshold:
            step.done = True