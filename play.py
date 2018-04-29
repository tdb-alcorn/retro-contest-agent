import numpy as np
import sys
import tty
from contextlib import closing
import retro

tty.setcbreak(sys.stdin)

game = "SonicTheHedgehog-Genesis"
level = "GreenHillZone.Act1"

keymap = {
    "q": 0,
    "w": 1,
    "e": 2,
    "r": 3,
    "t": 4,
    "y": 5,
    "u": 6,
    "i": 7,
    "o": 8,
    "p": 9,
    "[": 10,
    "]": 11,
}

def generate_action(n):
    action = np.zeros((12,))
    action[n] = 1
    return action

with closing(retro.make(game=game, state=level)) as env:
    env.reset()
    env.render()
    done = False
    while not done:
        valid_action = False
        while not valid_action:
            c = sys.stdin.read(1)
            if c in keymap:
                valid_action = True
        n = keymap[c]
        action = generate_action(n)
        _, _, done, _ = env.step(action)
        env.render()
