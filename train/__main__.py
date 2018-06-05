import argparse
import sys
import tensorflow as tf
import retro
from typing import List
from agents import all_agents
from objective import Passthrough
from train.regimen import Regimen
from train.plugins import DisplayFramerate, ManualOverride, NoProgressEarlyStopping
from train.utils import random_if_empty, ensure_directory_exists, random_choice
from train.regimens import all_regimens, Online


def err_not_found(item:str, available:List[str], item_type:str=''):
    show_item_type = len(item_type) > 0
    sys.stderr.write('{}{} not found. Available{}: {}.\n'.format(
        item_type.title() + ' ' if show_item_type else '',
        args.agent,
        ' ' + item_type + 's' if show_item_type else '',
        ', '.join(all_agents.keys())),
    )
    sys.exit(1)


parser = argparse.ArgumentParser(
    description="Train an agent against an environment.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('agent', type=str, default=argparse.SUPPRESS, help='name of the agent')
parser.add_argument('--regimen', type=str, dest='regimen', default='online', help='training regimen to use')
parser.add_argument('--epochs', type=int, dest='num_epochs', default=1, metavar='N', help='number of epochs to run')
parser.add_argument('--episodes', type=int, dest='episodes_per_epoch', default=1, metavar='N', help='number of episodes to run per epoch')
parser.add_argument('--environment', type=str, dest='environment', default='', help='name of the environment (empty string means a random one will be chosen for you)')
parser.add_argument('--state', type=str, dest='state', default='', help='name of the initial environment state (empty string means a random one will be chosen for you)')
parser.add_argument('--render', const=True, default=False, action='store_const', dest='render', help='enable rendering of training to video')
parser.add_argument('--bk2dir', type=str, dest='bk2dir', default=None, help='optional directory to store .bk2 gameplay files')
parser.add_argument('--output', type=str, dest='out_filename', default='', help='file prefix in which to save training losses and rewards')

args = parser.parse_args()

if args.agent in all_agents:
    agent_constructor = all_agents[args.agent]
else:
    err_not_found(args.agent, all_agents.keys(), 'agent')

if args.regimen in all_regimens:
    regimen_constructor = all_regimens[args.regimen]
else:
    err_not_found(args.regimen, all_regimens.keys(), 'regimen')

objective = Passthrough()

# Ensure that the bk2 save directory is present
if args.bk2dir is not None:
    ensure_directory_exists(args.bk2dir)

if args.out_filename[-4:] == '.csv':
    args.out_filename = args.out_filename[:-4]

valid_environment = False
while not valid_environment:
    try:
        all_environments = retro.list_games()
        environment = random_choice(all_environments) if args.environment == '' else args.environment
        all_states = retro.list_states(environment)
        state = random_choice(all_states) if args.state == '' else args.state
        # test environment
        retro.make(environment, state)
        valid_environment = True
    except FileNotFoundError:
        continue

regimen = regimen_constructor(agent_constructor, objective)
regimen.use(DisplayFramerate())
regimen.use(ManualOverride())
regimen.use(NoProgressEarlyStopping(500, progress_threshold=0))

# Run training
regimen.train(environment, state, args.num_epochs, args.episodes_per_epoch, render=args.render, bk2dir=args.bk2dir, out_filename=args.out_filename)
