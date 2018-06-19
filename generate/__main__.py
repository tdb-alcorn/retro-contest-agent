import sys
import argparse
from datetime import datetime
from generate import object_types, read_objects, write_objects
import os


def pluralize(s:str):
    return s + 's'

parser = argparse.ArgumentParser(
    description="Generate a new agent, training regimen, objective or environment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('object_type', choices=object_types, default=argparse.SUPPRESS, help='type of object to create')
parser.add_argument('name', type=str, default=argparse.SUPPRESS, help='name of the new object')

args = parser.parse_args()

objects = read_objects()

if args.name in objects[args.object_type]:
    print("An {} called {} already exists.".format(args.object_type, args.name), file=sys.stderr)
    sys.exit(1)

object_data = {
    # TODO: Use UTC?
    'created_at': datetime.now().isoformat(),
    'path': os.path.join(os.path.curdir, pluralize(args.object_type), args.name),
    'main': args.name.title() + args.object_type.title(),
}

contents = '''
from q2 import *

class {}(object):
    pass
'''

with open(object_data['path'], 'w') as f:
    f.write(contents.format(object_data['main']))

objects[args.object_type][args.name] = object_data

write_objects(objects)