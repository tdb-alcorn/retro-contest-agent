import os
import yaml
from typing import Dict

tld = os.path.join(os.path.dirname(__file__), '..')
objects_file = os.path.join(tld, 'objects.yaml')


def read_objects() -> Dict:
    with open(objects_file) as f:
        objects = yaml.load(f.read())
    return objects

def write_objects(objects:Dict):
    with open(objects_file, 'w') as f:
        f.write(yaml.dump(objects))

object_types = [
    'agent',
    'environment',
    'objective',
    'regimen',
]
