from typing import NamedTuple, Callable, List


Maker = NamedTuple('Maker', [
    ('make', Callable),
    ('states', List[str]),
])