from abc import ABC
from functools import cached_property

class Identifiable:
    _COUNTERS: dict[str, int] = dict()

    def __init__(self):
        clazz = type(self).__name__
        id = self._get_id(clazz)
        self._uid = f'{clazz}_{id}'

    @cached_property
    def uid(self) -> str:
        return self._uid

    @staticmethod
    def _get_id(clazz: str) -> str:
        instances_count = Identifiable._COUNTERS.get(clazz, 1)
        Identifiable._COUNTERS[clazz] = instances_count + 1
        return instances_count