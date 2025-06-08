from enum import Enum


class CloudType(Enum):
    RUNPOD = 'RUNPOD'
    LINUX = 'LINUX'
    REST = 'REST'
    def __str__(self):
        return self.value
