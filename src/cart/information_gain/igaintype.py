from enum import Enum


class iGainType(Enum):
    entropy = 0
    gini = 1
    misclass = 2