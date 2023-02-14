import pandas as pd

class AttrNode:
    def __init__(self, name: pd.DataFrame, info_gain: float, is_leaf: bool=False) -> None:
        self.name: str = name
        self.info_gain: float = info_gain
        self.children: list[str, AttrNode] = dict()
        self.is_leaf: bool = is_leaf

    def __str__(self) -> str:
        return f"attribute=[name:{self.name}, gini:{self.info_gain}], is_leaf={self.is_leaf}]"

def print_tree() -> None:
    pass