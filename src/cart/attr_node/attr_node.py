import pandas as pd

class AttrNode:
    def __init__(self, attr_name: pd.DataFrame, value: str, info_gain: float, is_leaf: bool=False) -> None:
        self.attr_name: str = attr_name
        self.value: str = value
        self.info_gain: float = info_gain
        self.children: list[str, AttrNode] = dict()
        self.is_leaf: bool = is_leaf

    def __str__(self) -> str:
        if self.is_leaf:
            return str(self.attr_name)

        return f"{self.attr_name}\n{self.info_gain:.4f}"

def print_tree() -> None:
    pass