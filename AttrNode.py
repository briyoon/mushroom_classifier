class AttrNode:
    def __init__(self, name: str, info_gain: float, is_leaf=False):
        self.name = name
        self.info_gain = info_gain
        self.children = {}
        self.is_leaf = is_leaf

    def __str__(self) -> str:
        return f"attribute=[name:{self.name}, gini:{self.info_gain}], is_leaf={self.is_leaf}"
