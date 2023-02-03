class AttrNode:
    def __init__(self, name: str, gini: float, is_leaf=False):
        self.name = name
        self.gini = gini
        self.children = {}
        self.is_leaf = is_leaf

    def __str__(self) -> str:
        return f"attribute=[name:{self.name}, gini:{self.gini}], is_leaf={self.is_leaf}"




