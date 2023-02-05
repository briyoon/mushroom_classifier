class IGNode:
    def __init__(self, name, impurity=0):
        self.name = name
        self.impurity = 0
        self.children = []

    def addChild(self, child):
        self.children.append(child)

    def getChildren(self):
        return self.children

    def getName(self):
        return self.name

    # def getParent(self):
    #     return self.parent

    # def setParent(self, parent):
    #     self.parent = parent

    def __str__(self):
        return self.name
