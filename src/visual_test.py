from sklearn import tree
import pandas as pd
import cart
import igraph
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import matplotlib.pyplot as plt

training_data = pd.read_csv(
    '../dataset/agaricus-lepiota - training.csv', sep=',')
attribute_value_data = pd.read_csv(
    '../dataset/all-attr-values.csv', sep=',')
training_data = training_data.drop(columns=['id'])

g_root = cart.create_decision_tree(
    dataset=training_data,
    classifier='class',
    attributes=training_data.columns,
    examples=attribute_value_data,
    criteria=cart.iGainType.entropy)

level = dict()
level[g_root] = 0
q = [g_root]
seen = []
depth = 0
edges = []
ids = dict()
index = 0
ids[g_root] = index

while q:
    curr = q.pop(0)
    for branch_val in curr.children:
        child = curr.children[branch_val]
        if child not in ids:
            ids[child] = index
            index += 1
        edges.append((ids[curr], ids[child]))
        if child not in seen:
            level[child] = level[curr] + 1
            seen.append(child)
            q.append(child)

depth = max(level.values())
print(f'depth={depth}')

# nr_vertices = len(seen)
# v_label = list(map(str, range(nr_vertices)))
fig, ax = plt.subplots()
G = Graph(edges)  # 2 stands for children number
layout = G.layout(layout='rt', root=[2])
igraph.plot(G, target=ax, layout=layout)
fig.show()
