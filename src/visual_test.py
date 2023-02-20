from sklearn import tree
import pandas as pd
import cart
import igraph
from igraph import Graph, EdgeSeq
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

training_data = pd.read_csv(
    'dataset/agaricus-lepiota - training.csv', sep=',')
# attribute_value_data = pd.read_csv(
#     'dataset/all-attr-values.csv', sep=',')
training_data = training_data.drop(columns=['id'])

# save unique values
unique_values = {}


g_root = cart.create_decision_tree(
    dataset=training_data,
    classifier='class',
    attributes=training_data.columns,
    unique_values=unique_values,
    criteria=cart.iGainType.entropy)

cart.create_visualization(g_root, "module_test.png")
