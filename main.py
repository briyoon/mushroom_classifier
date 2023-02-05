import pandas as pd

from decision_tree import *

df = pd.read_csv("dataset/agaricus-lepiota - training.csv")
x = df[df.columns[2:]]
y = df[df.columns[1]]
entropy = information_gain(x, y, iGainType.entropy)
me = information_gain(x, y, iGainType.missclass)

for k, v in entropy.items():
    print(f"\'{k}\': {v}")

print("\n\n")

for k, v in me.items():
    print(f"\'{k}\': {v}")
