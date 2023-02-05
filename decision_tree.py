import math
from enum import Enum
from collections import defaultdict, Counter
import AttrNode

import numpy as np


class iGainType(Enum):
    entropy = 0
    gini = 1
    missclass = 2


def information_gain(attributes, classification, param: iGainType = iGainType.entropy):
    retval = dict()

    freq_map = {}
    for attr in attributes:
        attr_map = defaultdict(lambda: defaultdict(int))
        for key, class_ in zip(attributes[attr], classification):
            attr_map[key][class_] += 1
        freq_map[attr] = attr_map

    if param == iGainType.entropy:
        # parent entropy = -(e/total)log_2(e/total) - (p/total)log_2(p/total)
        e_parent = sum(1 if val == 'e' else 0 for val in classification)
        p_parent = sum(1 if val == 'p' else 0 for val in classification)
        total_parent = e_parent + p_parent
        parent_entropy = (-((e_parent / total_parent) * np.log2(e_parent / total_parent))) + \
                         (-((p_parent / total_parent) *
                          np.log2(p_parent / total_parent)))

        for attr in freq_map:
            child_entropy = {}
            for val in freq_map[attr]:
                # attr entropy = parent entropy - average of children entropy
                e_child = freq_map[attr][val]['e']
                p_child = freq_map[attr][val]['p']
                total_child = e_child + p_child

                # if all one class, 0 entropy
                if e_child == 0 or p_child == 0:
                    child_entropy[val] = 0
                else:
                    child_entropy[val] = (-((e_child / total_child) * np.log2(e_child / total_child))) + \
                                         (-((p_child / total_child) *
                                          np.log2(p_child / total_child)))
            retval[attr] = parent_entropy - \
                (sum(child_entropy.values()) / len(child_entropy))

    elif param == iGainType.gini:
        pass
    elif param == iGainType.missclass:
        for attr in freq_map:
            attr_imp = []
            root_imp = 0
            root_pos = 0
            root_neg = 0

            for val in freq_map[attr]:
                e_child = freq_map[attr][val]['e']
                p_child = freq_map[attr][val]['p']
                total_child = e_child + p_child
                child_impurity = 1 - \
                    (max(e_child/total_child, p_child/total_child))
                root_pos += e_child
                root_neg += p_child
                val_imp = {"occurs": e_child + p_child,
                           "impurity": child_impurity}
                attr_imp.append(val_imp)

            total_occurs = root_neg + root_pos
            root_imp = 1 - \
                (max(root_pos/total_occurs, root_neg/total_occurs))

            infoGain = 0
            for imp in attr_imp:
                infoGain += (imp["occurs"] / total_occurs) * imp["impurity"]

            infoGain *= root_imp
            retval[attr] = infoGain

    return dict(sorted(retval.items(), key=lambda item: item[1])[::-1])


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("dataset/agaricus-lepiota - training.csv")
    x = df[df.columns[2:]]
    y = df[df.columns[1]]

    entropy = information_gain(x, y)

    for k, v in entropy.items():
        print(f"\'{k}\': {v}")
