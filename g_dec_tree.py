from AttrNode import AttrNode
import pandas as pd


def gini_split(data_set, attr, classifier):
    n = len(data_set.index)
    attr_dict = {}
    split_value = 0
    for datum in data_set.loc[:, attr]:
        if datum not in attr_dict:
            attr_dict[datum] = 1
        else:
            attr_dict[datum] += 1
    for attr_value in attr_dict:
        # proportion
        weight = attr_dict[attr_value] / n
        # choose all examples with this attr value
        subset = data_set.loc[data_set[attr] == attr_value]
        gini_attr_v = gini_index(subset, classifier)
        split_value += (weight * gini_attr_v)

    return split_value


def gini_index(data_set, classifier):
    n = len(data_set.index)
    prob = 0
    class_dict = {}
    for datum in data_set.loc[:, classifier]:
        if datum not in class_dict:
            class_dict[datum] = 1
        else:
            class_dict[datum] += 1
    for key in class_dict:
        prob += ((class_dict[key] / n) ** 2)
    return 1 - prob


def gini_attr_split_values(data_set, classifier, attributes):
    gini_split_val = {}
    for col in attributes:
        if col == classifier:
            continue
        gini_split_val[col] = gini_split(data_set=data_set, attr=col, classifier=classifier)
    return gini_split_val


def majority_classification(data_set, classifier):
    classification_count = {}
    for datum in data_set.loc[:, classifier]:
        if datum not in classification_count:
            classification_count[datum] = 1
        else:
            classification_count[datum] += 1
    majority = max(classification_count.values())
    for key in classification_count.keys():
        if classification_count[key] == majority:
            return key
    return -1


def best_attr(gini_split_values):
    argmin = min(gini_split_values.values())
    for k in gini_split_values.keys():
        if gini_split_values[k] == argmin:
            return k
    print('none found')
    return ''


def homogeneous(data_set, classifier):
    class_set = {val for val in data_set.loc[:, classifier]}
    return len(class_set) == 1


def g_ind_decision_tree(data_set, classifier, attributes, examples):
    if homogeneous(data_set, classifier) or len(attributes) == 1:
        cls = majority_classification(data_set, classifier)
        return AttrNode(name=cls, gini=0, is_leaf=True)

    attr_split_values = gini_attr_split_values(data_set, classifier, attributes)
    best_classifier = best_attr(attr_split_values)
    values = {val for val in examples.loc[:, best_classifier]}
    root = AttrNode(name=best_classifier, gini=attr_split_values[best_classifier])

    for value in values:
        value_subset = data_set.loc[data_set[best_classifier] == value]
        if len(value_subset.index) == 0:
            maj_class = majority_classification(data_set, classifier)
            child = AttrNode(name=maj_class, gini=0, is_leaf=True)
            root.children[value] = child
        else:
            root.children[value] = g_ind_decision_tree(value_subset,
                                                       classifier,
                                                       [atr for atr in attributes if atr != best_classifier],
                                                       examples)
    return root


def classify(root: AttrNode, example: {}):
    while not root.is_leaf:
        ex_value = example[root.name]
        root = root.children[ex_value]
    return root.name
