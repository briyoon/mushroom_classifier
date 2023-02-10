from AttrNode import AttrNode
import pandas as pd
from math import log2, sqrt
from random import sample
import numpy as np

from chi_square import chi_square_attr_val, chi_valid_attr
from dec_tree_test import iGainType


def get_feature_set_size(total_attrs, subset=False):
    if not subset:
        return len(total_attrs)
    return int(sqrt(len(total_attrs)))


def info_gain(data_set, classifier, iGainType):
    n = len(data_set.index)

    if (iGainType == iGainType.misclass):
        prob = []
    else:
        prob = 0
    class_dict = {}

    for datum in data_set.loc[:, classifier]:
        if datum not in class_dict:
            class_dict[datum] = 1
        else:
            class_dict[datum] += 1
    for key in class_dict:
        if (iGainType == iGainType.misclass):
            prob.append(class_dict[key] / n)
        elif (iGainType == iGainType.entropy):
            p = (class_dict[key] / n)
            prob += (p * log2(p))
        elif (iGainType == iGainType.gini):
            prob += ((class_dict[key] / n) ** 2)

    if (iGainType == iGainType.misclass):
        return 1 - max(prob)
    elif (iGainType == iGainType.entropy):
        return -1 * prob
    else:
        return 1 - prob


def split(data_set, attr, classifier, criteria: iGainType = iGainType.entropy):
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

        # impurity
        attr_v = info_gain(subset, classifier, criteria)

        # info gain
        split_value += (weight * attr_v)

    return split_value


def split_values(data_set, classifier, attributes, subsetFeatures, criteria: iGainType = iGainType.entropy):
    split_vals = {}

    # handling feature subsets for random forests
    if subsetFeatures:
        attr_list = list(attributes)
        attr_list = [attr for attr in attr_list if attr != 'class']
        attributes = sample(attr_list, get_feature_set_size(attr_list))

    for col in attributes:
        if col == classifier:
            continue

        split_vals[col] = split(
            data_set, col, classifier, criteria)

    return split_vals


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


def best_attr(attr_split_values):
    argmin = min(attr_split_values.values())
    for k in attr_split_values.keys():
        if attr_split_values[k] == argmin:
            return k
    print('none found')
    return ''


def homogeneous(data_set, classifier):
    class_set = {val for val in data_set.loc[:, classifier]}
    return len(class_set) == 1


def get_bagged_df(origin_df):
    # Generate a list of random indices, sample with replacement
    indices = np.random.choice(
        origin_df.index, size=origin_df.shape[0], replace=True)
    # Create a new dataframe using the selected indices
    bagged_df = origin_df.loc[indices, :]
    return bagged_df


def most_freq_pred(pred_list):
    pred_count = {}

    for p in pred_list:
        if p not in pred_count:
            pred_count[p] = 1
        else:
            pred_count[p] += 1

    return max(pred_count, key=pred_count.get)


def create_decision_tree(data_set, classifier, attributes, examples, criteria: iGainType = iGainType.entropy, subsetFeatures=False):
    if homogeneous(data_set, classifier) or len(attributes) == 1:
        cls = majority_classification(data_set, classifier)
        return AttrNode(cls, 0, True)

    attr_split_values = split_values(
        data_set, classifier, attributes, subsetFeatures, criteria)
    best_classifier = best_attr(attr_split_values)
    values = {val for val in examples.loc[:, best_classifier]}
    root = AttrNode(best_classifier, attr_split_values[best_classifier])

    for value in values:
        value_subset = data_set.loc[data_set[best_classifier] == value]
        if len(value_subset.index) == 0:
            # no data records in subset, classification node
            # w/ value set to most common class at root node
            maj_class = majority_classification(data_set, classifier)
            child = AttrNode(maj_class, 0, True)
            root.children[value] = child
        else:
            # sub trees for remaining attributes
            root.children[value] = create_decision_tree(value_subset,
                                                        classifier,
                                                        [atr for atr in attributes if atr !=
                                                         best_classifier],
                                                        examples,
                                                        subsetFeatures,
                                                        criteria)
    return root


def create_decision_tree_chi(data_set, classifier, attributes, examples, alpha,
                             criteria: iGainType = iGainType.entropy, subsetFeatures=False):
    for attr in data_set.columns:
        if attr == classifier:
            continue
        chi_stat = chi_square_attr_val(
            data_set[attr], classification=data_set[classifier])
        deg = len(examples[attr].unique()) - 1
        if not chi_valid_attr(chi_score=chi_stat, alpha=alpha, deg_freedom=deg):
            data_set = data_set.loc[:, data_set.columns != attr]
            attributes = [a for a in attributes if a != attr]

    if homogeneous(data_set, classifier) or len(attributes) == 1:
        cls = majority_classification(data_set, classifier)
        return AttrNode(name=cls, info_gain=0, is_leaf=True)

    attr_split_values = split_values(
        data_set, classifier, attributes, subsetFeatures, criteria)
    best_classifier = best_attr(attr_split_values)
    values = {val for val in examples.loc[:, best_classifier]}
    root = AttrNode(best_classifier, attr_split_values[best_classifier])

    for value in values:
        value_subset = data_set.loc[data_set[best_classifier] == value]
        if len(value_subset.index) == 0:
            # no data records in subset, classification node
            # w/ value set to most common class at root node
            maj_class = majority_classification(data_set, classifier)
            child = AttrNode(maj_class, 0, True)
            root.children[value] = child
        else:
            # sub trees for remaining attributes
            root.children[value] = create_decision_tree(value_subset,
                                                        classifier,
                                                        [atr for atr in attributes if atr !=
                                                         best_classifier],
                                                        examples,
                                                        criteria,
                                                        subsetFeatures)

    return root


def classify(root: AttrNode, example: dict):
    while not root.is_leaf:
        ex_value = example[root.name]
        root = root.children[ex_value]
    return root.name
