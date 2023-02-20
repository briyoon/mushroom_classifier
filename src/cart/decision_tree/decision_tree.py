from math import log2, sqrt
from random import sample, choice, randrange
import sys

import numpy as np
import pandas as pd

import cart

def get_feature_subset(total_attrs):
    attr_list = list(total_attrs)
    attr_list = [attr for attr in attr_list if attr != 'class']
    return sample(attr_list,  int(sqrt(len(total_attrs))))


def split(dataset, attr, classifier, criteria: cart.iGainType = cart.iGainType.entropy):
    n = len(dataset.index)
    attr_dict = {}
    split_value = 0

    for datum in dataset.loc[:, attr]:
        if datum not in attr_dict:
            attr_dict[datum] = 1
        else:
            attr_dict[datum] += 1

    for attr_value in attr_dict:
        # proportion
        weight = attr_dict[attr_value] / n
        # choose all examples with this attr value
        subset = dataset.loc[dataset[attr] == attr_value]

        # impurity
        attr_v = cart.info_gain(subset, classifier, criteria)

        # info gain
        split_value += (weight * attr_v)

    return split_value


def split_values(dataset, classifier, attributes, subset_features, criteria: cart.iGainType = cart.iGainType.entropy):
    split_vals = {}

    for col in attributes:
        if col == classifier:
            continue

        split_vals[col] = split(
            dataset, col, classifier, criteria)

    return split_vals


def majority_classification(dataset, classifier):
    classification_count = {}
    for datum in dataset.loc[:, classifier]:
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


def homogeneous(dataset, classifier):
    class_set = {val for val in dataset.loc[:, classifier]}
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


def process_missing(data, column, missing_indicator='?'):
    replacement_value_indexes = np.where(data[column] == missing_indicator)

    if len(replacement_value_indexes) == 0:
        sys.exit('Unable to replace missing values, no other values available')

    missing_value_indexes = np.where(data[column] == missing_indicator)

    for index in missing_value_indexes:
        data.at[index, column] = choice(replacement_value_indexes)

def attr_value_missing(dataset, column, missing_indicator='?'):
    missing_indicator in dataset[column].unique()

def chi_pruned_attr(dataset, attributes, classifier, examples, alpha):
    pruned_attr = attributes

    for attr in dataset.columns:
        if attr == classifier:
            continue
        chi_stat = cart.chi_square_attr_val(
            dataset[attr], classification=dataset[classifier])
        deg = len(examples[attr]) - 1
        if not cart.chi_valid_attr(chi_score=chi_stat, alpha=alpha, deg_freedom=deg):
            dataset = dataset.loc[:, dataset.columns != attr]
            pruned_attr = [a for a in attributes if a != attr]

    return pruned_attr


def create_decision_tree(dataset, classifier, attributes, unique_values: dict(),
                         criteria: cart.iGainType = cart.iGainType.entropy,
                         subset_features=False, chi_pruning=False, alpha=0):
    if chi_pruning:
        attributes = chi_pruned_attr(dataset, attributes, classifier, unique_values, alpha)

    if homogeneous(dataset, classifier) or len(attributes) == 1:
        cls = majority_classification(dataset, classifier)
        return cart.AttrNode(cls, None, 0, True)

    attr_split_values = split_values(
        dataset, classifier, attributes, subset_features, criteria)
    best_classifier = best_attr(attr_split_values)

    # replace missing attribute values
    if attr_value_missing(dataset, best_classifier):
        process_missing(dataset, best_classifier)

    values = unique_values[best_classifier]
    root = cart.AttrNode(best_classifier, "root", attr_split_values[best_classifier])

    for value in values:
        value_subset = dataset.loc[dataset[best_classifier] == value]
        if len(value_subset.index) == 0:
            # no data records in subset, classification node
            # w/ value set to most common class at root node
            maj_class = majority_classification(dataset, classifier)
            child = cart.AttrNode(maj_class, value, 0, True)
            root.children[value] = child
        else:
            # sub trees for remaining attributes
            root.children[value] = create_decision_tree(value_subset,
                                                        classifier,
                                                        [atr for atr in attributes if atr !=
                                                         best_classifier],
                                                        unique_values,
                                                        criteria,
                                                        subset_features=subset_features,
                                                        chi_pruning=chi_pruning,
                                                        alpha=alpha)
            root.children[value].value = value
    return root


def get_replacement_attr_val(dataset: pd.DataFrame, attr_name, missing_indicator='?'):
    present = dataset.loc[dataset[attr_name] != missing_indicator]
    return present[attr_name].values[randrange(len(present.index))]


def classify(root: cart.AttrNode, example: dict, dataset: pd.DataFrame, missing_indicator='?'):
    while not root.is_leaf:
        ex_value = example[root.attr_name]
        if ex_value == missing_indicator:
            ex_value = get_replacement_attr_val(
                dataset, root.name, missing_indicator)
        root = root.children[ex_value]
    return root.attr_name
