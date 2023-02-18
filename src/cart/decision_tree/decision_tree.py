from math import log2, sqrt
from random import sample
from random import randrange

import numpy as np
import pandas

import src.cart as cart


def get_feature_set_size(total_attrs, subset=False):
    if not subset:
        return len(total_attrs)
    return int(sqrt(len(total_attrs)))


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

    # handling feature subsets for random forests
    if subset_features:
        attr_list = list(attributes)
        attr_list = [attr for attr in attr_list if attr != 'class']
        attributes = sample(attr_list, get_feature_set_size(attr_list))

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
    present_values = data.loc[data[column] != missing_indicator]
    missing_values = data.loc[data[column] == missing_indicator]

    if len(present_values.index) == 0:
        print('Unable to replace missing values, no other values available')
        return

    for index in missing_values.index:
        present_index = randrange(len(present_values.index))
        replacement = present_values[column].values[present_index]
        data.at[index, column] = replacement


def attr_value_missing(dataset, column, missing_indicator='?'):
    missing_values = dataset.loc[dataset[column] == missing_indicator]
    return len(missing_values.index) > 0


def chi_pruned_attr(dataset, attributes, classifier, examples, alpha):
    pruned_attr = attributes

    for attr in dataset.columns:
        if attr == classifier:
            continue
        chi_stat = cart.chi_square_attr_val(
            dataset[attr], classification=dataset[classifier])
        deg = len(examples[attr].unique()) - 1
        if not cart.chi_valid_attr(chi_score=chi_stat, alpha=alpha, deg_freedom=deg):
            dataset = dataset.loc[:, dataset.columns != attr]
            pruned_attr = [a for a in attributes if a != attr]

    return pruned_attr


def create_decision_tree(dataset, classifier, attributes, examples,
                         criteria: cart.iGainType = cart.iGainType.entropy,
                         subset_features=False, chi_pruning=False, alpha=0):
    if chi_pruning:
        attributes = chi_pruned_attr(dataset, attributes, classifier, examples, alpha)

    if homogeneous(dataset, classifier) or len(attributes) == 1:
        cls = majority_classification(dataset, classifier)
        return cart.AttrNode(cls, 0, True)

    attr_split_values = split_values(
        dataset, classifier, attributes, subset_features, criteria)
    best_classifier = best_attr(attr_split_values)

    # replace missing attribute values
    if attr_value_missing(dataset, best_classifier):
        process_missing(dataset, best_classifier)

    values = {val for val in examples.loc[:, best_classifier]}
    root = cart.AttrNode(best_classifier, attr_split_values[best_classifier])

    for value in values:
        value_subset = dataset.loc[dataset[best_classifier] == value]
        if len(value_subset.index) == 0:
            # no data records in subset, classification node
            # w/ value set to most common class at root node
            maj_class = majority_classification(dataset, classifier)
            child = cart.AttrNode(maj_class, 0, True)
            root.children[value] = child
        else:
            # sub trees for remaining attributes
            root.children[value] = create_decision_tree(value_subset,
                                                        classifier,
                                                        [atr for atr in attributes if atr !=
                                                         best_classifier],
                                                        examples,
                                                        criteria,
                                                        subset_features=subset_features,
                                                        chi_pruning=chi_pruning,
                                                        alpha=alpha)
    return root


def get_replacement_attr_val(dataset: pandas.DataFrame, attr_name, missing_indicator='?'):
    present = dataset.loc[dataset[attr_name] != missing_indicator]
    return present[attr_name].values[randrange(len(present.index))]


def classify(root: cart.AttrNode, example: dict, dataset: pandas.DataFrame, missing_indicator='?'):
    while not root.is_leaf:
        ex_value = example[root.name]
        if ex_value == missing_indicator:
            ex_value = get_replacement_attr_val(dataset, root.name, missing_indicator)
        root = root.children[ex_value]
    return root.name
