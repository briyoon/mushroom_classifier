from .igaintype import iGainType
from math import log2

def info_gain(data_set, classifier, gain_type):
    n = len(data_set.index)

    if (gain_type == iGainType.misclass):
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
        if (gain_type == iGainType.misclass):
            prob.append(class_dict[key] / n)
        elif (gain_type == iGainType.entropy):
            p = (class_dict[key] / n)
            prob += (p * log2(p))
        elif (gain_type == iGainType.gini):
            prob += ((class_dict[key] / n) ** 2)

    if (gain_type == iGainType.misclass):
        return 1 - max(prob)
    elif (gain_type == iGainType.entropy):
        return -1 * prob
    else:
        return 1 - prob