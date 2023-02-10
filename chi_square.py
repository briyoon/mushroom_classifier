from collections import defaultdict
import scipy

def chi_square_attr_val(attribute, classification):
    attr_map = defaultdict(lambda: defaultdict(int))
    class_map = defaultdict(int)
    for val, cls in zip(attribute, classification):
        attr_map[val][cls] += 1
        class_map[cls] += 1
    prob_e = class_map['e'] / len(classification)
    prob_p = class_map['p'] / len(classification)
    chi_val = 0
    for v in attr_map:
        total = attr_map[v]['e'] + attr_map[v]['p']
        expected_e = total * prob_e
        expected_p = total * prob_p
        var_e = (attr_map[v]['e'] - expected_e) ** 2
        var_p = (attr_map[v]['p'] - expected_p) ** 2
        chi_e = (var_e / expected_e)
        chi_p = (var_p / expected_p)
        chi_val += (chi_p + chi_e)

    return chi_val


def chi_valid_attr(chi_score, alpha, deg_freedom):
    needed = scipy.stats.chi2.ppf(1 - alpha, df=deg_freedom)
    return chi_score >= needed
