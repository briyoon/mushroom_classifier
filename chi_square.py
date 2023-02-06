from collections import defaultdict
import scipy


def chi_square_attr_val(attribute, classification):
    attr_map = defaultdict(lambda: defaultdict(int))
    for val, cls in zip(attribute, classification):
        attr_map[val][cls] += 1
    chi_val = 0
    for v in attr_map:
        total = attr_map[v]['e'] + attr_map[v]['p']
        expected = 0.5
        var_e = ((attr_map[v]['e'] / total) - expected) ** 2
        var_p = ((attr_map[v]['p'] / total) - expected) ** 2
        chi_e = (var_e / expected)
        chi_p = (var_p / expected)
        chi_val += (chi_p + chi_e)

    return chi_val


def chi_valid_attr(chi_score, alpha, deg_freedom):
    needed = scipy.stats.chi2.ppf(1 - alpha, df=deg_freedom)
    return chi_score >= needed
