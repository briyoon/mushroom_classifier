from g_dec_tree import *

attribute_value_data = pd.read_csv(
    './dataset/all-attr-values.csv', sep=',')
training_data = pd.read_csv(
    './dataset/agaricus-lepiota - training.csv', sep=',')
test_data = pd.read_csv(
    'dataset/large_test.csv', sep=',')

training_data = training_data.loc[:, training_data.columns != 'id']
train = test_data.loc[0:20_000]
# test = training_data.loc[1500:7000]
test_ex = test_data.to_dict('records')
print(f'testing on {len(test_ex)} ex, training on {len(train.index)}')

alpha_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]

for alpha in alpha_values:
    chi_root = g_dec_tree_chi(train, 'class', train.columns, examples=attribute_value_data, alpha=alpha)
    g_root = g_ind_decision_tree(train, 'class', train.columns, examples=attribute_value_data)
    chi_total = 0
    g_total = 0
    total = len(test_ex)

    for ex in test_ex:
        chi_classification = classify(chi_root, ex)
        g_classification = classify(g_root, ex)
        expected_class = ex['class']
        if chi_classification == expected_class:
            chi_total += 1
        if g_classification == expected_class:
            g_total += 1

    chi_acc = chi_total / total
    g_acc = g_total / total
    print(f'chi tree: alpha={alpha}, chi_acc={chi_acc}, g_acc={g_acc}')


