from sklearn.model_selection import train_test_split
import csv
import pandas as pd

import cart

attribute_value_data = pd.read_csv(
    'dataset/all-attr-values.csv', sep=',')
training_data = pd.read_csv(
    'dataset/agaricus-lepiota - training.csv', sep=',')
test_data = pd.read_csv(
    'dataset/large_test.csv', sep=',')
training_data, validation_df = train_test_split(training_data, test_size=0.2)

training_data_ids = training_data['id'].values
training_data = training_data.drop(columns=['id'])

gini_bag = cart.get_bagged_df(test_data)
# gini_bag.drop(columns=['id'], inplace=True)
gini_root = cart.create_decision_tree(
    training_data, 'class', gini_bag.columns, gini_bag, cart.iGainType.gini)

entropy_bag = cart.get_bagged_df(test_data)
# entropy_bag.drop(columns=['id'], inplace=True)
entropy_root = cart.create_decision_tree(
    training_data, 'class', entropy_bag.columns, entropy_bag, cart.iGainType.gini)

misclass_bag = cart.get_bagged_df(test_data)
# misclass_bag.drop(columns=['id'], inplace=True)
misclass_root = cart.create_decision_tree(
    training_data, 'class', misclass_bag.columns, misclass_bag, cart.iGainType.gini)

validation_dict = validation_df.to_dict('records')
test_dict = test_data.to_dict('records')

total_correct = 0
total = len(validation_dict)
alpha_values = [0.00, 0.05, 0.95, 0.99]
# key = data record id, value = list of predictions from e/a model
validation_predictions = {}
test_predictions = {}

id = 0
for ex in test_dict:
    ex_id = id
    id += 1
    gini_pred = cart.classify(gini_root, ex, training_data)
    entropy_pred = cart.classify(entropy_root, ex, training_data)
    misclass_pred = cart.classify(misclass_root, ex, training_data)

    if ex_id not in test_predictions:
        test_predictions[ex_id] = []

    test_predictions[ex_id].append(gini_pred)
    test_predictions[ex_id].append(entropy_pred)
    test_predictions[ex_id].append(misclass_pred)

for prediction in validation_dict:
    actual_class = cart.classify(misclass_root, prediction, training_data)
    expected_class = prediction['class']
    if actual_class == expected_class:
        total_correct += 1

accuracy = total_correct / total
print(f'\ncorrect={total_correct}, total={total}, accuracy={accuracy}')

# Random forest construction
for alpha in alpha_values:
    chi_root = cart.create_decision_tree(dataset=training_data,
                                         classifier='class',
                                         attributes=training_data.columns,
                                         unique_values=attribute_value_data,
                                         criteria=cart.iGainType.gini,
                                         chi_pruning=True,
                                         alpha=alpha)
    g_root = cart.create_decision_tree(training_data, 'class', training_data.columns,
                                       unique_values=attribute_value_data, criteria=cart.iGainType.gini)
    chi_total = 0
    g_total = 0
    total = len(validation_dict)

    for ex in validation_dict:
        chi_classification = cart.classify(chi_root, ex, training_data)
        g_classification = cart.classify(g_root, ex, training_data)
        expected_class = ex['class']
        if chi_classification == expected_class:
            chi_total += 1
        if g_classification == expected_class:
            g_total += 1

    chi_acc = chi_total / total
    g_acc = g_total / total
    print(f'tree: alpha={alpha}, chi_acc={chi_acc}, g_acc={g_acc}')

test_pred_mean = {}

for k, v in test_predictions.items():
    test_pred_mean[k] = cart.most_freq_pred(v)

with open('test_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'class'])
    for k, v in test_pred_mean.items():
        writer.writerow([k, v])
