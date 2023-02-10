from dec_tree import *
from enum import Enum
from sklearn.model_selection import train_test_split
import csv


class iGainType(Enum):
    entropy = 0
    gini = 1
    misclass = 2


attribute_value_data = pd.read_csv(
    './dataset/all-attr-values.csv', sep=',')
training_data = pd.read_csv(
    './dataset/agaricus-lepiota - training.csv', sep=',')
test_data = pd.read_csv(
    'dataset/large_test.csv', sep=',')
training_data, validation_df = train_test_split(training_data, test_size=0.2)

training_data_ids = training_data['id'].values
training_data = training_data.drop(columns=['id'])

gini_bag = get_bagged_df(test_data)
gini_bag.drop(columns=['id'], inplace=True)
gini_root = create_decision_tree(
    training_data, 'class', gini_bag.columns, gini_bag, iGainType.gini)

entropy_bag = get_bagged_df(test_data)
entropy_bag.drop(columns=['id'], inplace=True)
entropy_root = create_decision_tree(
    training_data, 'class', entropy_bag.columns, entropy_bag, iGainType.entropy)

misclass_bag = get_bagged_df(test_data)
misclass_bag.drop(columns=['id'], inplace=True)
misclass_root = create_decision_tree(
    training_data, 'class', misclass_bag.columns, misclass_bag, iGainType.misclass)

validation_dict = validation_df.to_dict('records')
test_dict = test_data.to_dict('records')

total_correct = 0
total = len(validation_dict)
alpha_values = [0.95, 0.99]
# key = data record id, value = list of predictions from e/a model
validation_predictions = {}
test_predictions = {}

for ex in test_dict:
    ex_id = ex['id']
    gini_pred = classify(gini_root, ex)
    entropy_pred = classify(entropy_root, ex)
    misclass_pred = classify(misclass_root, ex)

    if ex_id not in test_predictions:
        test_predictions[ex_id] = []

    test_predictions[ex_id].append(gini_pred)
    test_predictions[ex_id].append(entropy_pred)
    test_predictions[ex_id].append(misclass_pred)


for prediction in validation_dict:
    actual_class = classify(misclass_root, prediction)
    expected_class = prediction['class']
    if actual_class == expected_class:
        total_correct += 1

accuracy = total_correct / total
print(f'\ncorrect={total_correct}, total={total}, accuracy={accuracy}')

for alpha in alpha_values:
    chi_root = create_decision_tree_chi(training_data, 'class', training_data.columns,
                                        examples=attribute_value_data, alpha=alpha, criteria=iGainType.gini)
    g_root = create_decision_tree(training_data, 'class', training_data.columns,
                                  examples=attribute_value_data, criteria=iGainType.gini)
    chi_total = 0
    g_total = 0
    total = len(validation_dict)

    for ex in validation_dict:
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


test_pred_mean = {}

for k, v in test_predictions.items():
    test_pred_mean[k] = most_freq_pred(v)

with open('test_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'class'])
    for k, v in test_pred_mean.items():
        writer.writerow([k, v])
