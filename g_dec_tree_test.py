from g_dec_tree import *

training_data = pd.read_csv(
    './dataset/agaricus-lepiota - training.csv', sep=',')

training_data = training_data.loc[:, training_data.columns != 'id']
test = training_data.loc[:, training_data.columns != 'id']
dictionary = test.to_dict('records')

root = g_ind_decision_tree(test, 'class', test.columns, test)
total_correct = 0
total = len(dictionary)

for ex in dictionary:
    actual_class = classify(root, ex)
    expected_class = ex['class']
    if actual_class == expected_class:
        total_correct += 1

accuracy = total_correct / total
print(f'correct={total_correct}, total={total}, accuracy={accuracy}')
