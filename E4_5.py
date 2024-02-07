import pandas as pd
import numpy as np
from E3_3 import sklearn_logistic_regression as logistic_regression
from E4_4 import DecisionTree, np_mode


def make_classifier(coefs):
    def classifier(x):
        np.append(x, 1)
        pred = np.dot(coefs, x)
        return 1 if pred > 0 else 0

    return classifier


def make_classifier_str(coefs, col_names):
    classifier_str = f'{coefs[0]:.3f}×{col_names[0]}'
    for i in range(1, len(col_names) - 1):
        classifier_str += f'{coefs[i]:+.3f}×{col_names[i]}'
    classifier_str += f'≤{-coefs[-1]:.3f}'
    return classifier_str


class LogisticRegressionDecisionTree(DecisionTree):
    def __init__(self):
        super().__init__(criterion_func=make_classifier)
        self.column_names = []

    def fit(self, features, labels, feature_values={}, train=True):
        self.column_names = features.columns
        self.root = self._train(features, labels)

    def _get_classifier(self, coefs):
        return make_classifier(coefs), make_classifier_str(coefs, self.column_names)

    def _train(self, features, labels):
        features_np = features.to_numpy()
        labels_np = labels.to_numpy()

        if len(np.unique(labels_np)) == 1:
            node = labels_np[0]
            return node

        if all([len(np.unique(features_np.T[i])) == 1 for i in range(len(features_np.T))]):
            node = np_mode(labels_np)
            print(f'返回情形2：叶结点{node}')
            return node

        coefficients = logistic_regression(features_np, labels_np).ravel()
        classifier, classifier_name = self._get_classifier(coefficients)
        node = {classifier_name: {}}

        classified_yes_list = []
        for i in range(len(features)):
            f = features.iloc[i]
            f_np = f.to_numpy()
            if classifier(f_np):
                classified_yes_list.append(i)

        classified_features = [labels.drop(classified_yes_list), labels[classified_yes_list]]
        classified_labels = [labels.drop(classified_yes_list), labels[classified_yes_list]]

        for i in range(2):
            if len(classified_features[i]) == 0:
                child = np_mode(classified_labels[i])
                print(f'返回情形3：叶结点{child}')
            else:
                child = self._train(classified_features[i], classified_labels[i])
            node[classifier_name]['是' if i else '否'] = child

        return node


if __name__ == '__main__':
    df = pd.read_csv('data/3.0.csv')
    features = df.iloc[:, 1:-1]
    labels = df.iloc[:, -1].map({'是': 1, '否': 0})
    features = pd.get_dummies(features, prefix_sep='=')
    column_names = []
    for name in features.columns:
        if '=' in name:
            column_names.append(f'𝕀({name})')
        else:
            column_names.append(name)
    features.columns = column_names

    tree = LogisticRegressionDecisionTree()
    tree.fit(features, labels)
    dict_stack = [tree.root]
    while dict_stack:
        node = dict_stack.pop()
        for k, v in node.items():
            if isinstance(v, dict):
                dict_stack.append(v)
            else:
                node[k] = '好瓜' if v == 1 else '坏瓜'
    tree.print()
