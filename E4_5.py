import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from E4_4 import DecisionTree, np_mode


def logistic_regression(x, y, C=1e5):
    logreg = LogisticRegression(C=C)
    logreg.fit(x, y)
    w = np.append(logreg.coef_, logreg.intercept_)
    return w


def make_classifier(coefs):
    def classifier(x):
        x = np.append(x, 1)
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
            return node

        coefficients = logistic_regression(features_np, labels_np)
        classifier, classifier_name = self._get_classifier(coefficients)

        classified_no_list = []
        for i in range(len(features)):
            f = features.iloc[i]
            f_np = f.to_numpy()
            if classifier(f_np):
                classified_no_list.append(i)

        classified_features = [labels[classified_no_list], labels.drop(classified_no_list)]
        classified_labels = [labels[classified_no_list], labels.drop(classified_no_list)]

        node = {classifier_name: {}}
        for i in range(2):
            if len(classified_features[i]) == 0:
                child = np_mode(classified_labels[i])
            else:
                child = self._train(classified_features[i], classified_labels[i])
            node[classifier_name]['是' if i else '否'] = child

        return node

    def _predict(self, node, feature):
        if isinstance(node, dict):
            feature_name = next(iter(node))
            classifier_str = feature_name.replace('×', '*').replace('≤', '<=')
            for col in self.column_names:
                classifier_str = classifier_str.replace(col, str(feature[col]))
            feature_value = eval(classifier_str)
            for condition in node[feature_name]:
                condition_value = True if condition == '是' else False
                if feature_value == condition_value:
                    return self._predict(node[feature_name][condition], feature)
        else:
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
