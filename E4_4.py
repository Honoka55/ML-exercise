import pandas as pd
import numpy as np
from E4_3 import decision_tree, descretize, np_mode


def calc_gini_impurity(labels):
    _, counts = np.unique(labels, return_counts=True)
    label_probs = counts / len(labels)
    gini_impurity = 1 - sum(p ** 2 for p in label_probs)
    return gini_impurity


def calc_gini_index(feature, labels):
    values, counts = np.unique(feature, return_counts=True)
    feat_probs = counts / len(feature)
    gini_index = sum(r * calc_gini_impurity(labels[feature == v]) for r, v in zip(feat_probs, values))
    return gini_index


def get_optimal_feat_gini(featuresT, labels):
    gini_indices = [calc_gini_index(featuresT[i], labels) for i in range(len(featuresT))]
    # gini_indices[0] += 0.000001
    return np.argmin(gini_indices)


class DecisionTree:
    def __init__(self, criterion_func=get_optimal_feat_gini):
        self.root = None
        self.crit_func = criterion_func
        self.feat_values = {}
        self.float_feats = []

    def fit(self, features, labels, feature_values={}, train=True):
        self.feat_values = feature_values
        self.float_feats = features.select_dtypes(float).columns
        if train:
            self.root = decision_tree(features, labels, features.columns, self.crit_func, self.feat_values)

    def _check_fit(self):
        if self.root is None:
            raise ValueError('当前决策树为空，请先用fit方法训练')

    def predict(self, features):
        self._check_fit()
        return self._predicts(self.root, features)

    def _predicts(self, node, features):
        labels_pred = []
        for i in range(len(features)):
            feature = features.iloc[i]
            labels_pred.append(self._predict(node, feature))
        return labels_pred

    def _predict(self, node, feature):
        if isinstance(node, dict):
            feature_name = next(iter(node))
            feature_value = feature[feature_name]
            if feature_name in self.float_feats:
                for condition in node[feature_name]:
                    classifier = eval('lambda x: x' + condition)
                    if classifier(feature_value):
                        return self._predict(node[feature_name][condition], feature)
            else:
                return self._predict(node[feature_name][feature_value], feature)
        else:
            return node

    def evaluate(self, features, labels):
        self._check_fit()
        return self._evaluate(self.root, features, labels)

    def _evaluate(self, node, features, labels):
        labels_pred = self._predicts(node, features)
        accuracy = sum(labels_pred[i] == labels.iloc[i] for i in range(len(labels_pred))) / len(labels_pred)
        return accuracy

    def print(self):
        self._check_fit()
        print(self.root)

    def prune(self, features_train, features_test, labels_train, labels_test, feature_values={}, prune='none'):
        feature_names = features_train.columns
        if prune.lower() in ['false', 'none', '无', '不剪枝']:
            self.fit(features_train, labels_train, feature_values)
        elif prune.lower() in ['pre', 'preprune', 'pre-prune', 'pre-pruning', '预剪枝']:
            self.fit(features_train, labels_train, feature_values, train=False)
            self.root = self._preprune(features_train, features_test, labels_train, labels_test, feature_names)
        elif prune.lower() in ['post', 'postprune', 'post-prune', 'post-pruning', '后剪枝']:
            self.fit(features_train, labels_train, feature_values)
            self.root = self._postprune(features_train, features_test, labels_train, labels_test, feature_names)
        else:
            raise ValueError('无效的剪枝方法')

    def _preprune(self, features_train, features_test, labels_train, labels_test, feature_names):
        features_train_np = features_train.to_numpy()
        labels_train_np = labels_train.to_numpy()
        labels_test_np = labels_test.to_numpy()

        featuresT = []
        for i in range(len(feature_names)):
            feature = features_train_np[:, i]
            if feature_names[i] in self.float_feats:
                feature = feature.astype(float)
                if len(np.unique(feature)) > max(len(feature) / 2, 1):
                    feature = descretize(features_train_np[:, i], labels_train_np, self.crit_func)
            featuresT.append(feature)

        if len(np.unique(labels_train_np)) == 1:
            node = labels_train_np[0]
            return node

        if len(feature_names) == 0 or all([len(np.unique(featuresT[i])) == 1 for i in range(len(featuresT))]):
            node = np_mode(labels_train_np)
            return node

        optimal_feature_index = self.crit_func(featuresT, labels_train_np)
        optimal_feature_name = feature_names[optimal_feature_index]
        if optimal_feature_name in self.float_feats:
            optimal_feature_values = np.unique(featuresT[optimal_feature_index])
        else:
            optimal_feature_values = self.feat_values[optimal_feature_name]

        before_node = np_mode(labels_train_np)
        before_accuracy = sum(labels_test_np == before_node) / len(labels_test_np)

        after_correct = 0
        for value in optimal_feature_values:
            labels_test_subset = labels_test_np[features_test[optimal_feature_name] == value]
            if len([f for f in featuresT[optimal_feature_index] if f == value]) == 0:
                child_labels = labels_train_np
            else:
                child_labels = labels_train[features_train[optimal_feature_name] == value]
                if optimal_feature_name in self.float_feats:
                    classifier = eval('lambda x: x' + value)
                    child_labels = labels_train[classifier(features_train[optimal_feature_name])]
                    labels_test_subset = labels_test_np[classifier(features_test[optimal_feature_name])]
            child_mode = np_mode(child_labels)
            after_correct += sum(labels_test_subset == child_mode)

        after_accuracy = after_correct / len(labels_test)
        # print(f'{optimal_feature_name}：划分前精度{before_accuracy}，划分后精度{after_accuracy}，'
        #       f'故{'剪' if after_accuracy <= before_accuracy else '不剪'}枝')
        if after_accuracy <= before_accuracy:
            return before_node

        node = {optimal_feature_name: {}}
        for value in optimal_feature_values:
            if len([f for f in featuresT[optimal_feature_index] if f == value]) == 0:
                child = np_mode(labels_train_np)
            else:
                child_features = features_train[features_train[optimal_feature_name] == value]
                child_labels = labels_train[features_train[optimal_feature_name] == value]
                if optimal_feature_name in self.float_feats:
                    child_feature_names = feature_names
                    classifier = eval('lambda x: x' + value)
                    child_features = features_train[classifier(features_train[optimal_feature_name])]
                    child_labels = labels_train[classifier(features_train[optimal_feature_name])]
                else:
                    child_feature_names = [f for f in feature_names if f != optimal_feature_name]
                    child_features = child_features.drop(columns=optimal_feature_name)
                child = self._preprune(child_features, features_test, child_labels, labels_test, child_feature_names)
            node[optimal_feature_name][value] = child

        return node

    def _postprune(self, features_train, features_test, labels_train, labels_test, feature_names):
        features_train_np = features_train.to_numpy()
        labels_train_np = labels_train.to_numpy()
        labels_test_np = labels_test.to_numpy()

        featuresT = []
        for i in range(len(feature_names)):
            feature = features_train_np[:, i]
            if feature_names[i] in self.float_feats:
                feature = feature.astype(float)
                if len(np.unique(feature)) > max(len(feature) / 2, 1):
                    feature = descretize(features_train_np[:, i], labels_train_np, self.crit_func)
            featuresT.append(feature)

        if len(np.unique(labels_train_np)) == 1:
            node = labels_train_np[0]
            return node

        if len(feature_names) == 0 or all([len(np.unique(featuresT[i])) == 1 for i in range(len(featuresT))]):
            node = np_mode(labels_train_np)
            return node

        optimal_feature_index = self.crit_func(featuresT, labels_train_np)
        optimal_feature_name = feature_names[optimal_feature_index]
        if optimal_feature_name in self.float_feats:
            optimal_feature_values = np.unique(featuresT[optimal_feature_index])
        else:
            optimal_feature_values = self.feat_values[optimal_feature_name]

        after_node = np_mode(labels_train_np)
        after_accuracy = sum(labels_test_np == after_node) / len(labels_test_np)

        node = {optimal_feature_name: {}}
        for value in optimal_feature_values:
            if len([f for f in featuresT[optimal_feature_index] if f == value]) == 0:
                child = np_mode(labels_train_np)
            else:
                child_features = features_train[features_train[optimal_feature_name] == value]
                child_labels = labels_train[features_train[optimal_feature_name] == value]
                if optimal_feature_name in self.float_feats:
                    child_feature_names = feature_names
                    classifier = eval('lambda x: x' + value)
                    child_features = features_train[classifier(features_train[optimal_feature_name])]
                    child_labels = labels_train[classifier(features_train[optimal_feature_name])]
                else:
                    child_feature_names = [f for f in feature_names if f != optimal_feature_name]
                    child_features = child_features.drop(columns=optimal_feature_name)
                child = self._postprune(child_features, features_test, child_labels, labels_test, child_feature_names)
            node[optimal_feature_name][value] = child

        before_accuracy = self._evaluate(node, features_test, labels_test)
        # print(f'{optimal_feature_name}：不剪枝精度{before_accuracy}，剪枝后精度{after_accuracy}，'
        #       f'故{'剪' if after_accuracy >= before_accuracy else '不剪'}枝')
        if after_accuracy >= before_accuracy:
            node = after_node

        return node


if __name__ == '__main__':
    df = pd.read_csv('data/2.0.csv')
    features = df.iloc[:, 1:-1]
    labels = df.iloc[:, -1].map({'是': '好瓜', '否': '坏瓜'})
    feat_values = {col: features[col].unique() for col in features.columns}

    train_list = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
    features_train = features.iloc[[i - 1 for i in train_list]]
    features_test = features.drop([i - 1 for i in train_list])
    labels_train = labels.iloc[[i - 1 for i in train_list]]
    labels_test = labels.drop([i - 1 for i in train_list])

    tree = DecisionTree(get_optimal_feat_gini)
    tree.fit(features_train, labels_train, feat_values)
    print('未剪枝决策树：')
    tree.print()
    print(f'测试集精度：{tree.evaluate(features_test, labels_test)}')
    tree.prune(features_train, features_test, labels_train, labels_test, feat_values, 'pre')
    print('预剪枝决策树：')
    tree.print()
    print(f'测试集精度：{tree.evaluate(features_test, labels_test)}')
    tree.prune(features_train, features_test, labels_train, labels_test, feat_values, 'post')
    print('后剪枝决策树：')
    tree.print()
    print(f'测试集精度：{tree.evaluate(features_test, labels_test)}')
