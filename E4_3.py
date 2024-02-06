import pandas as pd
import numpy as np


def np_mode(array):
    values, counts = np.unique(array, return_counts=True)
    return values[np.argmax(counts)]


def calc_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    label_probs = counts / len(labels)
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in label_probs)
    return entropy


def calc_info_gain(feature, labels):
    values, counts = np.unique(feature, return_counts=True)
    feat_probs = counts / len(feature)
    info_gain = calc_entropy(labels) - sum(r * calc_entropy(labels[feature == v]) for r, v in zip(feat_probs, values))
    return info_gain


def descretize(feature, labels, optimal_func):
    sfeat = sorted(feature)
    midpoints = [(sfeat[i] + sfeat[i + 1]) / 2 for i in range(len(sfeat) - 1) if sfeat[i] != sfeat[i + 1]]
    dfeats = [np.where(feature > midpoint, 1, 0) for midpoint in midpoints]

    threshold = midpoints[optimal_func(dfeats, labels)]
    threshold = max([f for f in feature if f <= threshold])

    dfeat = np.where(feature > threshold, f'>{threshold}', f'<={threshold}')
    return dfeat


def get_optimal_feature(featuresT, labels):
    info_gains = [calc_info_gain(featuresT[i], labels) for i in range(len(featuresT))]
    return np.argmax(info_gains)


def decision_tree(features, labels, feature_names, criterion_func, feature_values={}):
    float_features = features.select_dtypes(float).columns
    features_np = features.to_numpy()
    labels_np = labels.to_numpy()

    featuresT = []
    for i in range(len(feature_names)):
        feature = features_np[:, i]
        if feature_names[i] in float_features:
            feature = feature.astype(float)
            if len(np.unique(feature)) > max(len(feature) / 2, 1):
                feature = descretize(features_np[:, i], labels_np, criterion_func)
        featuresT.append(feature)

    if len(np.unique(labels_np)) == 1:
        node = labels_np[0]
        # print(f'返回情形1：叶结点{node}')
        return node

    if len(feature_names) == 0 or all([len(np.unique(featuresT[i])) == 1 for i in range(len(featuresT))]):
        node = np_mode(labels_np)
        # print(f'返回情形2：叶结点{node}')
        return node

    optimal_feature_index = criterion_func(featuresT, labels_np)
    optimal_feature_name = feature_names[optimal_feature_index]
    if optimal_feature_name not in float_features and optimal_feature_name in feature_values:
        optimal_feature_values = feature_values[optimal_feature_name]
    else:
        optimal_feature_values = np.unique(featuresT[optimal_feature_index])
    node = {optimal_feature_name: {}}
    for value in optimal_feature_values:
        if len([f for f in featuresT[optimal_feature_index] if f == value]) == 0:
            child = np_mode(labels_np)
            # print(f'返回情形3：叶结点{child}')
            # return child
        else:
            child_features = features[features[optimal_feature_name] == value]
            child_labels = labels[features[optimal_feature_name] == value]
            if optimal_feature_name in float_features:
                child_feature_names = feature_names
                classifier = eval('lambda x: x' + value)
                child_features = features[classifier(features[optimal_feature_name])]
                child_labels = labels[classifier(features[optimal_feature_name])]
            else:
                child_feature_names = [f for f in feature_names if f != optimal_feature_name]
                child_features = child_features.drop(columns=optimal_feature_name)
            child = decision_tree(child_features, child_labels, child_feature_names, criterion_func, feature_values)
        node[optimal_feature_name][value] = child

    # print(f'子树递归结束：{node}')
    return node


if __name__ == '__main__':
    df = pd.read_csv('data/3.0.csv')
    features = df.iloc[:, 1:-1]
    labels = df.iloc[:, -1].map({'是': '好瓜', '否': '坏瓜'})  # .fillna(df.iloc[:, -1])
    feat_values = {col: features[col].unique() for col in features.columns}

    tree = decision_tree(features, labels, features.columns, get_optimal_feature, feat_values)
    print(tree)
