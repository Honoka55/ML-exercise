import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def minus_log_likelihood(w, x, y):
    log_likelihood = 0
    for i in range(len(x)):
        log_likelihood += y[i].item() * x[i] @ w - np.log(1 + np.exp(x[i] @ w))
    return -log_likelihood


def grad_minus_log_likelihood(w, x, y):
    grad = np.zeros_like(w).astype(float)
    for i in range(len(x)):
        grad -= x[i].reshape(-1, 1) * (y[i].item() - np.exp(x[i] @ w) / (1 + np.exp(x[i] @ w)))
    return grad


def hessian_minus_log_likelihood(w, x):
    hessian = np.zeros((len(w), len(w)))
    for i in range(len(x)):
        hessian += x[i].reshape(-1, 1) * x[i] * (np.exp(x[i] @ w) / (1 + np.exp(x[i] @ w)) ** 2).item()
    return hessian


def gradient_descent(x, y, learning_rate, accuracy):
    w = np.random.rand(x.shape[1]).reshape(-1, 1)
    dif = minus_log_likelihood(w, x, y)
    count = 0

    while abs(dif) > accuracy:
        old_w = w
        w = old_w - learning_rate * grad_minus_log_likelihood(old_w, x, y)
        dif = np.linalg.norm(minus_log_likelihood(w, x, y) - minus_log_likelihood(old_w, x, y))
        count += 1

        if count >= 10000:
            break
        # print(f'当前w为{w.T}，函数值为{minus_log_likelihood(w, x, y)}，梯度为{grad_minus_log_likelihood(w, x, y).T}')

    print(f'计算完成，迭代{count}次')
    return w, count


def newton_method(x, y, accuracy):
    w = np.random.rand(x.shape[1]).reshape(-1, 1)
    dif = minus_log_likelihood(w, x, y)
    count = 0

    while abs(dif) > accuracy:
        old_w = w

        hessian = hessian_minus_log_likelihood(old_w, x)
        try:
            hessian_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(f'遇到了不可逆的黑塞矩阵\n{hessian}')

        w = old_w - hessian_inv @ grad_minus_log_likelihood(old_w, x, y)
        dif = abs(minus_log_likelihood(w, x, y) - minus_log_likelihood(old_w, x, y))
        count += 1

        if count >= 10000:
            break
        # print(f'当前w为{w.T}，函数值为{minus_log_likelihood(w, x, y)}，梯度为{grad_minus_log_likelihood(w, x, y).T}')

    print(f'计算完成，迭代{count}次')
    return w, count


def sklearn_logistic_regression(x, y, C=1e5):
    logreg = LogisticRegression(C=C)
    logreg.fit(x[:, 0:-1], y.ravel().astype(int))
    w = np.append(logreg.coef_, logreg.intercept_).reshape(-1, 1)
    return w


def draw_graph(x, y, w, method, count):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['SimSun'],
        'font.size': 12,
        'mathtext.fontset': 'stix'
    })

    x_min, x_max = x.iloc[:, 0].min() - 0.1, x.iloc[:, 0].max() + 0.1
    y_min, y_max = x.iloc[:, 1].min() - 0.1, x.iloc[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    grid_predictions = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())] @ w
    grid_predictions = grid_predictions.reshape(xx.shape)

    plt.contour(xx, yy, grid_predictions, levels=[0], linewidths=2)
    plt.contourf(xx, yy, grid_predictions, levels=[-10000, 0, 10000], cmap=plt.cm.RdYlBu, alpha=0.7)

    scatter = plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', marker='o')

    good_marker = plt.Line2D([0], [0], label='好瓜', mfc=scatter.cmap(1.0), mec='k', ls='None', marker='o')
    bad_marker = plt.Line2D([0], [0], label='坏瓜', mfc=scatter.cmap(0.0), mec='k', ls='None', marker='o')
    plt.legend(handles=[good_marker, bad_marker])

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.title(f'逻辑斯谛回归{method}预测结果')
    plt.text(0.05, 0.9, r'$y = {:.8f} x_1 {:+.8f} x_2 {:+.8f}$'.format(*(w.ravel())), transform=plt.gca().transAxes)
    if count != -1:
        plt.text(0.05, 0.85, f'迭代次数：${count}$', transform=plt.gca().transAxes)
    plt.show()


if __name__ == '__main__':
    # method = '梯度下降'
    method = '牛顿法'
    # method = 'sklearn'  # 默认solver为'lbfgs'
    df = pd.read_csv('data/3.0a.csv')
    df.iloc[:, -1] = df.iloc[:, -1].map({'是': 1, '否': 0})
    x = df.iloc[:, 1:-1]
    x['intercept'] = 1
    y = df.iloc[:, -1]
    x_np, y_np = x.to_numpy(), y.to_numpy().reshape(-1, 1)

    if method == '梯度下降':
        w, count = gradient_descent(x_np, y_np, 0.1, 0)
    elif method == '牛顿法':
        w, count = newton_method(x_np, y_np, 0)
    elif method == 'sklearn':
        w = sklearn_logistic_regression(x_np, y_np)
        count = -1
    else:
        raise ValueError('不支持的方法')

    # print(w)
    draw_graph(x, y, w, method, count)
