"""
sklearn实现单层神经网络
Copyright by Jack
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

SEED = 2019


# 激活函数
def sigmod(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # 创建内圈、外圈
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=SEED)
    outer = y == 0
    inner = y == 1

    # 绘制数据的分布(非线性可分)
    # plt.title('Two Circles')
    # plt.plot(X[outer, 0], X[outer, 1], 'ro')
    # plt.plot(X[inner, 0], X[inner, 1], 'bo')
    # plt.show()

    # 标准化数据，确保圆中心是(1,1)
    X = X + 1
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    # 定义超参数
    hidden_num = 50
    epochs = 1000
    learning_rate = 0.8
    # 初始化权重和其他变量
    weights_hidden = np.random.normal(0.0, size=(X_train.shape[1], hidden_num))
    weights_output = np.random.normal(0.0, size=hidden_num)

    # 训练神经网络
    hist_loss = []
    hist_accuracy = []
    for i in range(epochs):
        del_w_hidden = np.zeros(weights_hidden.shape)
        del_w_output = np.zeros(weights_output.shape)

        for x_, y_ in zip(X_train, y_train):
            # 前向计算
            hidden_input = np.dot(x_, weights_hidden)
            hidden_output = sigmod(hidden_input)
            output = sigmod(np.dot(hidden_output, weights_output))
            # 后向计算
            error = y_ - output
            output_error = error * output * (1 - output)
            hidden_error = np.dot(output_error, weights_output) * hidden_output * (1 - hidden_output)
            del_w_output += output_error * hidden_output
            del_w_hidden += hidden_error * x_[:, None]
        # 更新权值
        weights_hidden += learning_rate * del_w_hidden / X_train.shape[0]
        weights_output += learning_rate * del_w_output / X_train.shape[0]
        # 输出状态（验证损失的精度）
        if i % 50 == 0:
            hidden_output = sigmod(np.dot(X_val, weights_hidden))
            out = sigmod(np.dot(hidden_output, weights_output))
            loss = np.mean((out - y_val) ** 2)
            # 最终预测值基于阈值0.5
            predictions = out > 0.5
            accuracy = np.mean(predictions == y_val)
            print('Epoch: ', '{:>4}'.format(i), '; Validation loss: ', '{:>6}'.format(loss.round(4)),
                  '; Validation accuracy: ', '{:>6}'.format(accuracy.round(4)))
