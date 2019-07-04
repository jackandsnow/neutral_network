"""
sklearn实现线性感知器
Copyright by Jack
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

SEED = 2019

if __name__ == '__main__':
    # Iris数据库中的前两类线性可分（Iris-Setosa和Iris-Versicolour）
    iris = load_iris()
    idxs = np.where(iris.target < 2)
    X = iris.data[idxs]
    y = iris.target[idxs]

    # 绘制两个类的分布图
    # plt.scatter(X[y == 0][:, 0], X[y == 0][:, 2], color='green', label='Iris-Setosa')
    # plt.scatter(X[y == 1][:, 0], X[y == 1][:, 2], color='green', label='Iris-Versicolour')
    # plt.title('Iris Plants Database')
    # plt.xlabel('sepal length in cm')
    # plt.ylabel('sepal width in cm')
    # plt.legend()
    # plt.show()

    # 划分数据集，并定义超参数
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    weights = np.random.normal(size=X_train.shape[1])
    bias = 1
    learning_rate = 0.1
    epochs = 15

    # 训练感知器
    hist_loss = []
    hist_accuracy = []
    for i in range(epochs):
        # 使用阶跃函数，若输出>0.5则预测结果为1，否则为0
        output = np.where(X_train.dot(weights) + bias > 0.5, 1, 0)
        # 计算MSE
        error = np.mean((y_train - output) ** 2)
        # 更新权重和偏置
        weights -= learning_rate * np.dot(output - y_train, X_train)
        bias += learning_rate * np.sum(np.dot(output - y_train, X_train))
        # 计算MSE
        loss = np.mean((output - y_train) ** 2)
        hist_loss.append(loss)
        # 确定验证精度
        output_val = np.where(X_val.dot(weights) > 0.4, 1, 0)
        accuracy = np.mean(np.where(y_val == output_val, 1, 0))
        hist_accuracy.append(accuracy)

    fig = plt.figure(figsize=(8, 4))
    a = fig.add_subplot(1, 2, 1)
    plt.plot(hist_loss)
    plt.xlabel('epochs')
    a.set_title('Training loss')

    a = fig.add_subplot(1, 2, 2)
    plt.plot(hist_accuracy)
    plt.xlabel('epochs')
    a.set_title('Validation Accuracy')
    plt.show()
