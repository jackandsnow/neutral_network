"""
keras 实现多层神经网络
Copyright by Jack
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

SEED = 2019

if __name__ == '__main__':
    # 加载数据集
    data = pd.read_csv('data/winequality-red.csv', sep=',')
    y = data['quality']
    X = data.drop(['quality'], axis=1)
    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print('Average quality training set: ', '{:.4f}'.format(y_train.mean()))
    # print(X_train.head())
    # 输入数据标准化
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    # 确定基准预测
    print('MSE: ', np.mean((y_test - ([y_train.mean()] * y_test.shape[0])) ** 2).round(4))
    # 定义神经网络结构
    model = Sequential()
    # 第一层含有100个神经元
    model.add(Dense(200, input_dim=X_train.shape[1], activation='relu'))
    # 第二层含有50个神经元
    model.add(Dense(25, activation='relu'))
    # 输出层
    model.add(Dense(1, activation='linear'))
    # 设置优化器
    opt = Adam()
    # 编译模型
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    # 定义回调函数，使用早停技术并保存最佳模型
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=200, verbose=2),
        ModelCheckpoint('result/multi_layer_model.h5', monitor='val_acc', save_best_only=True, verbose=0)
    ]
    # 定义批次大小，以及运行轮数
    batch_size = 64
    epochs = 5000
    # model.fit(X_train.values, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=2)
    model.fit(X_train.values, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=2,
              callbacks=callbacks)
    best_model = model
    best_model.load_weights('result/multi_layer_model.h5')
    best_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # 评价测试集
    score = best_model.evaluate(X_test.values, y_test, verbose=0)
    print('Test accuracy: %.2f%%' % (score[1] * 100))
