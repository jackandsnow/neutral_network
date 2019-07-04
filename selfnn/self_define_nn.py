"""
python实现自定义神经网络
Copyright by Jack
"""
import numpy as np
import scipy.special as ss


# NeutralNetwork class definition
class NeutralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in each input. hidden and output layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        # set learning rate
        self.lr = learning_rate
        # set initial weights of input_to_hidden layer, hidden_to_output layer
        # 采用正态概率分布采样权重，中心设定为0.0，标准方差为节点传入链接数目的开方的倒数，即1/(N^0.5)
        self.w_ih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w_ho = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 激活函数使用的是sigmoid函数
        self.activation_function = lambda x: ss.expit(x)

    def train(self, inputs_list, targets_list):
        # Step 1: calculate output
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # calculate errors
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_ho.T, output_errors)

        # Step 2：back forward errors
        # update weights of hidden_to_output layers
        self.w_ho += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                      np.transpose(hidden_outputs))
        # update weights of input_to_hidden layers
        self.w_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                      np.transpose(inputs))

    # query neutral network
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        # 将输入矩阵I与链接权重W组合，生成信号矩阵X：(X = W·I)，计算隐藏层的输入与输出
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最后的输入与输出
        final_inputs = np.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == '__main__':
    # 输入节点维数
    input_nodes = 28 * 28
    # 隐含节点数目
    hidden_nodes = 100
    # 输出节点维数
    output_nodes = 10
    # 学习率
    learning_rate = 0.1

    network = NeutralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # load mnist data, and train nn
    train_data_file = open('data/mnist_train.csv', 'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    print('train data length is ', len(train_data_list))
    # 训练轮数
    epochs = 3
    for i in range(epochs):
        for record in train_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            network.train(inputs, targets)

    # test nn
    test_data_file = open('data/mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        # all_value[0] is the given label
        correct_label = int(all_values[0])
        # print(correct_label, ' is correct label')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = network.query(inputs)
        label = np.argmax(outputs)
        # print(label, ' is the network\'s answer')
        scores.append(label == correct_label)
    scores_array = np.asarray(scores, dtype=int)
    print('accuracy = ', scores_array.sum() / scores_array.size)
