# neutral_network

This project mainly implemented neutral networks by python3, which contains linear perceptron, feedforward neural network with single layer and multi-layers, self-defined neutral network.

The main datasets used in this project are mnist data and red-wine quality data, and you can download them in this [link](https://pan.baidu.com/s/1UmeCX9inga1aTYdCNcfqiw) with password `hibg`.

The detail introduction of neutral networks:

1.linear perceptron: Implemented through sklearn, and Iris dataset is used. You can see how the loss and accuracy changes in the following picture.

![Iris_loss_and_accuracy.PNG](https://github.com/jackandsnow/neutral_network/raw/master/images/Iris_loss_and_accuracy.PNG)

2.feedforward neural network with single layer: Implemented through sklearn, aiming to solve nonlinear classification problem. You can see the performance is excellent in the following picture. 

![fnn_ with_multi_layers.PNG](https://github.com/jackandsnow/neutral_network/raw/master/images/fnn_with_single_layer.PNG)


3.feedforward neural network with multi-layers:Implemented through keras, and EarlyStopping technology is used for saving best model. You can see part of the experiment results in the following picture.

![fnn_ with_multi_layers.PNG](https://github.com/jackandsnow/neutral_network/raw/master/images/fnn_with_multi_layers.PNG)


4.self-defined neutral network: This neutral network is just implemented by original python, which allows you to define inputs_nodes_number, hidden_nodes_number and outputs_nodes_number. And its accuracy reaches to 96.30%.
