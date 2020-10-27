import numpy as np
import math as math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def move(x):
    return 2 * x - 1


def dsigmoid(x):
    i = 0
    while i < len(x):
        x[i][0] = sigmoid(x[i][0]) * (1 - sigmoid(x[i][0]))
        i = i + 1
    return x


class HiddenLayer(object):

    def __init__(self, number_of_nodes, last_layer, next_layer):
        # 有前后层的信息
        self.last_layer = last_layer
        self.next_layer = next_layer
        # 本层的神经元数量
        self.number_of_nodes = number_of_nodes

    def initialize(self):
        random_array = np.array(list(map(move, np.random.rand(self.number_of_nodes * self.last_layer.number_of_nodes))))
        self.weight = random_array.reshape(self.number_of_nodes, self.last_layer.number_of_nodes)
        self.delta = np.zeros([self.number_of_nodes, self.last_layer.number_of_nodes], dtype=float, order='C')
        self.bias = np.array(list(map(move, np.random.rand(self.number_of_nodes)))).reshape(self.number_of_nodes, 1)
        self.bias_delta = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.output = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.z = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')

    def output_update(self):
        self.z = np.dot(self.weight, self.last_layer.output)
        i = 0
        while i < self.number_of_nodes:
            self.output[i][0] = sigmoid(self.z[i][0])
            i = i + 1

    def delta_update(self):
        self.delta = np.dot(self.bias_delta, self.last_layer.output.T)

    def weight_update(self):
        self.weight = self.weight - self.delta

    def bias_delta_update(self):
        self.bias_delta = np.dot(self.next_layer.weight.T, self.next_layer.bias_delta) * dsigmoid(self.z)

    def bias_update(self):
        self.bias = self.bias - self.bias_delta


class InputLayer(object):
    def __init__(self):
        self.number_of_nodes = 1

    def set_input(self, val):
        self.output = np.array([val, ])


class OutputLayer(object):
    def __init__(self, last_layer):
        self.last_layer = last_layer
        self.number_of_nodes = 1
        self.rate = 0.02  # 学习率，应该是要调整的
        self.bias_rate = 0.01

    def initialize(self):
        random_array = np.array(list(map(move, np.random.rand(self.number_of_nodes * self.last_layer.number_of_nodes))))
        self.weight = random_array.reshape(self.number_of_nodes, self.last_layer.number_of_nodes)
        self.delta = np.zeros([self.number_of_nodes, self.last_layer.number_of_nodes], dtype=float, order='C')
        self.bias = np.array(list(map(move, np.random.rand(self.number_of_nodes)))).reshape(self.number_of_nodes, 1)
        self.bias_delta = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.output = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')

    def output_update(self):
        self.output = np.dot(self.weight, self.last_layer.output)

    def delta_update(self):
        self.delta = np.dot(self.bias_delta, self.last_layer.output.T) * np.array(
            list(map(lambda x: x * self.rate, np.ones([1, self.last_layer.number_of_nodes]))))

    def weight_update(self):
        self.weight = self.weight - self.delta

    def bias_delta_update(self, dest):
        destVector = np.array([dest, ])
        self.bias_delta = - (destVector - self.output)

    def bias_update(self):
        self.bias = self.bias - self.bias_delta * np.array([self.bias_rate, ])
