import numpy as np
import math as math
from PIL import Image


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
        self.bias_delta_sum = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.output = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.z = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')

    def output_update(self):
        self.z = np.dot(self.weight, self.last_layer.output) + self.bias
        i = 0
        while i < self.number_of_nodes:
            self.output[i][0] = sigmoid(self.z[i][0])
            i = i + 1

    def delta_update(self):
        self.delta += np.dot(self.bias_delta, self.last_layer.output.T)

    def weight_update(self, batch, weight_decay):
        self.weight = self.weight * (1 - weight_decay) - self.delta / batch
        self.delta = np.zeros([self.number_of_nodes, self.last_layer.number_of_nodes], dtype=float, order='C')

    def bias_delta_update(self):
        self.bias_delta = np.dot(self.next_layer.weight.T, self.next_layer.bias_delta) * dsigmoid(self.z)
        self.bias_delta_sum += self.bias_delta

    def bias_update(self, batch):
        self.bias = self.bias - self.bias_delta_sum / batch
        self.bias_delta_sum = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')

class InputLayer(object):
    def __init__(self):
        self.number_of_nodes = 28*28

    # path表示输入的图片路径
    def set_input(self, path):
        # 这个方法将二维的28 * 28的布尔值矩阵转化成一维的784个值(按行展
        fp = open(path, 'rb')
        img = Image.open(fp)   # 这里改为文件句柄
        self.output = np.array(img).reshape([28 * 28, 1])
        fp.close()


class OutputLayer(object):
    def __init__(self, last_layer):
        self.last_layer = last_layer
        self.number_of_nodes = 12
        self.rate = 0.04  # 学习率，应该是要调整的
        self.bias_rate = 0.01  # 可调


    def initialize(self):
        random_array = np.array(list(map(move, np.random.rand(self.number_of_nodes * self.last_layer.number_of_nodes))))
        self.weight = random_array.reshape(self.number_of_nodes, self.last_layer.number_of_nodes)
        self.delta = np.zeros([self.number_of_nodes, self.last_layer.number_of_nodes], dtype=float, order='C')
        self.bias = np.array(list(map(move, np.random.rand(self.number_of_nodes)))).reshape(self.number_of_nodes, 1)
        self.bias_delta = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.bias_delta_sum = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.z = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
        self.output = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')


    def output_update(self):
        self.z = np.dot(self.weight, self.last_layer.output) + self.bias
        sum0 = 0
        i = 0
        while i < 12:
            sum0 += math.exp(self.z[i][0])
            i = i + 1
        i = 0
        while i < 12:
            self.output[i][0] = math.exp(self.z[i][0]) / sum0
            i += 1

    def delta_update(self):
        self.delta += np.dot(self.bias_delta, self.last_layer.output.T)

    def weight_update(self, batch, weight_decay):
        self.weight = self.weight * (1 - weight_decay) - self.delta * self.rate / batch
        self.delta =  np.zeros([self.number_of_nodes, self.last_layer.number_of_nodes], dtype=float, order='C')

    # destIndex 也就是target [0,11]
    def bias_delta_update(self, destIndex):
        tmp = self.output.copy()
        tmp[destIndex][0] -= 1
        self.bias_delta = tmp
        self.bias_delta_sum += tmp

    def bias_update(self, batch):
        self.bias = self.bias - self.bias_delta_sum * self.bias_rate / batch
        self.bias_delta_sum = np.zeros([self.number_of_nodes, 1], dtype=float, order='C')
