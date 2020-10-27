import layer
import numpy as np

class Network(object):
    def __init__(self, layers):
        # 输入一个list，存放每一个hiddenLayer的节点数
        self.network = []
        self.network.append(layer.InputLayer())  # 输入层
        self.bp_count = 0
        i = 0
        while i < len(layers):
            self.network.append(layer.HiddenLayer(layers[i], self.network[i], None))
            i = i + 1
        self.network.append(layer.OutputLayer(self.network[i]))
        i = 0
        while i < len(self.network) - 1:
            self.network[i].next_layer = self.network[i + 1]
            if i != 0:
                self.network[i].initialize()  # 对中间层重新进行初始化
            i = i + 1
        self.network[-1].initialize()

    def save_weights(self):
        # 貌似不能追加写， 每次改层数的时候记得修改这里
        np.savez("data//weight_data2",self.network[1].weight,self.network[2].weight,self.network[3].weight,
                 self.network[4].weight)

    def load_weights(self):
        data = np.load("data//weight_data2.npz")
        self.network[1].weight = data["arr_0"]
        self.network[2].weight = data["arr_1"]
        self.network[3].weight = data["arr_2"]
        self.network[4].weight = data["arr_3"]

    def save_bias(self):
        # 貌似不能追加写， 每次改层数的时候记得修改这里
        np.savez("data//bias_data2", self.network[1].bias, self.network[2].bias, self.network[3].bias,
                 self.network[4].bias)

    def load_bias(self):
        data = np.load("data//bias_data2.npz")
        self.network[1].bias = data["arr_0"]
        self.network[2].bias = data["arr_1"]
        self.network[3].bias = data["arr_2"]
        self.network[4].bias = data["arr_3"]

    # path还是图片路径
    # 返回的是[0,11]中的值，即神经网络认为该输入图片是哪一类
    def input(self, path):
        self.network[0].set_input(path)
        i = 1
        while i < len(self.network):
            self.network[i].output_update()
            i = i + 1
        j = 0
        max_val = 0
        max_index = 0
        while j < len(self.network[-1].output):
            if self.network[-1].output[j][0] > max_val:
                max_val = self.network[-1].output[j][0]
                max_index = j
            j += 1
        return max_index

    def back_propagation(self, target, batch, weight_decay):
        self.bp_count += 1
        # 先将outputlayer进行更新
        self.network[-1].bias_delta_update(target)
        self.network[-1].delta_update()
        if self.bp_count % batch == 0:
            self.network[-1].weight_update(batch, weight_decay)
            self.network[-1].bias_update(batch)

        i = len(self.network) - 2
        while i > 0:
            self.network[i].bias_delta_update()
            self.network[i].delta_update()
            if self.bp_count % batch == 0:
                self.network[i].weight_update(batch, weight_decay)
                self.network[i].bias_update(batch)
            i = i - 1
