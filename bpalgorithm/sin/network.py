import layer


class Network(object):
    def __init__(self, layers):
        # 输入一个list，存放每一个hiddenLayer的节点数
        self.network = []
        self.network.append(layer.InputLayer())  # 输入层
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

    def input(self, num):
        self.network[0].set_input(num)
        i = 1
        while i < len(self.network):
            self.network[i].output_update()
            i = i + 1
        return self.network[-1].output[0]

    def back_propagation(self, dest):
        # 先将outputlayer进行更新
        self.network[-1].bias_delta_update(dest)
        self.network[-1].delta_update()
        self.network[-1].weight_update()
        self.network[-1].bias_update()
        i = len(self.network) - 2
        while i > 0:
            self.network[i].bias_delta_update()
            self.network[i].delta_update()
            self.network[i].weight_update()
            self.network[i].bias_update()
            i = i - 1
