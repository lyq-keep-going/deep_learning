import layer
import network
import numpy as np
import matplotlib.pyplot as plt

n = network.Network([10])
bias = 1000  # 随便写的初始值

#准备训练集
l = []
for val in range(100):
    inp = (np.random.rand(1) * 2 - 1) * np.pi
    dest = np.sin(inp)
    tmp = [inp, dest]
    l.append(tmp)

test_result = open("testResult","w")


for val in range(10000):
    loss = 0
    for val2 in range(10):
        res = n.input(l[val2][0])
        loss += (l[val2][1] - res) * (l[val2][1] - res) / 2
        test_result.write("%d - %d:真实值为: %f ,实际输出为：%f ,与真实值之间的误差为 %f \n" % (val, val2, l[val2][1], res, l[val2][1] - res))
        n.back_propagation(l[val2][1])
    test_result.write("loss均值为：%f \n" % (loss / 10))
    inp = (np.random.rand(1) * 2 - 1) * np.pi
    dest = np.sin(inp)
    res = n.input(inp)
    loss0 = (dest - res) * (dest - res) / 2
    test_result.write("突击检查：%f \n" % loss0)

test_result.close()