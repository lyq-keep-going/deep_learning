import layer
import network
import numpy as np
import random
import matplotlib.pyplot as plt
import math

n = network.Network([390, 180, 100])

# 准备训练集
# [path, target]的二维数组（包括所有图片） train//train//x//y.bmp
train_cases = []

for x in range(1,13):
    for y in range(1,601):
        tmp_str = "train//train//" + str(x) + "//" + str(y) + ".bmp"
        tmp = [tmp_str, x - 1]
        train_cases.append(tmp)

# 得想办法打乱，然后以相同顺序输入多遍 random.shuffle

# 记得加入batch训练
# 使用matlib来画图看最小值点
# 思考怎么保存数据最好

test_result = open("data//testResult3","w")

# 训练
test_result.write("以下是训练记录的数据：\n")
random.shuffle(train_cases)
min_cross_entropy = 100
for val in range(100):
    loss = 0
    correct_count = 0
    for case in train_cases:
        network_res = n.input(case[0])
        if network_res == case[1]:
            correct_count += 1
        loss += - math.log(n.network[-1].output[case[1]][0])
        n.back_propagation(case[1], 120, 0)  # batch为120, weight_decay = 0.01
    cross_entropy = loss / len(train_cases)
    if cross_entropy < min_cross_entropy:
        n.save_weights()
        n.save_bias()
        min_cross_entropy = cross_entropy
    print("%d 轮交叉熵为：%f  正确率为: %f \n" % (val, cross_entropy, correct_count / len(train_cases) ))
    test_result.write("%d 轮交叉熵为：%f  正确率为: %f \n" % (val,loss / len(train_cases), correct_count / len(train_cases)))
    random.shuffle(train_cases)

#准备测试集
test_cases = []
for x in range(1,13):
    for y in range(601,621):
        tmp_str = "train//train//" + str(x) + "//" + str(y) + ".bmp"
        tmp = [tmp_str, x - 1]
        test_cases.append(tmp)

# 测试
test_result.write("以下是测试记录的数据：\n")
loss = 0
correct_count = 0
n.load_weights()
n.load_bias()
for case in test_cases:
    network_res = n.input(case[0])
    if network_res == case[1]:
        correct_count += 1
    loss += - math.log(n.network[-1].output[case[1]][0])
    test_result.write("网络输出为 %d, 真实值为 %d \n" % (network_res,case[1]))
test_result.write("本轮交叉熵为：%f ,正确率为： %f \n" % (loss/ len(test_cases), correct_count / len(test_cases)))

test_result.close()