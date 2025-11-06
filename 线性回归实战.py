# 定义数据集

# 定义数据特征
x_data = [1, 2, 3]

# 定义数据标签
y_data = [2, 4, 6]

# 初始化参数w
w = 4


# 定义线性回归模型
def forward(x):
    return x * w


# 定义损失函数
def cost(xs, ys):
    costvalue = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        # 损失函数 每一项((预测值 - 真实值）** 2) 求和 / 数组长度
        costvalue += (y - y_pred) ** 2
    return costvalue / len(xs)  # 返回平均损失


# 定义计算梯度的函数
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        """
        J(w)为损失函数 (wx - y)**2
        梯度就是：J(w)对w的导数为2(wx -y)*x   理解：导数就是斜率，根据 加/减斜率去找到最小损失值
        """
        grad = grad + 2 * x * (w * x - y)
    return grad / len(xs)  # 求出这组数据的平均梯度


#
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad = gradient(x_data, y_data)
    """
    根据梯度公式：w = w - 常数a（表示学习率，学习率理解为一次跨度） * J(w)对w的导数
            w = w - a * 2(wx - y)*x
    逐步算出最合适的w，使得损失函数值最小
    """
    a = 0.01 # 学习率取值太大，步子会跨太大
    w = w - a * grad;
    print("训练次数:", epoch, "w=", w, "损失值", cost_val)


print("100轮训练已经做好了，此时我们用训练好的w进行推理，学习时间为4个小时的时候最终得分为：", forward(4))