import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report

import os
# 禁用类型断言（临时解决方案）
os.environ['TF_DISABLE_CAST_ASSERTS'] = '1'

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv("../breast_cancer_data.csv")
# print(dataset)

# 提取数据中的特征
X = dataset.iloc[:, :-1]
# print(X)

# 提取数据集中的标签
Y = dataset['target']


# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 将数据标签转化为one——hot向量格式
y_train_one = to_categorical(y_train, 2)
y_test_one = to_categorical(y_test, 2)

# 将数据特征进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# 利用Keras框架帮助搭建深度学习网络模型
model = keras.Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 对神经网络进行编译
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])


history = model.fit(x_train, y_train_one, epochs=110, batch_size=64, verbose=2, validation_data=(x_test, y_test_one))
model.save('model.h5')


# 绘制训练集和验证集的loss值对比
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()


# 绘制训练集和验证集的准确率的对比图
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("全连接神经网络accuracy值图")
plt.legend()
plt.show()











