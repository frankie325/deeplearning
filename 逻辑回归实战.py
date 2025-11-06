import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据
dataset = pd.read_csv("breast_cancer_data.csv")

# print(dataset)

# 提取x
X = dataset.iloc[:, : -1]

# 提取数据中的标签
Y = dataset['target']
# print(Y)

# 划分数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 进行数据的归一化
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# 逻辑回归模型搭建
lr = LogisticRegression()
lr.fit(x_train, y_train)

# 打印模型的参数
# print('w:', lr.coef_)
# print('b', lr.intercept_)

# 利用训练好的模型进行推理测试
pre_reslut = lr.predict(x_test)
# print(pre_reslut)

# 打印预测结果的概率:正例和反例的概率
pre_reslut_proba = lr.predict_proba(x_test)
# print(pre_reslut_proba)

# 获取恶性肿瘤的概率
pre_list = pre_reslut_proba[:, 1]
# print(pre_list)

# 设置阈值
thresholds = 0.3

# 设置保存结果的列表
result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')


# 打印阈值调整后的结果
print(result)
print(result_name)

# 输出结果的精确率和召回还有f1值
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性肿瘤', '恶性肿瘤'])
print(report)