# Pandas数据分析

# 一、Pandas框架概述

## 学习目标

- 知道Pandas的作用
- 能够搭建使用Pandas的开发环境

## Pandas介绍

Python在数据处理上独步天下：代码灵活、开发快速；尤其是Python的Pandas包，无论是在数据分析领域、还是大数据开发场景中都具有显著的优势：

- Pandas是Python的一个第三方包，也是商业和工程领域最流行的**结构化数据**工具集，用于数据清洗、处理以及分析
- Pandas在数据处理上具有独特的优势：
  - 底层是基于Numpy构建的，所以运行速度特别的快
  - 有专门的处理缺失数据的API
  - 强大而灵活的分组、聚合、转换功能

![image-20240831102204286](./images/image-20240831102204286.png)

适用场景:

- 数据量大到Excel严重卡顿，且又都是单机数据的时候，我们使用Pandas
  - Pandas用于处理单机数据(小数据集(相对于大数据来说))
- 在大数据ETL数据仓库中，对数据进行清洗及处理的环节使用Pandas

## 安装Pandas

打开cmd窗口，输入如下命令：

```python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ pandas
```

![image-20240831102459782](./images/image-20240831102459782.png)

> 注意：Anaconda默认已经安装了Pandas以及Numpy等内容

## Pandas初体验

* 1- 将资料中提供的数据集导入到data目录中

![image-20240831102838020](./images/image-20240831102838020.png)

* 2- 创建python脚本, 导入pandas库

```python
import pandas as pd
```

* 3- 基于pandas加载数据

```python
df = pd.read_csv('./1960-2019全球GDP数据.csv', encoding='gbk')  
```

> \# 设置显示的最大行数和列数为None 
>
> pd.set_option('display.max_rows', None)
> pd.set_option('display.max_columns', None)

> \# 恢复显示的最大行数到默认值 
>
> pd.reset_option('display.max_rows') # 恢复显示的最大列数到默认值 pd.reset_option('display.max_columns')

> \# 恢复所有选项到默认值 pd.reset_option('all')

* 4- 基于pandas完成相关查询:

```python
# 查询中国的GDP
china_gdp = df[df.country=='中国'] # df.country 选中名为country的列
china_gdp.head(10) # 显示前10条数据
```

运行结果：

![image-20240831102944259](./images/image-20240831102944259.png)

* 5- 查询中国GDP

```python
china_gdp = df[df.country=='中国'] # df.country 选中名为country的列
china_gdp.head(10) # 显示前10条数据
```

![image-20240831103558777](./images/image-20240831103558777.png)

* 6- 将year年份设置为索引

```python
china_gdp = china_gdp.set_index('year')
china_gdp.head() # 默认显示前5条
```

![image-20240831103707688](./images/image-20240831103707688-1725071836973-3.png)

* 7- 画出GDP逐年变化的曲线图

```python
china_gdp.GDP.plot()
```

![image-20240831103753090](./images/image-20240831103753090.png)

使用同样的方法画出日本的GDP变化曲线，和中国的GDP变化曲线进行对比

```python
jp_gdp = df[df.country=='日本'].set_index('year') # 按条件选取数据后，重设索引
jp_gdp.GDP.plot() 
china_gdp.GDP.plot()
```

![image-20240831103842060](./images/image-20240831103842060.png)

分别查询中国、美国、日本三国的GDP数据，并绘制GDP变化曲线、进行对比

```python
china_gdp = df[df.country=='中国'].set_index('year')
us_gdp = df[df.country=='美国'].set_index('year')
jp_gdp = df[df.country=='日本'].set_index('year')
jp_gdp.GDP.plot()
china_gdp.GDP.plot()
us_gdp.GDP.plot()
```

![image-20240831103923889](./images/image-20240831103923889.png)

设置图例：

```python
# 按条件选取数据
china_gdp = df[df.country=='中国'].set_index('year')
us_gdp = df[df.country=='美国'].set_index('year')
jp_gdp = df[df.country=='日本'].set_index('year')
# 出图并添加图例
jp_gdp.GDP.plot(legend=True)
china_gdp.GDP.plot(legend=True)
us_gdp.GDP.plot(legend=True)
```

![image-20240831104008883](./images/image-20240831104008883.png)

修改列名使图例显示为各国名称

```python
# 按条件选取数据
china_gdp = df[df.country=='中国'].set_index('year')
us_gdp = df[df.country=='美国'].set_index('year')
jp_gdp = df[df.country=='日本'].set_index('year')
# 对指定的列修改列名
jp_gdp.rename(columns={'GDP':'japan'}, inplace=True)
china_gdp.rename(columns={'GDP':'china'}, inplace=True)
us_gdp.rename(columns={'GDP':'usa'}, inplace=True)
# 画图
jp_gdp.japan.plot(legend=True)
china_gdp.china.plot(legend=True)
us_gdp.usa.plot(legend=True)
```

![image-20240831104043579](./images/image-20240831104043579.png)

* 8- 解决中文不能在图表中正常显示的问题

将列名改为中文，使图例显示为各国名称

```python
# 按条件选取数据
china_gdp = df[df.country=='中国'].set_index('year')
us_gdp = df[df.country=='美国'].set_index('year')
jp_gdp = df[df.country=='日本'].set_index('year')
# 对指定的列修改列名
jp_gdp.rename(columns={'GDP':'日本'}, inplace=True)
china_gdp.rename(columns={'GDP':'中国'}, inplace=True)
us_gdp.rename(columns={'GDP':'美国'}, inplace=True)
# 画图
jp_gdp['日本'].plot(legend=True)
china_gdp['中国'].plot(legend=True)
us_gdp['美国'].plot(legend=True)

# 解决中文显示问题，下面的代码只需运行一次即可
import matplotlib as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

运行结果：

![image-20240831104226246](./images/image-20240831104226246.png)

## 小结

Python Pandas的作用：（清洗、处理、分析数据）

Pandas环境搭建：

- 安装Anaconda，默认自带Python以及其他相关三方包
- 使用默认的base虚拟环境启动`JupyterNotebook`

# 二、Pandas数据结构与数据类型

## 学习目标

- 知道什么是DataFrame对象、什么是Seires对象
- 对Series和DataFrame的常用API有印象、能找到、能看懂
- 了解Pandas中常用数据类型
- 知道Series以及DataFrame的运算规则
- 知道numpy.where()函数的用法

## Pandas数据结构和数据类型

![image-20240831104546666](./images/image-20240831104546666.png)

上图为上一节中读取并展示出来的数据，以此为例我们来讲解Pandas的核心概念，以及这些概念的层级关系：

- DataFrame
  - Series
    - 索引列
      - 索引名、索引值
      - 索引下标、行号
    - 数据列
      - 列名
      - 列值，具体的数据

其中最核心的就是Pandas中的两个数据结构：DataFrame和Series

## Series对象

Series也是Pandas中的最基本的数据结构对象，下文中简称s对象；是DataFrame的列对象，series本身也具有索引。

Series是一种类似于一维数组的对象，由下面两个部分组成：

![image-20240831105304751](./images/image-20240831105304751.png)

- values：一组数据（numpy.ndarray类型）
- index：相关的数据索引标签；如果没有为数据指定索引，于是会自动创建一个0到N-1(N为数据的长度)的整数型索引。

###  创建Series对象

* 1- 导入pandas 

```python
import pandas as pd
```

* 2- 通过list列表来创建

```python
# 使用默认自增索引
s2 = pd.Series([1, 2, 3])
print(s2)
# 自定义索引
s3 = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
s3


结果为:
0    1
1    2
2    3
dtype: int64
A    1
B    2
C    3
dtype: int64
```

* 3- 使用字典或元组创建series对象

```python
#使用元组
tst = (1,2,3,4,5,6)
pd.Series(tst)

#使用字典：
dst = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6}
pd.Series(dst)
```

* 4- 使用numpy创建series对象

```python
pd.Series(np.arange(10))
# 运行结果
0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: int64
```

###  Series对象属性

构造一个series对象

```python
s4 = pd.Series([i for i in range(6)], index=[i for i in 'ABCDEF'])
s4
# 返回结果如下
A    0
B    1
C    2
D    3
E    4
F    5
dtype: int64
```

* 1- series对象常用属性

```python
s4.index
```

* 2- values

```python
s4.values
```

* 3- 也可以通过索引来获取数据

```python
s4['A']
```

## DataFrame

### 创建DataFrame对象

DataFrame是一个类似于二维数组或表格(如excel)的对象，既有行索引，又有列索引

- 行索引，表明不同行，横向索引，叫index，0轴，axis=0
- 列索引，表名不同列，纵向索引，叫columns，1轴，axis=1

![image-20240831110104120](./images/image-20240831110104120.png)

DataFrame的创建有很多种方式

读取文件数据返回df：在之前的学习中我们使用了`pd.read_csv('csv格式数据文件路径')`的方式获取了df对象

使用字典、列表、元组创建df：接下来就展示如何使用字典、列表+元组、numpy创建df对象

---

* 1- 使用字典加列表创建df，使默认自增索引

```python
df1_data = {
    '日期': ['2021-08-21', '2021-08-22', '2021-08-23'],
    '温度': [25, 26, 50],
    '湿度': [81, 50, 56] 
}
df1 = pd.DataFrame(data=df1_data)
df1

# 返回结果如下
        日期    温度    湿度
0    2021-08-21    25    81
1    2021-08-22    26    50
2    2021-08-23    50    56
```

- 2- 使用列表加元组创建df，并自定义索引

```python
df2_data = [
    ('2021-08-21', 25, 81),
    ('2021-08-22', 26, 50),
    ('2021-08-23', 27, 56)
]

df2 = pd.DataFrame(
    data=df2_data, 
    columns=['日期', '温度', '湿度'],
    index = ['row_1','row_2','row_3'] # 手动指定索引
)
df2

# 返回结果如下
            日期    温度    湿度
row_1    2021-08-21    25    81
row_2    2021-08-22    26    50
row_3    2021-08-23    27    56
```

* 3- 使用numpy创建df

通过已有数据创建

```python
pd.DataFrame(np.random.randn(2,3))  # 2行3列
```

创建学生成绩表

```python
# 生成10名同学，5门功课的数据
score = np.random.randint(40, 100, (10, 5))

# 结果
array([[92, 55, 78, 50, 50],
       [71, 76, 50, 48, 96],
       [45, 84, 78, 51, 68],
       [81, 91, 56, 54, 76],
       [86, 66, 77, 67, 95],
       [46, 86, 56, 61, 99],
       [46, 95, 44, 46, 56],
       [80, 50, 45, 65, 57],
       [41, 93, 90, 41, 97],
       [65, 83, 57, 57, 40]])
```

但是这样的数据形式很难看到存储的是什么的样的数据，可读性比较差!

问题：如何让数据更有意义的显示？

```python
# 使用Pandas中的数据结构
score_df = pd.DataFrame(score)
```

![image-20240831110743788](./images/image-20240831110743788.png)

给分数数据增加行列索引,显示效果更佳

效果：

![image-20240831110830968](./images/image-20240831110830968.png)

增加行、列索引

```python
# 构造行索引序列
subjects = ["语文", "数学", "英语", "政治", "体育"]

# 构造列索引序列
stu = ['同学' + str(i) for i in range(score_df.shape[0])]

# 添加行索引
data = pd.DataFrame(score, columns=subjects, index=stu)
```

### DataFrame对象属性

属性

* 1- shape属性

```python
data.shape

# 结果
(10, 5)
```

* 2- index属性

DataFrame的行索引列表

```python
data.index

# 结果
Index(['同学0', '同学1', '同学2', '同学3', '同学4', '同学5', '同学6', '同学7', '同学8', '同学9'], dtype='object')
```

* 3- columns

```python
data.columns

# 结果
Index(['语文', '数学', '英语', '政治', '体育'], dtype='object')
```

* 4- values

直接获取其中array的值

```python
data.values

array([[92, 55, 78, 50, 50],
       [71, 76, 50, 48, 96],
       [45, 84, 78, 51, 68],
       [81, 91, 56, 54, 76],
       [86, 66, 77, 67, 95],
       [46, 86, 56, 61, 99],
       [46, 95, 44, 46, 56],
       [80, 50, 45, 65, 57],
       [41, 93, 90, 41, 97],
       [65, 83, 57, 57, 40]])
```

* 5- T

转置

```python
data.T
```

结果

![image-20240831111307825](./images/image-20240831111307825.png)

### DataFrame对象方法

1- head(n)

显示前n行内容

```python
data.head(5)
```

![image-20240831111502466](./images/image-20240831111502466.png)

2- tail(n)

显示后n行内容

如果不补充参数，默认5行。填入参数n则显示后n行

```python
data.tail(5)
```

### DatatFrame索引的设置

需求：

![image-20240831111650351](./images/image-20240831111650351.png)

* 1- 修改行列索引值

```python
stu = ["学生_" + str(i) for i in range(score_df.shape[0])]

# 必须整体全部修改
data.index = stu
```

注意：以下修改方式是错误的

```python
# 错误修改方式
data.index[3] = '学生_3'
```

* 2- 重设索引
  - reset_index(drop=False)
    - 设置新的下标索引
    - drop:默认为False，不删除原来索引，如果为True,删除原来的索引值

```python
# 重置索引,drop=False
data.reset_index()
```

![image-20240831111957305](./images/image-20240831111957305.png)

```python
# 重置索引,drop=True
data.reset_index(drop=True)
```

* 3- 以某列值设置为新的索引
  - set_index(keys, drop=True)
    - **keys** : 列索引名成或者列索引名称的列表
    - **drop** : boolean, default True.当做新的索引，删除原来的列

设置新索引案例

第一步：创建

```python
df = pd.DataFrame({'month': [1, 4, 7, 10],
                    'year': [2012, 2014, 2013, 2014],
                    'sale':[55, 40, 84, 31]})

   month  sale  year
0  1      55    2012
1  4      40    2014
2  7      84    2013
3  10     31    2014
```

第二步：以月份设置新的索引

```python
df.set_index('month')
       sale  year
month
1      55    2012
4      40    2014
7      84    2013
10     31    2014
```

第三步：设置多个索引，以年和月份

```python
df = df.set_index(['year', 'month'])
df
            sale
year  month
2012  1     55
2014  4     40
2013  7     84
2014  10    31
```



## Pandas的数据类型

df或s对象中具体每一个值的数据类型有很多，如下表所示

| Pandas数据类型 | 说明         | 对应的Python类型            |
| -------------- | ------------ | --------------------------- |
| Object         | 字符串类型   | string                      |
| int            | 整数类型     | int                         |
| float          | 浮点数类型   | float                       |
| datetime       | 日期时间类型 | datetime包中的datetime类型  |
| timedelta      | 时间差类型   | datetime包中的timedelta类型 |
| category       | 分类类型     | 无原生类型，可以自定义      |
| bool           | 布尔类型     | bool（True,False）          |
| nan            | 空值类型     | None                        |

- 可以通过下列API查看s对象或df对象中数据的类型

```python
s1.dtypes
df1.dtypes
df1.info() # s对象没有info()方法
```

* 几个特殊类型演示

  * datetime类型

  ```python
  import pandas as pd
  
  # 创建一个datetime类型的Series
  dates = pd.to_datetime(['2024-09-01', '2024-09-02', '2024-09-03'])
  print(dates)
  ```

  * timedelta类型

  ```python
  import pandas as pd
  
  # 计算两个日期之间的差值
  start_date = pd.to_datetime('2024-09-01')
  end_date = pd.to_datetime('2024-09-05')
  delta = end_date - start_date
  print(delta)
  ```

  * category类型

  类型用于表示分类数据，通常用于有限集合中的数据类型，例如性别、颜色、产品类型等。这种类型的优点在于占用更少的内存，并且对分类数据的操作更快。

  ```python
  import pandas as pd
  
  # 创建一个category类型的Series
  categories = pd.Series(['apple', 'banana', 'apple', 'orange'], dtype='category')
  print(categories)
  ```

## 小结

- series【知道】
  - 创建
    - pd.Series([], index=[])
    - pd.Series({})
  - 属性
    - 对象.index
    - 对象.values
- DataFrame【掌握】
  - 创建
    - pd.DataFrame(data=None, index=None, columns=None)
  - 属性
    - shape -- 形状
    - index -- 行索引
    - columns -- 列索引
    - values -- 查看值
    - T -- 转置
    - head() -- 查看头部内容
    - tail() -- 查看尾部内容
  - DataFrame索引
    - 修改的时候,需要进行全局修改
    - 对象.reset_index()
    - 对象.set_index(keys)

# 三、Pandas基本数据操作

## 学习目标

- 记忆DataFrame的形状、行列索引名称获取等基本属性
- 应用Series和DataFrame的索引进行切片获取
- 应用sort_index和sort_values实现索引和值的排序

## 数据集

为了更好的理解这些基本操作，我们将读取一个真实的股票数据。关于文件操作，后面在介绍，这里只先用一下API

```python
# 读取文件
data = pd.read_csv("./data/stock_day.csv")

# 删除一些列，让数据更简单些，再去做后面的操作
data = data.drop(["ma5","ma10","ma20","v_ma5","v_ma10","v_ma20"], axis=1)
```

![image-20240831113957256](./images/image-20240831113957256.png)

## 索引操作

### 直接使用行列索引(先列后行)

获取'2018-02-27'这天的'close'的结果

```python
# 直接使用行列索引名字的方式（先列后行）
data['open']['2018-02-27']
23.53

# 不支持的操作
# 错误
data['2018-02-27']['open']
# 错误
data[:1, :2]
```

### 结合loc或者iloc使用索引

获取从'2018-02-27':'2018-02-22'，'open'的结果

```python
# 使用loc:只能指定行列索引的名字
data.loc['2018-02-27':'2018-02-22', 'open']

2018-02-27    23.53
2018-02-26    22.80
2018-02-23    22.88
Name: open, dtype: float64

# 使用iloc可以通过索引的下标去获取
# 获取前3天数据,前5列的结果
data.iloc[:3, :5]

            open    high    close    low
2018-02-27    23.53    25.88    24.16    23.53
2018-02-26    22.80    23.78    23.53    22.80
2018-02-23    22.88    23.37    22.82    22.71
```

## 赋值操作

对DataFrame当中的close列进行重新赋值为1

```python
# 直接修改原来的值
data['close'] = 1
# 或者
data.close = 1
```

## 排序操作

排序有两种形式，一种对于索引进行排序，一种对于内容进行排序

### DataFrame排序

- 使用df.sort_values(by=, ascending=)
  - 单个键或者多个键进行排序,
  - 参数：
    - by：指定排序参考的键
    - ascending:默认升序
      - ascending=False:降序
      - ascending=True:升序

```python
# 按照开盘价大小进行排序 , 使用ascending指定按照大小排序
data.sort_values(by="open", ascending=True).head()
```

![image-20240831115123196](./images/image-20240831115123196.png)

```python
# 按照多个键进行排序
data.sort_values(by=['open', 'high'])
```

![image-20240831115137667](./images/image-20240831115137667.png)

- 使用df.sort_index给索引进行排序

这个股票的日期索引原来是从大到小，现在重新排序，从小到大

```python
# 对索引进行排序
data.sort_index()
```

![image-20240831115237626](./images/image-20240831115237626.png)

### Series排序

- 使用series.sort_values(ascending=True)进行排序

series排序时，只有一列，不需要参数

```python
data['p_change'].sort_values(ascending=True).head()

2015-09-01   -10.03
2015-09-14   -10.02
2016-01-11   -10.02
2015-07-15   -10.02
2015-08-26   -10.01
Name: p_change, dtype: float64
```

- 使用series.sort_index()进行排序

与df一致

```python
# 对索引进行排序
data['p_change'].sort_index().head()

2015-03-02    2.62
2015-03-03    1.44
2015-03-04    1.57
2015-03-05    2.02
2015-03-06    8.51
Name: p_change, dtype: float64
```

## 小结

- 1.索引【掌握】
  - 直接索引 -- 先列后行,是需要通过索引的字符串进行获取
  - loc -- 先行后列,是需要通过索引的字符串进行获取
  - iloc -- 先行后列,是通过下标进行索引
- 2.赋值【知道】
  - data[""] = **
  - data. **=**
- 3.排序【知道】
  - dataframe
    - 对象.sort_values()
    - 对象.sort_index()
  - series
    - 对象.sort_values()
    - 对象.sort_index()

# 四、DataFrame运算

## 学习目标

- 应用add等实现数据间的加、减法运算
- 应用逻辑运算符号实现数据的逻辑筛选
- 应用isin, query实现数据的筛选
- 使用describe完成综合统计
- 使用max, min, mean, std完成统计计算
- 使用idxmin、idxmax完成最大值最小值的索引
- 使用cumsum等实现累计分析
- 应用apply函数实现数据的自定义处理

## 算法运算

- add(other)

比如进行数学运算加上具体的一个数字

```python
data['open'].add(1)

2018-02-27    24.53
2018-02-26    23.80
2018-02-23    23.88
2018-02-22    23.25
2018-02-14    22.49
```

- sub(other)

比如进行数学运算减去具体的一个数字

```python
data['open'].sub(1)
```

## 逻辑运算符

### 逻辑运算符号

- 例如筛选data["open"] > 23的日期数据
  - data["open"] > 23返回逻辑结果

```python
data["open"] > 23

2018-02-27     True
2018-02-26    False
2018-02-23    False
2018-02-22    False
2018-02-14    False
# 逻辑判断的结果可以作为筛选的依据
data[data["open"] > 23].head()
```

![image-20240831115813316](./images/image-20240831115813316.png)

- 完成多个逻辑判断，

```python
data[(data["open"] > 23) & (data["open"] < 24)].head()
```

![image-20240831115819104](./images/image-20240831115819104.png)

### 逻辑运算函数

- query(expr)
  - expr:查询字符串

通过query使得刚才的过程更加方便简单

```python
data.query("open<24 & open>23").head()
```

- isin(values)

例如判断'open'是否为23.53和23.85

```python
# 可以指定值进行一个判断，从而进行筛选操作
data[data["open"].isin([23.53, 23.85])]
```

## 统计运算

###  describe

综合分析: 能够直接得出很多统计结果,`count`, `mean`, `std`, `min`, `max` 等

```python
# 计算平均值、标准差、最大值、最小值
data.describe()
```

![image-20240831123336370](./images/image-20240831123336370.png)

### 统计函数

Numpy当中已经详细介绍，在这里我们演示min(最小值), max(最大值), mean(平均值), median(中位数), var(方差), std(标准差),mode(众数)结果:

| `count`  | Number of non-NA observations                  |
| -------- | ---------------------------------------------- |
| `sum`    | **Sum of values**                              |
| `mean`   | **Mean of values**                             |
| `median` | Arithmetic median of values                    |
| `min`    | **Minimum**                                    |
| `max`    | **Maximum**                                    |
| `mode`   | Mode（众数）                                   |
| `abs`    | Absolute Value                                 |
| `prod`   | Product of values（乘积）                      |
| `std`    | **Bessel-corrected sample standard deviation** |
| `var`    | **Unbiased variance**                          |
| `idxmax` | compute the index labels with the maximum      |
| `idxmin` | compute the index labels with the minimum      |

> 对于单个函数去进行统计的时候，坐标轴还是按照默认列“columns” (axis=0, default)，如果要对行“index” 需要指定(axis=1)

- max()、min()

```python
# 使用统计函数：0 代表列求结果， 1 代表行求统计结果
data.max(0)

open                   34.99
high                   36.35
close                  35.21
low                    34.01
volume             501915.41
price_change            3.03
p_change               10.03
turnover               12.56
my_price_change         3.41
dtype: float64
```

* std()、var()

```python
# 方差
data.var(0)

open               1.545255e+01
high               1.662665e+01
close              1.554572e+01
low                1.437902e+01
volume             5.458124e+09
price_change       8.072595e-01
p_change           1.664394e+01
turnover           4.323800e+00
my_price_change    6.409037e-01
dtype: float64

# 标准差
data.std(0)

open                   3.930973
high                   4.077578
close                  3.942806
low                    3.791968
volume             73879.119354
price_change           0.898476
p_change               4.079698
turnover               2.079375
my_price_change        0.800565
dtype: float64
```

* median()：中位数

中位数为将数据从小到大排列，在最中间的那个数为中位数。如果没有中间数，取中间两个数的平均值。

```python
df = pd.DataFrame({'COL1' : [2,3,4,5,4,2], 'COL2' : [0,1,2,3,4,2]})

df.median()

COL1    3.5
COL2    2.0
dtype: float64
```

* idxmax()、idxmin()

```python
# 求出最大值的位置
data.idxmax(axis=0)

open               2015-06-15
high               2015-06-10
close              2015-06-12
low                2015-06-12
volume             2017-10-26
price_change       2015-06-09
p_change           2015-08-28
turnover           2017-10-26
my_price_change    2015-07-10
dtype: object


# 求出最小值的位置
data.idxmin(axis=0)

open               2015-03-02
high               2015-03-02
close              2015-09-02
low                2015-03-02
volume             2016-07-06
price_change       2015-06-15
p_change           2015-09-01
turnover           2016-07-06
my_price_change    2015-06-15
dtype: object
```

### 累计统计函数

| 函数      | 作用                        |
| --------- | --------------------------- |
| `cumsum`  | **计算前1/2/3/…/n个数的和** |
| `cummax`  | 计算前1/2/3/…/n个数的最大值 |
| `cummin`  | 计算前1/2/3/…/n个数的最小值 |
| `cumprod` | 计算前1/2/3/…/n个数的积     |

那么这些累计统计函数怎么用？

![image-20240831123744166](./images/image-20240831123744166.png)

以上这些函数可以对series和dataframe操作

这里我们按照时间的从前往后来进行累计

* 排序

```python
# 排序之后，进行累计求和
data = data.sort_index()
```

- 对p_change进行求和

```python
stock_rise = data['p_change']
# plot方法集成了前面直方图、条形图、饼图、折线图
stock_rise.cumsum()

2015-03-02      2.62
2015-03-03      4.06
2015-03-04      5.63
2015-03-05      7.65
2015-03-06     16.16
2015-03-09     16.37
2015-03-10     18.75
2015-03-11     16.36
2015-03-12     15.03
2015-03-13     17.58
2015-03-16     20.34
2015-03-17     22.42
2015-03-18     23.28
2015-03-19     23.74
2015-03-20     23.48
2015-03-23     23.74
```

那么如何让这个连续求和的结果更好的显示呢？

![image-20240831124825773](./images/image-20240831124825773.png)

如果要使用plot函数，需要导入matplotlib.

```python
import matplotlib.pyplot as plt
# plot显示图形
stock_rise.cumsum().plot()
# 需要调用show，才能显示出结果
plt.show()
```

### apply自定义运算

- apply(func, axis=0)
  - func:自定义函数
  - axis=0:默认是列，axis=1为行进行运算
- 定义一个对列，最大值-最小值的函数

```python
data[['open', 'close']].apply(lambda x: x.max() - x.min(), axis=0)

open     22.74
close    22.85
dtype: float64
```

## 小结

- 算术运算【知道】
- 逻辑运算【知道】
  - 1.逻辑运算符号
  - 2.逻辑运算函数
    - 对象.query()
    - 对象.isin()
- 统计运算【知道】
  - 1.对象.describe()
  - 2.统计函数
  - 3.累积统计函数
- 自定义运算【知道】
  - apply(func, axis=0)

# 五、文件读取与存储

## 学习目标

- 了解Pandas的几种文件读取存储操作
- 应用CSV方式、MySQL方式和JSON方式实现文件的读取和存储



我们的数据大部分存在于文件当中，所以pandas会支持复杂的IO操作，pandas的API支持众多的文件格式，如CSV、SQL、XLS、JSON、HDF5。

> 注：最常用的HDF5和CSV文件

![image-20240831125310359](./images/image-20240831125310359.png)

## CSV

### read_csv

- pandas.read_csv(filepath_or_buffer, sep =',', usecols )
  - filepath_or_buffer:文件路径
  - sep :分隔符，默认用","隔开
  - usecols:指定读取的列名，列表形式
- 举例：读取之前的股票的数据

```python
# 读取文件,并且指定只获取'open', 'close'指标
data = pd.read_csv("./data/stock_day.csv", usecols=['open', 'close'])

            open    close
2018-02-27    23.53    24.16
2018-02-26    22.80    23.53
2018-02-23    22.88    22.82
2018-02-22    22.25    22.28
2018-02-14    21.49    21.92
```

### to_csv

- DataFrame.to_csv(path_or_buf=None, sep=', ’, columns=None, header=True, index=True, mode='w', encoding=None)
  - path_or_buf :文件路径
  - sep :分隔符，默认用","隔开
  - columns :选择需要的列索引
  - header :boolean or list of string, default True
  - index:是否写进行索引,是否写进列索引值
  - mode:'w'：重写, 'a' 追加
- 举例：保存读取出来的股票数据
  - 保存'open'列的数据，然后读取查看结果

```python
# 选取10行数据保存,便于观察数据
data[:10].to_csv("./data/test.csv", columns=['open'])

# 读取，查看结果
pd.read_csv("./data/test.csv")

     Unnamed: 0    open
0    2018-02-27    23.53
1    2018-02-26    22.80
2    2018-02-23    22.88
3    2018-02-22    22.25
4    2018-02-14    21.49
5    2018-02-13    21.40
6    2018-02-12    20.70
7    2018-02-09    21.20
8    2018-02-08    21.79
9    2018-02-07    22.69
```

会发现将索引存入到文件当中，变成单独的一列数据。如果需要删除，可以指定index参数,删除原来的文件，重新保存一次。

```python
# index:存储不会讲索引值变成一列数据
data[:10].to_csv("./data/test.csv", columns=['open'], index=False)
```

## MySQL

以MySQL数据库为例，**此时默认你已经在本地安装好了MySQL数据库**。如果想利用pandas和MySQL数据库进行交互，需要先安装与数据库交互所需要的python包

```python
pip install pymysql==1.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 如果后边的代码运行提示找不到sqlalchemy的包，和pymysql一样进行安装即可
pip install sqlalchemy==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

![image-20220125205916505](./images/image-20220125205916505.png)

* 准备要写入数据库的数据

```python
import pandas as pd 
df = pd.read_csv('./csv示例文件.csv', sep=',', index_col=[0]) 
df
```

![image-20220123165416808](./images/image-20220123165416808.png)

* 创建数据库操作引擎对象并指定数据库

```python
# 需要安装pymysql，部分版本需要额外安装sqlalchemy
# 导入sqlalchemy的数据库引擎
from sqlalchemy import create_engine

# 创建数据库引擎，传入uri规则的字符串
engine = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/test?charset=utf8')
# mysql+pymysql://root:123456@127.0.0.1:3306/test?charset=utf8
# mysql 表示数据库类型
# pymysql 表示python操作数据库的包
# root:123456 表示数据库的账号和密码，用冒号连接
# 127.0.0.1:3306/test 表示数据库的ip和端口，以及名叫test的数据库
# charset=utf8 规定编码格式
```

* 将数据写入MySQL数据库

```python
# df.to_sql()方法将df数据快速写入数据库
df.to_sql('test_pdtosql', engine, index=False, if_exists='append')
# 第一个参数为数据表的名称
# 第二个参数engine为数据库交互引擎
# index=False 表示不添加自增主键
# if_exists='append' 表示如果表存在就添加，表不存在就创建表并写入
```

* 此时我们就可以在本地test库的test_pdtosql表中看到写入的数据

  ![image-20220123165745328](./images/image-20220123165745328.png)



* 从数据库中加载数据:

  * 读取整张表, 返回dataFrame

  ```python
  from sqlalchemy import create_engine
  engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/test?charset=utf8')
  
  # 指定表名，传入数据库连接引擎对象
  pd.read_sql('test_pdtosql', engine)
  ```
  
  * 使用SQL语句获取数据，返回dataframe
  
  ```python
  # 传入sql语句，传入数据库连接引擎对象
  pd.read_sql('select name,AKA from test_pdtosql', engine)
  ```
  
  ![image-20220123165932157](./images/image-20220123165932157.png)



可能出现的问题:

![image-20220125211715019](./images/image-20220125211715019.png)

```python
说明: 
	sqlalche 库版本过低导致的

解决方案:
	先删除原有版本: 
		pip uninstall sqlalchemy
	
	重新安装:
		pip install sqlalchemy==1.4.31
```



## JSON

JSON是我们常用的一种数据交换格式，前面在前后端的交互经常用到，也会在存储的时候选择这种格式。所以我们需要知道Pandas如何进行读取和存储JSON格式。

### read_json

- pandas.read_json(path_or_buf=None, orient=None, typ='frame', lines=False)

  - 将JSON格式准换成默认的Pandas DataFrame格式

  - orient : string,Indication of expected JSON string format.

    - 'split' : dict like {index -> [index], columns -> [columns], data -> [values]}

      - split 将索引总结到索引，列名到列名，数据到数据。将三部分都分开了

    - 'records' : list like [{column -> value}, ... , {column -> value}]

      - records 以`columns：values`的形式输出

    - 'index' : dict like {index -> {column -> value}}

      - index 以`index：{columns：values}...`的形式输出

    - 'columns' : dict like {column -> {index -> value}}

      ,默认该格式

      - colums 以`columns:{index:values}`的形式输出

    - 'values' : just the values array

      - values 直接输出值

  - lines : boolean, default False

    - 按照每行读取json对象

  - typ : default ‘frame’， 指定转换成的对象类型series或者dataframe

### read_json案例

- 数据介绍

这里使用一个新闻标题讽刺数据集，格式为json。`is_sarcastic`：1讽刺的，否则为0；`headline`：新闻报道的标题；`article_link`：链接到原始新闻文章。存储格式为：

```jso
{"article_link": "https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5", "headline": "former versace store clerk sues over secret 'black code' for minority shoppers", "is_sarcastic": 0}
{"article_link": "https://www.huffingtonpost.com/entry/roseanne-revival-review_us_5ab3a497e4b054d118e04365", "headline": "the 'roseanne' revival catches up to our thorny political mood, for better and worse", "is_sarcastic": 0}
```

- 读取

orient指定存储的json格式，lines指定按照行去变成一个样本

```python
json_read = pd.read_json("./data/Sarcasm_Headlines_Dataset.json", orient="records", lines=True)
```

结果：

![image-20240831130046220](./images/image-20240831130046220.png)

### to_json

- DataFrame.to_json(path_or_buf=None,orient=None,lines=False)
  - 将Pandas 对象存储为json格式
  - *path_or_buf=None*：文件地址
  - orient:存储的json形式，{‘split’,’records’,’index’,’columns’,’values’}
  - lines:一个对象存储为一行

### 案例

- 存储文件

```python
json_read.to_json("./data/test.json", orient='records')
```

结果

```
[{"article_link":"https:\/\/www.huffingtonpost.com\/entry\/versace-black-code_us_5861fbefe4b0de3a08f600d5","headline":"former versace store clerk sues over secret 'black code' for minority shoppers","is_sarcastic":0},{"article_link":"https:\/\/www.huffingtonpost.com\/entry\/roseanne-revival-review_us_5ab3a497e4b054d118e04365","headline":"the 'roseanne' revival catches up to our thorny political mood, for better and worse","is_sarcastic":0},{"article_link":"https:\/\/local.theonion.com\/mom-starting-to-fear-son-s-web-series-closest-thing-she-1819576697","headline":"mom starting to fear son's web series closest thing she will have to grandchild","is_sarcastic":1},{"article_link":"https:\/\/politics.theonion.com\/boehner-just-wants-wife-to-listen-not-come-up-with-alt-1819574302","headline":"boehner just wants wife to listen, not come up with alternative debt-reduction ideas","is_sarcastic":1},{"article_link":"https:\/\/www.huffingtonpost.com\/entry\/jk-rowling-wishes-snape-happy-birthday_us_569117c4e4b0cad15e64fdcb","headline":"j.k. rowling wishes snape happy birthday in the most magical way","is_sarcastic":0},{"article_link":"https:\/\/www.huffingtonpost.com\/entry\/advancing-the-worlds-women_b_6810038.html","headline":"advancing the world's women","is_sarcastic":0},....]
```

- 修改lines参数为True

```python
json_read.to_json("./data/test.json", orient='records', lines=True)
```

结果

```
{"article_link":"https:\/\/www.huffingtonpost.com\/entry\/versace-black-code_us_5861fbefe4b0de3a08f600d5","headline":"former versace store clerk sues over secret 'black code' for minority shoppers","is_sarcastic":0}
{"article_link":"https:\/\/www.huffingtonpost.com\/entry\/roseanne-revival-review_us_5ab3a497e4b054d118e04365","headline":"the 'roseanne' revival catches up to our thorny political mood, for better and worse","is_sarcastic":0}
{"article_link":"https:\/\/local.theonion.com\/mom-starting-to-fear-son-s-web-series-closest-thing-she-1819576697","headline":"mom starting to fear son's web series closest thing she will have to grandchild","is_sarcastic":1}
{"article_link":"https:\/\/politics.theonion.com\/boehner-just-wants-wife-to-listen-not-come-up-with-alt-1819574302","headline":"boehner just wants wife to listen, not come up with alternative debt-reduction ideas","is_sarcastic":1}
{"article_link":"https:\/\/www.huffingtonpost.com\/entry\/jk-rowling-wishes-snape-happy-birthday_us_569117c4e4b0cad15e64fdcb","headline":"j.k. rowling wishes snape happy birthday in the most magical way","is_sarcastic":0}...
```

## 小结

- pandas的CSV、MySQL、JSON文件的读取【知道】
  - 对象.read_()
  - 对象.to_()

# 六、DataFrame数据的增删改查操作

* 导包并加载数据:

```python
import pandas as pd

df = pd.read_csv('../数据集/1960-2019全球GDP数据.csv', encoding='gbk')  
df2 = df.head()
```

## 增加列

- 方式一: 通过直接赋值的方式添加新列

```python
# 拷贝一份df
df3 = df2.copy()

# 一列数据都是固定值
df3['new col 1'] = 33

# 新增列数据数量必须和行数相等
df3['new col 2'] = [1, 2, 3, 4, 5]
df3['new col 3'] = df3.year * 2

# 分别查看增加数据列之后的df和原df
df3 
df2
```

![image-20220123170950953](./images/image-20220123170950953.png)



* 方式二: df.assign函数添加列

```python
# 1. 新列名=单个数据或一组数据，一组数据的数量必须和df的行数相同
df2.assign(new0=66)
# df2.assign(new1=[1, 2, 3, 4]) # 报错
df2.assign(new1=[1, 2, 3, 4, 5])

# 2.1 新列名=Series对象，该s对象的索引和df索引一致
s = pd.Series([1, 2, 3, 4, 5]) 
df2.assign(new2=s)

# 2.2 新列名=Series对象
df2.assign(new3=df2.year+df2.GDP)

# 3. 新列名=自定义函数名
# 该自定义函数必须接收df作为参数
# 该自定义函数可以返回：
# 3.1.单个数据 
# 3.2.一组数量和df的行数相同的数据 
# 3.3.和df索引相同的Series对象
def foo(df):
    # 函数必须接收一个参数，该参数就是被传入的df对象
    print('='*10)
    print(df)
    print('='*5 + '上面输出的是传入的df')
    ret = df.index.values
    # 可以返回一个变量
    # return 'hahah'
    # 也可以返回一组变量
    return ret 
df2.assign(new4=foo)

解析：当我们使用函数的方式向df对象添加新列时，foo函数要求必须有一个参数，这个参数来自于调用assign的df对象，如上图所示
当我们给foo(df)定义一个df参数时，当assign方法开始执行，则系统会自动将df2对象作为参数传递给foo函数中的df参数
```

![image-20220123171301624](./images/image-20220123171301624.png)

![image-20220123171341864](./images/image-20220123171341864.png)

![image-20220123171406363](./images/image-20220123171406363.png)

* df.assign函数可以同时添加多列

```python
df2

def foo(df):
    return 22

def bar(df):
    return df.year + 1


df2.assign(
    new0='hahaha',
    new1=[1, 2, 3, 4, 5],
    new2=pd.Series([1, 2, 3, 4, 5]),
    new3=df.year*2,
    new4=foo,
    new5=bar
)
```

![image-20220123171512469](./images/image-20220123171512469.png)

## 删除与去重

* 1- df.drop删除行数据

```python
df3.drop([0]) # 默认删除行
df3.drop([0, 2, 4]) # 可以删除多行
df3.GDP.drop([0, 2]) # 对series对象按索引删除
```

* 2- df.drop删除列数据
  - df.drop默认删除指定索引值的行；如果添加参数`axis=1`，则删除指定列名的列

```python
df3.drop(['new col 3'], axis=1)
```

* 3- 使用del删除指定的列
  - 注意区别：
    - del是直接永久删除原df中的列【慎重使用】
    - drop是返回删除后的df或seires，原df或seires没有被修改

```python
del df3['new col 3']
df3
# 重复运行本段代码将会报错，因为df3中的指定列在第一次运行时就被删除了
```

* 4- Dataframe数据去重

```python
# 添加一部分重复的数据
df4 = df2.append(df2).reset_index(drop=True)

# 实施去重操作
df4.drop_duplicates()
```

![image-20220123175505374](./images/image-20220123175505374.png)

* 5- series去重

```python
方式一:
df4.country.drop_duplicates()
# 返回结果如下
0    美国
1    英国
2    法国
3    中国
4    日本
Name: country, dtype: object


方式二:
df4.country.unique()
# 返回结果如下
array(['美国', '英国', '法国', '中国', '日本'], dtype=object)
```

## 修改DataFrame中的数据

* 1- df.assign替换列

```python
df = pd.read_csv('../数据集/1960-2019全球GDP数据.csv', encoding='gbk', )  
df5 = df.head()
df5
df5 = df5.assign(GDP=66) # 可以接收单变量或列表、数组
df5
df # 此时原始的df不会发生改变
```

![image-20220124002357527](./images/image-20220124002357527.png)

![image-20220124002420378](./images/image-20220124002420378.png)

* 2- 直接对原始的DF进行赋值修改处理
  * 一般不建议直接修改操作

```python
df = pd.read_csv('../数据集/1960-2019全球GDP数据.csv', encoding='gbk', )  
df5 = df.head()
df5
df5['GDP'] = [5, 4, 3, 2, 1]
df5
df # 此时原始的df会发生改变
```

![image-20220124002643359](./images/image-20220124002643359.png)

* 3- replace函数替换数据

```python
# 读取数据选取前5行作为一个新的df
df = pd.read_csv('../数据集/1960-2019全球GDP数据.csv', encoding='gbk', )  
df6 = df.head()
df6
# series对象替换数据，返回的还是series对象，不会对原来的df造成修改
df6.year.replace(1960, 19600)
# 如果加上inplace=True参数，则会修改原始df
df6.country.replace('日本', '扶桑', inplace=True)
df6
# df也可以直接调用replace函数，用法和s.replace用法一致，只是返回的是df对象
df6.replace(1960, 19600)
df6

备注：replace()是按照单元格元素值进行完全匹配
```

![image-20220124002916191](./images/image-20220124002916191.png)

![image-20220124002939445](./images/image-20220124002939445.png)



## 查询dataFrame中的数据

* 1- 从前从后取多行数据

  * head()

  ```python
  # 导包 
  import pandas as pd
  # 加载csv数据，指定gbk编码格式来读取文件，返回df
  df = pd.read_csv('../数据集/1960-2019全球GDP数据.csv', encoding='gbk') 
  
  # 默认取前5行数据
  df.head()
  df.head(10) # 取前10行
  ```

  * tail()

  ```python
  # 默认取后5行数据
  df.tail()
  df2 = df.tail(15) # 倒数15行
  df2
  ```

* 2- 获取一列或多列数据

  * 获取一列数据`df[col_name]`等同于`df.col_name`

  ```python
  df2['country']
  df2.country
  # 注意！如果列名字符串中间有空格的，只能使用df['country']这种形式
  ```

  * 获取多列数据`df[[col_name1,col_name2,...]]`

  ```python
  df2[['country', 'GDP']] # 返回新的df
  ```

* 3- 索引下标切片取行

  * `df[start:stop:step]`:

  >`df[start:stop:step]` == `df[起始行下标:结束行下标:步长]` ， 遵循`顾头不顾尾`原则（包含起始行，不包含结束行），`步长`默认为1

  ```python
  df4 = df.head(10) # 取原df前10行数据作为df4，默认自增索引由0到9
  df4[0:3] # 取前3行
  df4[:5:2] # 取前5行，步长为2
  df4[1::3] # 取第2行到最后所有行，步长为3
  ```

  ![image-20220124003930052](./images/image-20220124003930052.png)

* 4- 查询函数获取子集: df.query()

  > - `df.query(判断表达式)`可以依据判断表达式返回的符合条件的df子集
  > - 与`df[布尔值向量]`效果相同
  > - 特别注意`df.query()`中传入的字符串格式

  * 示例:

  ```python
  df3.query('country=="帕劳"')
  df3[df3['country']=='帕劳']
  ```

  ![image-20220124004422387](./images/image-20220124004422387.png)

  * 查询中国, 美国  日本 三国 2015年至2019年的数据

  ```python
  df.query('country=="中国" or country=="日本" or country=="美国"').query('year in ["2015", "2016", "2017", "2018", "2019"]')
  df.query('(country=="中国" or country=="日本" or country=="美国") and year in ["2015", "2016", "2017", "2018", "2019"]')
  ```

  ![image-20220124004525553](./images/image-20220124004525553.png)

* 5- 排序函数

  * sort_values函数: 按照指定的一列或多列的值进行排序

  ```properties
  # 按GDP列的数值由小到大进行排序
  df2.sort_values(['GDP'])
  # 按GDP列的数值由大到小进行排序
  df2.sort_values(['GDP'], ascending=False) # 倒序， ascending默认为True
  # 先对year年份进行由小到大排序，再对GDP由小到大排序
  df2.sort_values(['year', 'GDP'])
  ```

  * rank函数:

  >- rank函数用法：`DataFrame.rank()` 或 `Series.rank()`
  >- rank函数返回值：以Series或者DataFrame的类型返回数据的排名（哪个类型调用返回哪个类型）
  >- rank函数包含有6个参数：
  > - **axis**：设置沿着哪个轴计算排名（0或者1)，默认为0按纵轴计算排名
  > - **numeric_only**：是否仅仅计算数字型的columns，默认为False
  > - na_option ：NaN值是否参与排序及如何排序，固定参数：keep  top  bottom
  >   - keep: NaN值保留原有位置
  >   - top: NaN值全部放在前边
  >   - bottom: NaN值全部放在最后
  > - **ascending**：设定升序排还是降序排，默认True升序
  > - **pct**：是否以排名的百分比显示排名（所有排名与最大排名的百分比），默认False
  > - method：排名评分的计算方式，固定值参数，常用固定值如下：
  >   - average : 默认值，排名评分不连续；数值相同的评分一致，都为平均值
  >   - min : 排名评分不连续；数值相同的评分一致，都为最小值
  >   - max : 排名评分不连续；数值相同的评分一致，都为最大值
  >   - dense : 排名评分是连续的；数值相同的评分一致

  ```properties
  df2
  df2.rank()
  df2.rank(axis=0)
  df2.rank(numeric_only=True) # 只对数值类型的列进行统计
  df2.rank(ascending=False) # 降序
  df2.rank(pct=True) # 以最高分作为1，放回百分数形式的评分，pct参数默认为False
  ```

  ![image-20220124010302102](./images/image-20220124010302102.png)

  ```properties
  df2.rank(method='average')
  df2.rank(method='min')
  df2.rank(method='max')
  df2.rank(method='dense')
  ```

  ![image-20220124010717941](./images/image-20220124010717941.png)

  rank使用详解

  ```python
  df7 = pd.DataFrame({
      '姓名':['小明', '小美', '小强', '小兰'],
      '成绩':[100, 90, 90, 80]
  })
  
  df7.rank()  # 等价于df7.rank(method='average', ascending=True)
  
  df7['成绩排名'] = df7.成绩.rank(method='min', ascending=False)
  df7
  
  df7['成绩排名'] = df7.成绩.rank(method='max', ascending=False)
  df7
  
  df7['成绩排名'] = df7.成绩.rank(method='dense', ascending=False)
  df7
  
  最后一种希望大家重点掌握，其他几种搞清楚原理即可。
  ```

* 6- 聚合函数:

  >常用聚合函数有：
  >
  >- min 最小值
  >- max 最大值
  >- mean 平均值
  >- sum 求和
  >- count  求个数

  * min函数

  ```properties
  df2.min()
  
  df2['year'].min() 
  
  ```

  * max函数

  ```properties
  df2.max()
  df2['year'].max()
  ```

  * mean 平均值

  ```properties
  df2.mean()
  df2['year'].mean()
  df2['GDP'].mean()
  ```



# 七、高级处理-缺失值处理

## 学习目标

- 应用isnull判断是否有缺失数据NaN
- 应用fillna实现缺失值的填充
- 应用dropna实现缺失值的删除
- 应用replace实现数据的替换

![image-20240831130543557](./images/image-20240831130543557.png)

## 如何处理NAN

- 获取缺失值的标记方式(NaN或者其他标记方式)
- 如果缺失值的标记方式是NaN
  - 判断数据中是否包含NaN：
    - pd.isnull(df),
    - pd.notnull(df)
  - 存在缺失值nan:
    - 1、删除存在缺失值的:dropna(axis='rows')
      - 注：不会修改原数据，需要接受返回值
    - 2、替换缺失值:fillna(value, inplace=True)
      - value:替换成的值
      - inplace:True:会修改原数据，False:不替换修改原数据，生成新的对象
- 如果缺失值没有使用NaN标记，比如使用"？"
  - 先替换‘?’为np.nan，然后继续处理

## 电影数据的缺失值处理

- 电影数据文件获取

```python
# 读取电影数据
movie = pd.read_csv("./dataset/movie.csv")
```

![image-20240831130641684](./images/image-20240831130641684.png)

## 判断缺失值是否存在

- pd.notnull()

```
pd.notnull(movie)

Rank    Title    Genre    Description    Director    Actors    Year    Runtime (Minutes)    Rating    Votes    Revenue (Millions)    Metascore
0    True    True    True    True    True    True    True    True    True    True    True    True
1    True    True    True    True    True    True    True    True    True    True    True    True
2    True    True    True    True    True    True    True    True    True    True    True    True
3    True    True    True    True    True    True    True    True    True    True    True    True
4    True    True    True    True    True    True    True    True    True    True    True    True
5    True    True    True    True    True    True    True    True    True    True    True    True
6    True    True    True    True    True    True    True    True    True    True    True    True
7    True    True    True    True    True    True    True    True    True    True    False    True
```

* np.all(pd.notnull(movie))

- pd.isnull()

## 存在缺失值nan,并且是np.nan

- 1- 删除

pandas删除缺失值，使用dropna的前提是，缺失值的类型必须是np.nan

```python
# 不修改原数据
movie.dropna()

# 可以定义新的变量接受或者用原来的变量名
data = movie.dropna()
```

> 获取空值行 => ① row_with_null = movie.isnull().any(axis=1) ② moive[row_with_null]

- 2- 替换缺失值

```python
# 替换存在缺失值的样本的两列
# 替换填充平均值，中位数
 movie['Revenue (Millions)'].fillna(movie['Revenue (Millions)'].mean(), inplace=True)
```

替换所有缺失值：

```python
for i in movie.columns:
    if np.all(pd.notnull(movie[i])) == False:
        print(i)
        movie[i].fillna(movie[i].mean(), inplace=True)
```

## 不是缺失值nan，有默认标记的

数据是这样的：

![image-20240831130954530](./images/image-20240831130954530.png)

```python
wis = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
```

以上数据在读取时，可能会报如下错误：

```python
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:833)>
```

解决办法：

```python
# 全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

**处理思路分析：**

- 1、先替换‘?’为np.nan
  - df.replace(to_replace=, value=)
    - to_replace:替换前的值
    - value:替换后的值

```python
# 把一些其它值标记的缺失值，替换成np.nan
wis = wis.replace(to_replace='?', value=np.nan)
```

- 2、在进行缺失值的处理

```python
# 删除
wis = wis.dropna()
```

## 小结

- isnull、notnull判断是否存在缺失值【知道】
  - np.any(pd.isnull(movie)) # 里面如果有一个缺失值,就返回True
  - np.all(pd.notnull(movie)) # 里面如果有一个缺失值,就返回False
- dropna删除np.nan标记的缺失值【知道】
  - movie.dropna()
- fillna填充缺失值【知道】
  - movie[i].fillna(value=movie[i].mean(), inplace=True)
- replace替换具体某些值【知道】
  - wis.replace(to_replace="?", value=np.NaN)

# 八、高级处理-数据合并

## 学习目标

- 应用pd.concat实现数据的合并 => union
- 应用pd.merge实现数据的合并 => join

## pd.concat实现数据合并

- pd.concat([data1, data2], axis=1)
  - 按照行或列进行合并,axis=0为列索引，axis=1为行索引

比如我们将刚才处理好的one-hot编码与原数据合并

![image-20240831131551086](./images/image-20240831131551086.png)

```python
# 按照行索引进行
pd.concat([data, dummies], axis=1)
```

## pd.merge

- pd.merge(left, right, how='inner', on=None)
  - 可以指定按照两组数据的共同键值对合并或者左右各自
  - `left`: DataFrame
  - `right`: 另一个DataFrame
  - `on`: 指定的共同键
  - how:按照什么方式连接

| Merge method | SQL Join Name      | Description                               |
| ------------ | ------------------ | ----------------------------------------- |
| `left`       | `LEFT OUTER JOIN`  | Use keys from left frame only             |
| `right`      | `RIGHT OUTER JOIN` | Use keys from right frame only            |
| `outer`      | `FULL OUTER JOIN`  | Use union of keys from both frames        |
| `inner`      | `INNER JOIN`       | Use intersection of keys from both frames |

内连接：

```python
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                        'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                        'key2': ['K0', 'K0', 'K0', 'K0'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']})

# 默认内连接
result = pd.merge(left, right, on=['key1', 'key2'])
```

![image-20240831231631642](./images/image-20240831231631642.png)

左连接

```python
result = pd.merge(left, right, how='left', on=['key1', 'key2'])
```

![image-20240831231730179](./images/image-20240831231730179.png)

右连接

```python
result = pd.merge(left, right, how='right', on=['key1', 'key2'])
```

![image-20240831231751903](./images/image-20240831231751903.png)

外连接

```python
result = pd.merge(left, right, how='outer', on=['key1', 'key2'])
```

![image-20240831231814065](./images/image-20240831231814065.png)

## 小结

- pd.concat([数据1, 数据2], axis=**)【知道】
- pd.merge(left, right, how=, on=)【知道】
  - how -- 以何种方式连接
  - on -- 连接的键的依据是哪几个

# 九、高级处理-数据分组

## 数据准备

- 加载优衣库的销售数据集，包含了不同城市优衣库门店的所有产品类别的销售记录，数据字段说明如下
  - store_id 门店随机id
  - city 城市
  - channel 销售渠道 网购自提 门店购买
  - gender_group 客户性别 男女
  - age_group 客户年龄段
  - wkd_ind 购买发生的时间（周末，周间）
  - product 产品类别
  - customer 客户数量
  - revenue 销售金额
  - order 订单数量
  - quant 购买产品的数量
  - unit_cost 成本（制作+运营）

```properties
# 导包 加载数据集
import pandas as pd 
df = pd.read_csv('../数据集/uniqlo.csv')
```

## groupby分组聚合

* 1- df.groupby分组函数返回分组对象

  【基于一列进行分组】

```python
# 基于顾客性别分组
gs = df.groupby(['gender_group'])
gs
gs['city']
# 返回结果如下
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001B1DA988B80>
<pandas.core.groupby.generic.SeriesGroupBy object at 0x000001B1DB2B4FA0>
```

​		【基于多列进行分组】

```python
# 基于顾客性别、不同城市分组
gs2 = df.groupby(['gender_group', 'city'])
gs2
# 返回结果如下
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001B1DB24F1F0>
```

* 2- 分组后获取各个组内的数据

  * 2.1 取出每组第一条或最后一条数据

  ```properties
  gs2 = df.groupby(['gender_group', 'channel'])
  gs2.first() # 取出每组第一条数据
  gs2.last() # 取出每组最后一条数据
  ```

  ![image-20220124012357765](./images/image-20220124012357765.png)

  * 2.2 - 按分组依据获取其中一组

  ```python
  gs2.get_group(('Female', '线上'))
  ```

  ![image-20220124012526559](./images/image-20220124012526559.png)

* 3- 分组聚合

  * 格式:分组后对多列分别使用不同的聚合函数

  ```python
  df.groupby(['列名1', '列名2']).agg({
      '指定列1':'聚合函数名', 
      '指定列2':'聚合函数名', 
      '指定列3':'聚合函数名'
  })
  ```

  - 按城市和线上线下划分，分别计算销售金额的平均值、成本的总和

  ```python
  df.groupby(['city', 'channel']).agg({
      'revenue':'mean', 
      'unit_cost':'sum'
  })
  ```

  ![image-20220124013227222](./images/image-20220124013227222.png)

  ### 3、分组过滤操作

  * 格式:

  ```python
  df.groupby(['列名1',...]).filter(
      lambda x: dosomething returun True or False
  )
  ```

  案例: 按城市分组，查询每组销售金额平均值大于200的全部数据

  ```python
  df.groupby(['city']).filter(lambda s: s['revenue'].mean() > 200)
  df.groupby(['city'])['revenue'].filter(lambda s: s.mean() > 200)
  ```




# 十、高级处理-交叉表与透视表

## 学习目标

- 应用crosstab和pivot_table实现交叉表与透视表

## 交叉表与透视表有什么作用？

探究股票的涨跌与星期几有关？

以下图当中表示，week代表星期几，1,0代表这一天股票的涨跌幅是好还是坏，里面的数据代表比例

可以理解为所有时间为星期一等等的数据当中涨跌幅好坏的比例

![image-20240831232835369](./images/image-20240831232835369.png)

![image-20240831232858699](./images/image-20240831232858699.png)

- 交叉表：

  交叉表用于计算一列数据对于另外一列数据的分组个数(用于统计分组频率的特殊透视表)

  - pd.crosstab(value1, value2)

- 透视表：

  透视表是将原有的DataFrame的列分别作为行索引和列索引，然后对指定的列应用聚集函数

  - data.pivot_table(）
  - DataFrame.pivot_table([], index=[])

## 交叉表

```python
import pandas as pd

# 创建一个示例数据集
data = {
    '性别': ['男', '女', '男', '女', '男', '女', '女', '男'],
    '购买': ['是', '否', '是', '是', '否', '否', '是', '否']
}

df = pd.DataFrame(data)

# 创建交叉表
crosstab = pd.crosstab(df['性别'], df['购买'])
print(crosstab)

购买  否  是
性别        
女    2   2
男    2   2
```

## 透视表

```python
import pandas as pd

data = {
    '性别': ['男', '女', '男', '女', '男', '女'],
    '购买': ['是', '否', '是', '是', '否', '否'],
    '金额': [100, 150, 200, 130, 160, 120]
}
df = pd.DataFrame(data)

# 创建透视表
pivot_table = pd.pivot_table(df, values='金额', index='性别', columns='购买', aggfunc='mean')
print(pivot_table)

购买      否     是
性别             
女   135.0  130.0
男   180.0  150.0
```

## 案例分析

###  数据准备

- 准备两列数据，星期数据以及涨跌幅是好是坏数据
- 进行交叉表计算

```python
# 寻找星期几跟股票张得的关系
# 1、先把对应的日期找到星期几
date = pd.to_datetime(data.index).weekday
data['week'] = date

# 2、假如把p_change按照大小去分个类0为界限
data['posi_neg'] = np.where(data['p_change'] > 0, 1, 0)

# 通过交叉表找寻两列数据的关系
count = pd.crosstab(data['week'], data['posi_neg'])
```

但是我们看到count只是每个星期日子的好坏天数，并没有得到比例，该怎么去做？

- 对于每个星期一等的总天数求和，运用除法运算求出比例

```python
# 算数运算，先求和
sum = count.sum(axis=1).astype(np.float32)

# 进行相除操作，得出比例
pro = count.div(sum, axis=0)
```

### 查看效果

使用plot画出这个比例，使用stacked的柱状图

```python
pro.plot(kind='bar', stacked=True)
plt.show()
```

### 使用pivot_table(透视表)实现

使用透视表，刚才的过程更加简单

```python
# 通过透视表，将整个过程变成更简单一些
data.pivot_table(['posi_neg'], index='week')
```

## 小结

- 交叉表与透视表的作用【知道】
  - 交叉表：计算一列数据对于另外一列数据的分组个数
  - 透视表：指定某一列对另一列的关系
