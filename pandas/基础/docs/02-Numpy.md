# Numpy

# 一、Numpy优势

## 学习目标

- 目标
  - 了解Numpy运算速度上的优势
  - 知道Numpy的数组内存块风格
  - 知道Numpy的并行化运算

----

## 1 Numpy介绍

![Numpy](./images/Numpy.png)

Numpy（Numerical Python）是一个开源的Python科学计算库，**用于快速处理任意维度的数组**。

Numpy**支持常见的数组和矩阵操作**。对于同样的数值计算任务，使用Numpy比直接使用Python要简洁的多。

Numpy**使用ndarray对象来处理多维数组**，该对象是一个快速而灵活的大数据容器。

## 2 ndarray介绍

```python
NumPy provides an N-dimensional array type, the ndarray, 
which describes a collection of “items” of the same type.
```

NumPy提供了一个**N维数组类型ndarray**，它描述了**相同类型**的“items”的集合。

![学生成绩数据](./images/%E5%AD%A6%E7%94%9F%E6%88%90%E7%BB%A9%E6%95%B0%E6%8D%AE.png)

用ndarray进行存储：

```python
import numpy as np

# 创建ndarray
score = np.array(
[[80, 89, 86, 67, 79],
[78, 97, 89, 67, 81],
[90, 94, 78, 67, 74],
[91, 91, 90, 67, 69],
[76, 87, 75, 67, 86],
[70, 79, 84, 67, 84],
[94, 92, 93, 67, 64],
[86, 85, 83, 67, 80]])

score
```

返回结果：

```python
array([[80, 89, 86, 67, 79],
       [78, 97, 89, 67, 81],
       [90, 94, 78, 67, 74],
       [91, 91, 90, 67, 69],
       [76, 87, 75, 67, 86],
       [70, 79, 84, 67, 84],
       [94, 92, 93, 67, 64],
       [86, 85, 83, 67, 80]])
```

**提问:**

**使用Python列表可以存储一维数组，通过列表的嵌套可以实现多维数组，那么为什么还需要使用Numpy的ndarray呢？**

## 3 ndarray与Python原生list运算效率对比

在这里我们通过一段代码运行来体会到ndarray的好处

```python
import random
import time
import numpy as np
a = []
for i in range(100000000):
    a.append(random.random())
    
# 通过%time魔法方法, 查看当前行的代码运行一次所花费的时间
%time sum1=sum(a)

b=np.array(a)

%time sum2=np.sum(b)
```

其中第一个时间显示的是使用原生Python计算时间,第二个内容是使用numpy计算时间:

```
CPU times: user 852 ms, sys: 262 ms, total: 1.11 s
Wall time: 1.13 s
CPU times: user 133 ms, sys: 653 µs, total: 133 ms
Wall time: 134 ms
```

从中我们看到ndarray的计算速度要快很多，节约了时间。

**机器学习的最大特点就是大量的数据运算**，那么如果没有一个快速的解决方案，那可能现在python也在机器学习领域达不到好的效果。

![计算量大](./images/%E8%AE%A1%E7%AE%97%E9%87%8F%E5%A4%A7.png)

Numpy专门针对ndarray的操作和运算进行了设计，所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，数组越大，Numpy的优势就越明显。

**思考：**

**ndarray为什么可以这么快？**

## 4 ndarray的优势

#### 4.1 内存块风格

ndarray到底跟原生python列表有什么不同呢，请看一张图：

![numpy内存地址](./images/numpy%E5%86%85%E5%AD%98%E5%9C%B0%E5%9D%80.png)

从图中我们可以看出ndarray在存储数据的时候，数据与数据的地址都是连续的，这样就给使得批量操作数组元素时速度更快。

这是因为ndarray中的所有元素的类型都是相同的，而Python列表中的元素类型是任意的，所以ndarray在存储元素时内存可以连续，而python原生list就只能通过寻址方式找到下一个元素，这虽然也导致了在通用性能方面Numpy的ndarray不及Python原生list，但在科学计算中，Numpy的ndarray就可以省掉很多循环语句，代码使用方面比Python原生list简单的多。

#### 4.2 ndarray支持并行化运算（向量化运算）

numpy内置了并行运算功能，当系统有多个核心时，做某种计算时，numpy会自动做并行计算

#### 4.3 效率远高于纯Python代码

Numpy底层使用C语言编写，内部解除了GIL（全局解释器锁），其对数组的操作速度不受Python解释器的限制，所以，其效率远高于纯Python代码。

## 5 小结

- numpy介绍【了解】
  - 一个开源的Python科学计算库
  - 计算起来要比python简洁高效
  - Numpy使用ndarray对象来处理多维数组
- ndarray介绍【了解】
  - NumPy提供了一个N维数组类型ndarray，它描述了相同类型的“items”的集合。
  - 生成numpy对象:np.array()
- ndarray的优势【掌握】
  - 内存块风格
    - list -- 分离式存储,存储内容多样化
    - ndarray -- 一体式存储,存储类型必须一样
  - ndarray支持并行化运算（向量化运算）
  - ndarray底层是用C语言写的,效率更高,释放了GIL

# 二、N维数组-ndarray

## 学习目标

- 目标
  - 说明数组的属性，形状、类型

---

## 1 ndarray的属性

数组属性反映了数组本身固有的信息。

|     属性名字     |          属性解释          |
| :--------------: | :------------------------: |
|  ndarray.shape   |       数组维度的元组       |
|   ndarray.ndim   |          数组维数          |
|   ndarray.size   |      数组中的元素数量      |
| ndarray.itemsize | 一个数组元素的长度（字节） |
|  ndarray.dtype   |       数组元素的类型       |

## 2 ndarray的形状

首先创建一些数组。

```python
# 创建不同形状的数组
>>> a = np.array([[1,2,3],[4,5,6]])
>>> b = np.array([1,2,3,4])
>>> c = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
```

分别打印出形状

```python
>>> a.shape
>>> b.shape
>>> c.shape

(2, 3)  # 二维数组
(4,)	# 一维数组
(2, 2, 3) # 三维数组
```

如何理解数组的形状？

二维数组：

![数组1](./images/%E6%95%B0%E7%BB%841.png)

三维数组：

![数组2](./images/%E6%95%B0%E7%BB%842.png)



## 3 ndarray的类型

```python
>>> type(score.dtype)

<type 'numpy.dtype'>
```

dtype是numpy.dtype类型，先看看对于数组来说都有哪些类型

|     名称      |                          描述                           | 简写  |
| :-----------: | :-----------------------------------------------------: | :---: |
|    np.bool    |         用一个字节存储的布尔类型（True或False）         |  'b'  |
|    np.int8    |            tinyint一个字节大小，-128 至 127             |  'i'  |
|   np.int16    |              smallint整数，-32768 至 32767              | 'i2'  |
|   np.int32    |                int整数，-2^31​ 至 2^32 -1                | 'i4'  |
|   np.int64    |              bigint整数，-2^63 至 2^63 - 1              | 'i8'  |
|   np.uint8    |          tinyint unsigned无符号整数，0 至 255           |  'u'  |
|   np.uint16   |         smallint unsigned无符号整数，0 至 65535         | 'u2'  |
|   np.uint32   |                无符号整数，0 至 2^32 - 1                | 'u4'  |
|   np.uint64   |                无符号整数，0 至 2^64 - 1                | 'u8'  |
|  np.float16   |    半精度浮点数：16位，正负号1位，指数5位，精度10位     | 'f2'  |
|  np.float32   |  float单精度浮点数：32位，正负号1位，指数8位，精度23位  | 'f4'  |
|  np.float64   | double双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8'  |
| np.complex64  |        复数，分别用两个32位浮点数表示实部和虚部         | 'c8'  |
| np.complex128 |        复数，分别用两个64位浮点数表示实部和虚部         | 'c16' |
|  np.object_   |                       python对象                        |  'O'  |
|  np.string_   |                         字符串                          |  'S'  |
|  np.unicode_  |                  unicode类型（字符串）                  |  'U'  |

常用的几个：

**`np.int32`**：32位整数，是最常用的整数类型，适用于大多数整数运算。

**`np.float64`**：64位浮点数，是默认的浮点数类型，广泛用于科学计算。

**`np.bool_`**：布尔类型，用于表示True或False，常用于条件判断和逻辑操作。

**`np.string_/np.unicode_`**：定长字符串类型，常用于二进制数据 或 多语言文本数据

**`np.object_`**：用于存储任意Python对象，特别是在处理混合类型数据或需要灵活性的时候。

> `np.string_`只支持ASCII编码，不支持Unicode，而`np.unicode_`支持Unicode字符。
>
> `np.string_`更适合处理旧有的二进制数据，而`np.unicode_`更适合处理现代文本数据。

**创建数组的时候指定类型**

```python
>>> a = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
>>> a.dtype
dtype('float32')

>>> arr = np.array(['python', 'tensorflow', 'scikit-learn', 'numpy'], dtype = np.string_)
>>> arr
array([b'python', b'tensorflow', b'scikit-learn', b'numpy'], dtype='|S12')
```

- 注意：若不指定，整数默认int64，小数默认float64

## 4 总结

数组的基本属性【知道】

|     属性名字      |          属性解释          |
| :---------------: | :------------------------: |
| **ndarray.shape** |       数组维度的元组       |
|   ndarray.ndim    |          数组维数          |
|   ndarray.size    |      数组中的元素数量      |
| ndarray.itemsize  | 一个数组元素的长度（字节） |
| **ndarray.dtype** |       数组元素的类型       |

# 三、基本操作

## 学习目标

- 目标
  - 理解数组的各种生成方法
  - 应用数组的索引机制实现数组的切片获取
  - 应用维度变换实现数组的形状改变
  - 应用类型变换实现数组类型改变
  - 应用数组的转换

---

## 1 生成数组的方法

### 1.1 生成0和1的数组

* **np.ones(shape, dtype)**
* np.ones_like(a, dtype) ：用于创建一个与数组 `a` 形状相同且所有元素都为1的数组的函数。
* **np.zeros(shape, dtype)**
* np.zeros_like(a, dtype) : ：用于创建一个与数组 `a` 形状相同且所有元素都为0的数组的函数。

```python
ones = np.ones([4,8])
ones
```

返回结果:

```python
array([[1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1.]])
```

```python
np.zeros_like(ones)
```

返回结果:

```python
array([[0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]])
```



### 1.2 从现有数组生成

#### 1.2.1 生成方式

* **np.array(object, dtype)**

- **np.asarray(a, dtype)**

```python
a = np.array([[1,2,3],[4,5,6]])
# 从现有的数组当中创建
a1 = np.array(a)
# 相当于索引的形式，并没有真正的创建一个新的
a2 = np.asarray(a)
```

#### 1.2.2 关于array和asarray的不同

![image-20190618211642426](./images/array%E5%92%8Casarray%E7%9A%84%E5%8C%BA%E5%88%AB.png)



### 1.3 生成固定范围的数组

类似于之前讲过的range()

#### 1.3.1 np.linspace (start, stop, num, endpoint)

* 创建等差数组 — ==指定数量==
* 参数:
  * start:序列的起始值
  * stop:序列的终止值
  * num:要生成的等间隔样例数量，默认为50
  * endpoint:序列中是否包含stop值，默认为True

```python
# 生成等间隔的数组
np.linspace(0, 100, 11)
```

返回结果：

```python
array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
```

#### 1.3.2 np.arange(start,stop, step, dtype)

* 创建等差数组 — ==指定步长==
* 参数
  * step:步长,默认值为1

```python
np.arange(10, 50, 2)
```

返回结果：

```python
array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
       44, 46, 48])
```

#### 1.3.3 np.logspace(start,stop, num)

- 创建等比数列

- 参数:
  - num:要生成的等比数列数量，默认为50

```python
# 生成10^x
np.logspace(0, 2, 3) 
```

返回结果:

```shell
array([  1.,  10., 100.])
```



### 1.4 生成随机数组（绘图专用）

#### 1.4.1 使用模块介绍

* np.random模块

```python
import numpy as np

# 生成一个[0.0, 1.0)之间的均匀分布的随机浮点数
rand_num = np.random.rand()
print("均匀分布的随机浮点数:", rand_num)

# 生成一个形状为(3, 2)的均匀分布的随机浮点数组
rand_array = np.random.rand(3, 2)
print("均匀分布的随机数组:\n", rand_array)
```

```python
import numpy as np

# 生成一个从0到9的随机整数
rand_int = np.random.randint(0, 10)
print("随机整数:", rand_int)

# 生成一个形状为(4, 3)的随机整数数组，范围在[1, 100)之间
rand_int_array = np.random.randint(1, 100, size=(4, 3))
print("随机整数数组:\n", rand_int_array)
```

## 2 数组的索引、切片

一维、二维、三维的数组如何索引？

* 直接进行索引,切片
* 对象[:, :] -- 先行后列

基本索引：

```python
import numpy as np

# 创建一个 2x3 的数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 访问第1行第2列的元素（注意：索引从0开始）
element = arr[0, 1]
print("第1行第2列的元素:", element)  # 输出: 2

# 访问第2行第3列的元素
element = arr[1, 2]
print("第2行第3列的元素:", element)  # 输出: 6

```

切片操作

**二维数组**：可以通过 `[row, column]` 进行索引和切片，提取特定的行、列或子矩阵。

**三维数组**：可以通过 `[depth, row, column]` 进行索引和切片，提取特定的层、行、列或子阵列。

```python
import numpy as np

# 创建一个 3x4 的数组
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 提取第1行和第2行的所有列
sub_array = arr[0:2, :]
print("第1行和第2行的所有列:\n", sub_array)

# 提取第2列和第3列的所有行
sub_array = arr[:, 1:3]
print("第2列和第3列的所有行:\n", sub_array)

# 提取第1行第2列到第3列的元素
sub_array = arr[0, 1:3]
print("第1行第2列到第3列的元素:", sub_array)
```

- 三维数组索引方式：


```python
# 三维
a1 = np.array([[[1,2,3],[4,5,6]], [[12,3,34],[5,6,7]]])
# 返回结果
array([[[ 1,  2,  3],
        [ 4,  5,  6]],

       [[12,  3, 34],
        [ 5,  6,  7]]])
# 索引、切片
>>> a1[0, 0, 1]   # 输出: 2
```



## 3 形状修改

### 3.1 ndarray.reshape(shape, order)

* 返回一个具有相同数据域，但shape不一样的**视图**
* 行、列不进行互换

```python
import numpy as np

# 创建一个 1x9 的数组
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 将其重新构造成 3x3 的数组
reshaped_arr = arr.reshape(3, 3)
print("原数组:\n", arr)
print("reshape后的数组:\n", reshaped_arr)

运行结果
原数组:
 [1 2 3 4 5 6 7 8 9]
reshape后的数组:
 [[1 2 3]
  [4 5 6]
  [7 8 9]]
```

### 3.2 ndarray.resize(new_shape)

* 修改数组本身的形状（需要保持元素个数前后相同）
* 行、列不进行互换

```python
import numpy as np

# 创建一个 1x9 的数组
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 调整数组的大小为 2x5
resized_arr = np.resize(arr, (2, 5))
print("原数组:\n", arr)
print("resize后的数组:\n", resized_arr)

运行结果
原数组:
 [1 2 3 4 5 6 7 8 9]
resize后的数组:
 [[1 2 3 4 5]
  [6 7 8 9 1]]
```

### 3.3 ndarray.T

* 数组的转置
* 将数组的行、列进行互换

```python
import numpy as np

# 创建一个 2x3 的二维数组
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 对数组进行转置
transposed_arr = arr.T

print("原始数组:\n", arr)
print("转置后的数组:\n", transposed_arr)

运行结果
原始数组:
 [[1 2 3]
  [4 5 6]]
转置后的数组:
 [[1 4]
  [2 5]
  [3 6]]
```


## 4 类型修改

### 4.1 ndarray.astype(type)

* 返回修改了类型之后的数组

```python
import numpy as np

# 创建一个浮点数类型的数组
arr = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

# 使用 .astype(np.int32) 将数组的元素类型转换为 int32
arr_int32 = arr.astype(np.int32)

print("原始数组:", arr)
print("原始数组的类型:", arr.dtype)

print("转换后的数组:", arr_int32)
print("转换后的数组类型:", arr_int32.dtype)
```

### 4.2 ndarray.tobytes([order])

* 构造包含数组中原始数据字节的Python字节

```python
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[12, 3, 34], [5, 6, 7]]])
arr.tobytes()
```

为什么转二进制？方便网络传输



## 5 数组的去重

### 5.1 np.unique()

```python
temp = np.array([[1, 2, 3, 4],[3, 4, 5, 6]])
np.unique(temp)
array([1, 2, 3, 4, 5, 6])
```



## 6 小结

- 创建数组【掌握】

  - 生成0和1的数组
    - np.ones()
    - np.ones_like()
  - 从现有数组中生成
    - np.array -- 深拷贝
    - np.asarray -- 浅拷贝
  - 生成固定范围数组
    - np.linspace()
      - nun -- 生成等间隔的多少个
    - np.arange()
      - step -- 每间隔多少生成数据
    - np.logspace()
      - 生成以10的N次幂的数据

  - 生层随机数组
    - 正态分布
      - 里面需要关注的参数:均值:u, 标准差:σ
        - u -- 决定了这个图形的左右位置
        - σ -- 决定了这个图形是瘦高还是矮胖
      - np.random.randn()
      - np.random.normal(0, 1, 100)
    - 均匀
      - np.random.rand()
      - np.random.uniform(0, 1, 100)
      - np.random.randint(0, 10, 10)

- 数组索引【知道】
  - 直接进行索引,切片
  - 对象[:, :] -- 先行后列

- 数组形状改变【掌握】
  - 对象.reshape()
    - 没有进行行列互换,新产生一个ndarray
  - 对象.resize()
    - 没有进行行列互换,修改原来的ndarray
  - 对象.T
    - 进行了行列互换

- 数组去重【知道】

  - np.unique(对象)

# 四、ndarray运算

## 学习目标

- 目标
  - 应用数组的通用判断函数
  - 应用np.where实现数组的三元运算

---

## 问题

**如果想要操作符合某一条件的数据，应该怎么做？**

## 1 逻辑运算

```python
# 生成10名同学，5门功课的数据
>>> score = np.random.randint(40, 100, (10, 5))

# 取出最后4名同学的成绩，用于逻辑判断
>>> test_score = score[6:, 0:5]

# 逻辑判断, 如果成绩大于60就标记为True 否则为False
>>> test_score > 60
array([[ True,  True,  True, False,  True],
       [ True,  True,  True, False,  True],
       [ True,  True, False, False,  True],
       [False,  True,  True,  True,  True]])

# BOOL赋值, 将满足条件的设置为指定的值-布尔索引
>>> test_score[test_score > 60] = 1
>>> test_score
array([[ 1,  1,  1, 52,  1],
       [ 1,  1,  1, 59,  1],
       [ 1,  1, 44, 44,  1],
       [59,  1,  1,  1,  1]])
```

## 2 通用判断函数

* np.all()

  当你需要检查数组中的所有元素是否都满足条件时使用，如果所有元素都满足条件，返回 `True`，否则返回 `False`

```python
# 判断前两名同学的成绩[0:2, :]是否全及格
>>> np.all(score[0:2, :] > 60)
False
```

* np.any()

  当你需要检查数组中是否至少有一个元素满足条件时使用，如果有一个元素满足条件，返回 `True`，否则返回 `False`

```python
# 判断前两名同学的成绩[0:2, :]是否有大于90分的
>>> np.any(score[0:2, :] > 80)
True
```

## 3 np.where（三元运算符）

通过使用np.where能够进行更加复杂的运算

* np.where() => 类似Python中的if...else结构

```python
# 判断前四名学生,前四门课程中，成绩中大于60的置为1，否则为0
temp = score[:4, :4]
np.where(temp > 60, 1, 0)
```

* 复合逻辑需要结合np.logical_and和np.logical_or使用

```python
# 判断前四名学生,前四门课程中，成绩中大于60且小于90的换为1，否则为0
np.where(np.logical_and(temp > 60, temp < 90), 1, 0)

# 判断前四名学生,前四门课程中，成绩中大于90或小于60的换为1，否则为0
np.where(np.logical_or(temp > 90, temp < 60), 1, 0)
```

## 4  统计运算

**如果想要知道学生成绩最大的分数，或者做小分数应该怎么做？**

###  4.1 统计指标

在数据挖掘/机器学习领域，统计指标的值也是我们分析问题的一种方式。常用的指标如下：

- min(a, axis)
  - Return the minimum of an array or minimum along an axis.
- max(a, axis])
  - Return the maximum of an array or maximum along an axis.
- median(a, axis)
  - Compute the median along the specified axis.
- mean(a, axis, dtype)
  - Compute the arithmetic mean along the specified axis.
- std(a, axis, dtype)	
  - Compute the standard deviation along the specified axis.
- var(a, axis, dtype)	
  - Compute the variance along the specified axis.

> var方差是衡量数据点离平均值的平方偏差程度。方差的值总是非负的，方差越大，数据越分散。

> std标准方差是衡量数据点离平均值的平均偏差程度。值越小，数据越集中；值越大，数据越分散。

### 4.2  案例：学生成绩统计运算

```python
# 接下来对于前四名学生,进行一些统计运算
# 指定列 去统计
temp = score[:4, 0:5]
print("前四名学生,各科成绩的最大分：{}".format(np.max(temp, axis=0)))
print("前四名学生,各科成绩的最小分：{}".format(np.min(temp, axis=0)))
print("前四名学生,各科成绩波动情况：{}".format(np.std(temp, axis=0)))
print("前四名学生,各科成绩的平均分：{}".format(np.mean(temp, axis=0)))
```

**axis = 0**: 沿着每一列进行操作，意味着在每一列上进行统计计算。可以理解为“跨行操作”。

**axis = 1**: 沿着每一行进行操作，意味着在每一行上进行统计计算。可以理解为“跨列操作”。

结果：

```
前四名学生,各科成绩的最大分：[96 97 72 98 89]
前四名学生,各科成绩的最小分：[55 57 45 76 77]
前四名学生,各科成绩波动情况：[16.25576821 14.92271758 10.40432602  8.0311892   4.32290412]
前四名学生,各科成绩的平均分：[78.5  75.75 62.5  85.   82.25]
```

如果需要统计出某科最高分对应的是哪个同学？

- np.argmax(temp, axis=)
- np.argmin(temp, axis=)

```python
print("前四名学生，各科成绩最高分对应的学生下标：{}".format(np.argmax(temp, axis=0)))
```

结果：

```
前四名学生，各科成绩最高分对应的学生下标：[0 2 0 0 1]
```

## 5 小结

- 逻辑运算【知道】
  - 直接进行大于,小于的判断
  - 合适之后,可以直接进行赋值
- 通用判断函数【知道】
  - np.all()
  - np.any()
- 统计运算【掌握】
  - np.max()
  - np.min()
  - np.median()
  - np.mean()
  - np.std()
  - np.var()
  - np.argmax(axis=)  — 最大元素对应的下标
  - np.argmin(axis=)  — 最小元素对应的下标

# 五、数组间运算

## 学习目标

- 目标
  - 知道数组与数之间的运算
  - 知道数组与数组之间的运算
  - 说明数组间运算的广播机制

----

## 1 数组与数的运算

```python
arr = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]])
arr + 1
arr / 2

# 可以对比python列表的运算，看出区别 => 列表是整体操作，numpy是每个元素单独操作
a = [1, 2, 3, 4, 5]
a * 3
```

## 2 数组与数组的运算

### 2.1 思考

```python
arr1 = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]])   # 2 x 6
arr2 = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])  # 2 x 4
```

上面这个能进行运算吗，结果是不行的！

### 2.2 广播机制

数组在进行矢量化运算时，**要求数组的形状是相等的**。当形状不相等的数组执行算术运算的时候，就会出现广播机制，该机制会对数组进行扩展，使数组的shape属性值一样，这样，就可以进行矢量化运算了。下面通过一个例子进行说明：

```python
arr1 = np.array([[0],[1],[2],[3]])  # 4 x 1
arr1.shape
# (4, 1)

arr2 = np.array([1,2,3])  # 1 x 3
arr2.shape
# (3,)

arr1+arr2
# 结果是：
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5],
       [4, 5, 6]])
```

上述代码中，数组arr1是4行1列，arr2是1行3列。这两个数组要进行相加，按照广播机制会对数组arr1和arr2都进行扩展，使得数组arr1和arr2都变成4行3列。

下面通过一张图来描述广播机制扩展数组的过程：

![image-20190620005224076](./images/image-20190620005224076.png)



这句话乃是理解广播的核心。广播主要发生在两种情况，一种是两个数组的维数不相等，但是它们的后缘维度的轴长相符，另外一种是有一方的长度为1。





广播机制：数组与数组之间结构不同的情况

> 规则 1：如果数组的维度数不同，那么将维度数较少的数组在前面补充 1，使其维度数与维度数较多的数组一致。

> 规则 2：从最后一个维度开始比较，如果两个数组在该维度上的长度相同，或其中一个数组在该维度的长度为 1，那么它们在该维度上是兼容的，可以进行运算。

> 规则 3：如果在任何一个维度上，两个数组的长度既不同又都不为 1，则它们无法进行广播运算。



如果是下面这样，则不匹配：

```python
A  (1d array): 10
B  (1d array): 12

A  (2d array):         2 x 1      1 x 2 x 1
B  (3d array):  8 x 4 x 3      8 x 4 x 3
```

**思考：下面两个ndarray是否能够进行运算？**

```python
arr1 = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]])   # 2 x 6
arr2 = np.array([[1], [3]])  # 2 
```



## 3 小结

- 数组运算,满足广播机制,就OK【知道】
  - 1.维度相等
  - 2.shape(其中对应的地方为1,也是可以的)
