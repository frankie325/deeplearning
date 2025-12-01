# Matplotlib

# 一、Matplotlib快速入门

## 学习目标

- 了解什么是matplotlib
- 为什么要学习matplotlib
- matplotlib简单图形的绘制

## 什么是Matplotlib

![image-20240902053742408](./images/image-20240902053742408.png)

- 是专门用于开发2D图表(包括3D图表)

- 以渐进、交互式方式实现数据可视化

## 为什么要学习Matplotlib

可视化是在整个数据挖掘的关键辅助工具，可以清晰的理解数据，从而调整我们的分析方法。

- 能将数据进行可视化,更直观的呈现
- 使数据更加客观、更具说服力

例如下面两个图为数字展示和图形展示：

![image-20240902053758921](./images/image-20240902053758921.png)

## 实现一个简单的Matplotlib画图 — 以折线图为例

### matplotlib.pyplot模块

matplotlib.pytplot包含了一系列类似于matlab的画图函数。

```python
import matplotlib.pyplot as plt
```

### 图形绘制流程：

- 1.创建画布 -- plt.figure()

- ```
  plt.figure(figsize=(), dpi=)
      figsize:指定图的长宽
      dpi:图像的清晰度
      返回fig对象
  ```

- 2.绘制图像 -- plt.plot(x, y)

- ```
  以折线图为例
  ```

- 3.显示图像 -- plt.show()

### 折线图绘制与显示

**举例：展现上海一周的天气,比如从星期一到星期日的天气温度如下**

```python
import matplotlib.pyplot as plt

# 1.创建画布
plt.figure(figsize=(10, 10), dpi=100)

# 2.绘制折线图
plt.plot([1, 2, 3, 4, 5, 6 ,7], [17,17,18,15,11,11,13])

# 3.显示图像
plt.show()
```

![image-20240902053843435](./images/image-20240902053843435.png)

## 认识Matplotlib图像结构(了解)

![image-20240902053935190](./images/image-20240902053935190.png)

## 小结

- 什么是matplotlib【了解】
  - 是专门用于开发2D(3D)图表的包
- 绘制图像流程【掌握】
  - 1.创建画布 -- plt.figure(figsize=(20,8), dpi=100)
  - 2.绘制图像 -- plt.plot(x, y)
  - 3.显示图像 -- plt.show()

# 二、Matplotlib基础绘图功能

## 学习目标

* 掌握给图形添加辅助功能(如：标注、x,y轴名称、标题等)

- 知道图形的保存
- 知道如何多次plot绘制图形
- 知道如何多个坐标系显示图形
- 知道折线图的应用场景

## 完善原始折线图 — 给图形添加辅助功能

为了更好地理解所有基础绘图功能，我们通过天气温度变化的绘图来融合所有的基础API使用

**需求：画出某城市11点到12点1小时内每分钟的温度变化折线图，温度范围在15度~18度**

效果：

![image-20240902054526938](./images/image-20240902054526938.png)

### 准备数据并画出初始折线图

```python
import matplotlib.pyplot as plt
import random

# 画出温度变化图

# 0.准备x, y坐标的数据
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]

# 1.创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 2.绘制折线图
plt.plot(x, y_shanghai)

# 3.显示图像
plt.show()
```

> **作用**：返回一个 **随机的浮点数**，其值在区间 `[a, b]` 之间（包括 `a` 和 `b`）。
>
> **分布**：数值在指定区间内是 **均匀分布** 的，这意味着每个数值出现的概率是相等的。

![image-20240902054833670](./images/image-20240902054833670.png)

### 添加自定义x,y刻度

- plt.xticks(x, **kwargs)

  x:要显示的刻度值

- plt.yticks(y, **kwargs)

  y:要显示的刻度值

```python
# 增加以下两行代码

# 构造x轴刻度标签
x_ticks_label = ["11点{}分".format(i) for i in x]
# 构造y轴刻度
y_ticks = range(40)

# 修改x,y轴坐标的刻度显示
plt.xticks(x[::5], x_ticks_label[::5])
plt.yticks(y_ticks[::5])
```

![image-20240902060239499](./images/image-20240902060239499.png)

如果没有解决过中文问题的话，会显示这个样子：

![image-20240902060218522](./images/image-20240902060218522.png)

### 中文显示问题解决

**解决方案一：**

下载中文字体（黑体，看准系统版本）

- 步骤一：下载 [SimHei](images/SimHei.ttf) 字体（或者其他的支持中文显示的字体也行）

- 步骤二：安装字体

  - linux下：拷贝字体到 usr/share/fonts 下：

    ```
    sudo cp ~/SimHei.ttf /usr/share/fonts/SimHei.ttf
    ```

  - windows和mac下：双击安装

- 步骤三：删除~/.matplotlib中的缓存文件

  ```python
  cd ~/.matplotlib
  rm -r *
  ```

- 步骤四：修改配置文件matplotlibrc

  ```python
  vi ~/.matplotlib/matplotlibrc
  ```

  将文件内容修改为：

  ```
  font.family         : sans-serif
  font.sans-serif         : SimHei
  axes.unicode_minus  : False
  ```

**解决方案二：**

在Python脚本中动态设置matplotlibrc,这样也可以避免由于更改配置文件而造成的麻烦，具体代码如下：

```python
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
```

有时候，字体更改后，会导致坐标轴中的部分字符无法正常显示，此时需要更改axes.unicode_minus参数：

```python
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
```

### 添加网格显示

为了更加清楚地观察图形对应的值

```python
plt.grid(True, linestyle='--', alpha=0.5)
```

![image-20240902060406346](./images/image-20240902060406346.png)

### 添加描述信息

添加x轴、y轴描述信息及标题

> 通过fontsize参数可以修改图像中字体的大小

```python
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("中午11点0分到12点之间的温度变化图示", fontsize=20)
```

![image-20240902060626347](./images/image-20240902060626347.png)

### 图片保存

```python
# 保存图片到指定路径
plt.savefig("test.png")
```

- 注意：plt.show()会释放figure资源，如果在显示图像之后保存图片将只能保存空图片。

### 完整代码

```python
import matplotlib.pyplot as plt
import random
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 0.准备数据
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]

# 1.创建画布
plt.figure(figsize=(20, 8), dpi=100)

# 2.绘制图像
plt.plot(x, y_shanghai)

# 2.1 添加x,y轴刻度
# 构造x,y轴刻度标签
x_ticks_label = ["11点{}分".format(i) for i in x]
y_ticks = range(40)

# 刻度显示
plt.xticks(x[::5], x_ticks_label[::5])
plt.yticks(y_ticks[::5])

# 2.2 添加网格显示
plt.grid(True, linestyle="--", alpha=0.5)

# 2.3 添加描述信息
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("中午11点--12点某城市温度变化图", fontsize=20)

# 2.4 图像保存
plt.savefig("./test.png")

# 3.图像显示
plt.show()
```

## 在一个坐标系中绘制多个图像

### 多次plot

需求：再添加一个城市的温度变化

收集到北京当天温度变化情况，温度在1度到3度。怎么去添加另一个在同一坐标系当中的不同图形，**其实很简单只需要再次plot即可**，但是需要区分线条，如下显示

![image-20240902060840535](./images/image-20240902060840535-1725228522315-1.png)

```python
# 增加北京的温度数据
y_beijing = [random.uniform(1, 3) for i in x]

# 绘制折线图
plt.plot(x, y_shanghai)
# 使用多次plot可以画多个折线
plt.plot(x, y_beijing, color='r', linestyle='--')
```

我们仔细观察，用到了两个新的地方，一个是对于不同的折线展示效果，一个是添加图例。

### 设置图形风格

| 颜色字符 |    风格字符    |
| :------: | :------------: |
|  r 红色  |     - 实线     |
|  g 绿色  |    - - 虚线    |
|  b 蓝色  |   -. 点划线    |
|  w 白色  |    : 点虚线    |
|  c 青色  | ' ' 留空、空格 |
|  m 洋红  |                |
|  y 黄色  |                |
|  k 黑色  |                |

### **显示图例**

- 注意：如果只在plt.plot()中设置label还不能最终显示出图例，还需要通过plt.legend()将图例显示出来。

```python
# 绘制折线图
plt.plot(x, y_shanghai, label="上海")
# 使用多次plot可以画多个折线
plt.plot(x, y_beijing, color='r', linestyle='--', label="北京")

# 显示图例
plt.legend(loc="best")
```

| Location String | Location Code |
| --------------- | ------------- |
| 'best'          | 0             |
| 'upper right'   | 1             |
| 'upper left'    | 2             |
| 'lower left'    | 3             |
| 'lower right'   | 4             |
| 'right'         | 5             |
| 'center left'   | 6             |
| 'center right'  | 7             |
| 'lower center'  | 8             |
| 'upper center'  | 9             |
| 'center'        | 10            |

###  完整代码

```python
# 0.准备数据
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1,3) for i in x]

# 1.创建画布
plt.figure(figsize=(20, 8), dpi=100)

# 2.绘制图像
plt.plot(x, y_shanghai, label="上海")
plt.plot(x, y_beijing, color="r", linestyle="--", label="北京")

# 2.1 添加x,y轴刻度
# 构造x,y轴刻度标签
x_ticks_label = ["11点{}分".format(i) for i in x]
y_ticks = range(40)

# 刻度显示
plt.xticks(x[::5], x_ticks_label[::5])
plt.yticks(y_ticks[::5])

# 2.2 添加网格显示
plt.grid(True, linestyle="--", alpha=0.5)

# 2.3 添加描述信息
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("中午11点--12点某城市温度变化图", fontsize=20)

# 2.4 图像保存
plt.savefig("./test.png")

# 2.5 添加图例
plt.legend(loc=0)


# 3.图像显示
plt.show()
```

## 多个坐标系显示— plt.subplots(面向对象的画图方法)

如果我们想要将上海和北京的天气图显示在同一个图的不同坐标系当中，效果如下：

![image-20240902061104339](./images/image-20240902061104339.png)

可以通过subplots函数实现(旧的版本中有subplot，使用起来不方便)，推荐subplots函数

- matplotlib.pyplot.subplots(nrows=1, ncols=1, **fig_kw) 创建一个带有多个axes(坐标系/绘图区)的图

  ```
  Parameters:    
  
  nrows, ncols : 设置有几行几列坐标系
      int, optional, default: 1, Number of rows/columns of the subplot grid.
  
  Returns:    
  fig : 图对象
  axes : 返回相应数量的坐标系
  
  设置标题等方法不同：
      set_xticks
      set_yticks
      set_xlabel
      set_ylabel
  ```

  关于axes子坐标系的更多方法：参考https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

- 注意：**plt.函数名()**相当于面向过程的画图方法，**axes.set_方法名()**相当于面向对象的画图方法。

```python
    # 0.准备数据
    x = range(60)
    y_shanghai = [random.uniform(15, 18) for i in x]
    y_beijing = [random.uniform(1, 5) for i in x]

    # 1.创建画布
    # plt.figure(figsize=(20, 8), dpi=100)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi=100)


    # 2.绘制图像
    # plt.plot(x, y_shanghai, label="上海")
    # plt.plot(x, y_beijing, color="r", linestyle="--", label="北京")
    axes[0].plot(x, y_shanghai, label="上海")
    axes[1].plot(x, y_beijing, color="r", linestyle="--", label="北京")

    # 2.1 添加x,y轴刻度
    # 构造x,y轴刻度标签
    x_ticks_label = ["11点{}分".format(i) for i in x]
    y_ticks = range(40)

    # 刻度显示
    # plt.xticks(x[::5], x_ticks_label[::5])
    # plt.yticks(y_ticks[::5])
    axes[0].set_xticks(x[::5])
    axes[0].set_yticks(y_ticks[::5])
    axes[0].set_xticklabels(x_ticks_label[::5])
    axes[1].set_xticks(x[::5])
    axes[1].set_yticks(y_ticks[::5])
    axes[1].set_xticklabels(x_ticks_label[::5])

    # 2.2 添加网格显示
    # plt.grid(True, linestyle="--", alpha=0.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 2.3 添加描述信息
    # plt.xlabel("时间")
    # plt.ylabel("温度")
    # plt.title("中午11点--12点某城市温度变化图", fontsize=20)
    axes[0].set_xlabel("时间")
    axes[0].set_ylabel("温度")
    axes[0].set_title("中午11点--12点某城市温度变化图", fontsize=20)
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel("温度")
    axes[1].set_title("中午11点--12点某城市温度变化图", fontsize=20)

    # # 2.4 图像保存
    plt.savefig("./test.png")

    # # 2.5 添加图例
    # plt.legend(loc=0)
    axes[0].legend(loc=0)
    axes[1].legend(loc=0)


    # 3.图像显示
    plt.show()
```

## 折线图的应用场景

- 呈现公司产品(不同区域)每天活跃用户数

- 呈现app每天下载数量

- 呈现产品新功能上线后,用户点击次数随时间的变化

- 拓展：**画各种数学函数图像**

  - 注意：plt.plot()除了可以画折线图，也可以用于画各种数学函数图像

   ![image-20240902061313324](./images/image-20240902061313324.png)

代码：

```python
import numpy as np
# 0.准备数据
x = np.linspace(-10, 10, 1000)
y = np.sin(x)

# 1.创建画布
plt.figure(figsize=(20, 8), dpi=100)

# 2.绘制函数图像
plt.plot(x, y)
# 2.1 添加网格显示
plt.grid()

# 3.显示图像
plt.show()
```

# 三、常见图形绘制

## 官方案例库

https://matplotlib.org/stable/gallery/index.html

![image-20240902061545668](./images/image-20240902061545668.png)

## 常见图形种类及意义

###  折线图

- **折线图**：以折线的上升或下降来表示统计数量的增减变化的统计图

  **特点：能够显示数据的变化趋势，反映事物的变化情况。(变化)**

  api：plt.plot(x, y)

![image-20240902061634477](./images/image-20240902061634477.png)

### 柱形图

**柱状图：**排列在工作表的列或行中的数据可以绘制到柱状图中。

**特点：绘制连离散的数据,能够一眼看出各个数据的大小,比较数据之间的差别。(统计/对比)**

api：plt.bar(x, width, align='center', **kwargs)

```
Parameters:    
x : 需要传递的数据

width : 柱状图的宽度

align : 每个柱状图的位置对齐方式
    {'center', 'edge'}, optional, default: 'center'

**kwargs :
color:选择柱状图的颜色
```

![image-20240902061718147](./images/image-20240902061718147.png)

### 直方图

由一系列高度不等的纵向条纹或线段表示数据分布的情况。 一般用横轴表示数据范围，纵轴表示分布情况。

**特点：绘制连续性的数据展示一组或者多组数据的分布状况(统计)**

api：matplotlib.pyplot.hist(x, bins=None)

```
Parameters:    
x : 需要传递的数据
bins : 组距
```

![image-20240902061837887](./images/image-20240902061837887.png)

### 饼图

- **饼图：**用于表示不同分类的占比情况，通过弧度大小来对比各种分类。

  **特点：分类数据的占比情况(占比)**

  api：plt.pie(x, labels=,autopct=,colors)

  ```
  Parameters:  
  x:数量，自动算百分比
  labels:每部分名称
  autopct:占比显示指定%1.2f%%
  colors:每部分颜色
  ```

![image-20240902061901342](./images/image-20240902061901342.png)

### 散点图

**散点图：**用两组数据构成多个坐标点，考察坐标点的分布,判断两变量之间是否存在某种关联或总结坐标点的分布模式。

**特点：判断变量之间是否存在数量关联趋势,展示离群点(分布规律)**

api：plt.scatter(x, y)

![image-20240902061919118](./images/image-20240902061919118.png)

## 柱形图绘制

```python
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 4]

plt.bar(categories, values, color='blue')  # 绘制蓝色柱状图
plt.title("Simple Bar Chart")  # 设置图表标题
plt.xlabel("Categories")  # 设置X轴标签
plt.ylabel("Values")  # 设置Y轴标签
plt.show()  # 显示图表
```

## 直方图

```python
# 1. 生成数据
data = np.random.randn(500)  # 生成500个服从标准正态分布的随机数

# 2. 创建画布
plt.figure(figsize=(10, 6))

# 3. 绘制直方图
plt.hist(data, bins=30, color='blue', alpha=0.7, rwidth=0.85)

# 4. 添加标题和标签
plt.title("Histogram of Normally Distributed Data")
plt.xlabel("Value")
plt.ylabel("Frequency")

# 5. 显示图像
plt.grid(True)
plt.show()

参数说明
bins=30：将数据分成 30 个等宽的区间，每个区间的宽度相同，统计每个区间内数据的频率。
color='blue'：设置直方图条形为蓝色。
alpha=0.7：设置条形的透明度为 0.7，使得条形稍微透明。
rwidth=0.85：将条形的宽度设为区间宽度的 85%，留下 15% 的间隔，以便更清晰地分辨各个条形。
```

## 饼图

```python
sizes = [25, 35, 25, 15]
labels = ['Category A', 'Category B', 'Category C', 'Category D']

plt.pie(sizes, labels=labels, autopct='%1.1f%%')  # 绘制饼图，显示百分比
plt.title("Simple Pie Chart")  # 设置图表标题
plt.show()  # 显示图表
```

## 散点图绘制

```python
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.scatter(x, y, color='red')  # 绘制红色散点图
plt.title("Simple Scatter Plot")  # 设置图表标题
plt.xlabel("X-axis")  # 设置X轴标签
plt.ylabel("Y-axis")  # 设置Y轴标签
plt.show()  # 显示图表
```

