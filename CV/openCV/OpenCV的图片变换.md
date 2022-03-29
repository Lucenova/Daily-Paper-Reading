#### OpenCV的图片变换

##### 几何变换

- 图像平移

  ```python
  import cv2
  
  img = cv2.imread('img2.png')
  # 构造移动矩阵H
  # 在x轴方向移动的距离，在y轴方向移动的距离
  H = np.float32([[1, 0, 50], [0, 1, 25]])
  rows, cols = img.shape[:2]
  
  res = cv2.warpAffine(img, H, [cols, rows])
  
  """
  仿射变换函数 cv2.warpAffine(src, M, dsize, flags, borderMode, borderValue)
  其中：
  M:变换矩阵，一般反应平移或者旋转的关系，为inputArray类型的2*3变换矩阵
  dsize:输出图片的大小
  flags:插值方法的组合（int类型），默认为flags=cv2.INTER_LINEAR表示线性插值
  此外还有：
  cv2.INTER_NEAREST(最邻近插值)
  cv2.INTER_AREA(区域插值)
  cv2.INTER_CUBIC(三次样条插值)
  """
  ```

- 图像缩放

  ```python
  import cv2
  import numpy as np
  
  img = cv2.imread('img2.png')
  res = cv2.resize(img, (int(0.8*width), int(0.8*height)), interpolation=cv2.INTER_AREA)
  ```

  ##### 图像插值方法：

  - 最近邻插值

    最简单一种插值方法，不需要计算，在待求像素的四邻像素中，将距离待求像素最近的邻像素的灰度赋给待求像素。计算公式如下：
    $$
    srcX = dstX * (srcWidht / dstWidth) \\
    srcY = dstY * (srcHeight / dstHeight)
    $$
    <img src="/Users/zhuyue/Code/Python/collected-papers/CV/openCV/image-20220326221812024.png" alt="image-20220326221812024" style="zoom:50%;" />

  - 双线性插值

    双线性插值又叫一阶插值法，它要经过三次插值才能获得最终的结果。先对于两水平方向进行一阶线性插值，然后再垂直方向进行一阶线性插值。
    $$
    \frac{y-y_0}{x-x_0} = \frac{y_1-y_0}{x_1-x_0}\\
    y=\frac{x_1-x}{x_1-x_0}y_0+\frac{x-x_0}{x_1-x_0}y_1
    $$
    <img src="/Users/zhuyue/Code/Python/collected-papers/CV/openCV/image-20220326223144676.png" alt="image-20220326223144676" style="zoom:50%;" />

​	 	

- 图像旋转

  ```python
  import cv2
  import numpy as np
  
  img = cv2.imread('img.png', 1)
  rows, cols = img.shape[:2]
  # 参数1: 旋转中心，参数2: 旋转角度， 参数3:缩放因子
  # 参数3正的时候为逆时针，负值为顺时针
  M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, -1)
  
  dst=cv2.warpAffline(img, M, (cols, rows))
  ```

  

- 仿射变换

  就是平移、旋转、放缩、剪切这几种变换的组合

  ```python
  import cv2
  import numpy as np
  
  img = cv2.imread('bird.png')
  rows, cols = src.shape[:2]
  # pos1变换前的位置，pos2b变换
  pos1=np.float32([[50, 50], [200, 50], [50, 200]])
  pos2=np.float32([[10, 100], [200, 50], [100, 250]])
  
  M = cv2.getAffineTransform(pos1, pos2)
  result = cv2.warpAffine(src, M, (cols, rows))
  ```

- 透视变换

  本质上讲图像投影到一个新的视平面

  ```python
  import cv2
  rows, cols = src.shape[:2]
  
  pos1=np.float32([[114, 82], [287, 156], [8, 100], [143, 177]])
  pos2=np.float32([[0,0], [188,0], [0. 262], [188, 262]])
  
  M = cv2.getPerspectiveTransformer(pos1, pos2)
  
  result = cv2.warpPerspective(src, M, (cols, rows))
  ```

##### 图像滤波

滤波实际上是信号处理的一个概念，图像可以看作为一个二维信号，其中像素点的灰度值代表信号的强弱。信号中高频带遍图像上变化剧烈的部分，低频则是对应图片灰度值变化缓慢，平坦的地方，根据图像的高低频，设置高通和低通滤波器，高通滤波器可以检测变化尖锐明显的地方 ，低通滤波器可以让图片变得平滑，消除噪声，所以高通滤波器用于**边缘检测**，低通滤波器用于图像的**平滑去噪**。

##### 线性滤波

- 方框滤波

  方框的滤波的滤波核：

  <img src="/Users/zhuyue/Code/Python/collected-papers/CV/openCV/image-20220327220311160.png" alt="image-20220327220311160" style="zoom:50%;" />

  ```python
  import cv2
  import python	
  
  img = cv2.imread('girl2.png', cv2.IMREAD_UNCHANGED)
  # cv2.boxFilter(src, depth, ksize, normalize)
  r = cv2.boxFilter(img, -1, (3, 3), normalize=1)
  d = cv2.boxFilter(img, -1, (3, 3), normalize=0)
  
  ```

- 均值滤波

  从卷积核就能看出来就是利用卷积求平均

  ![image-20220327221000422](/Users/zhuyue/Code/Python/collected-papers/CV/openCV/image-20220327221000422.png)

  ```python
  import cv2
  import numpy as np
  
  img = cv2.read('image/opencv.png')
  # cv2.blur(img, ksize)
  blur = cv2.blur(img, (7,7))
  ```

- 高斯滤波

  高斯滤波是一种线性平滑滤波，使用于消除高斯噪声，广泛应用于图像处理的减噪过程。高斯滤波的卷积核权重并不相同，中间像素点的权重最高，越离中心的像素权重越小，就是一个2维高斯函数。

  ```python
  import cv2
  import numpy as np
  
  img = cv2.imread('image/opencv.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # cv2.GuassianBlur(src, ksize, std)
  # std：平滑时，调整std实际上再调整周围像素对当前像素的影响的程度，调大std即提高了远处像素对中心像素的影响程度，滤波结果就越平滑
  blur = cv2.GaussianBlur(img, (5,5), 0)
  ```

##### 非线性滤波

- 中值滤波

  中值滤波是一种非线性滤波，使用像素点领域灰度值的中值代替该点的灰度值，中值滤波可以去除椒盐噪声和斑点噪声

  ```python
  import cv2
  import numpy as np
  
  img = cv2.imread('image/median.png')
  median = cv2.medianBlur(img, 5)
  ```

  

- 双边滤波

  双边滤波是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折中处理，同时考虑空间与信息和灰度相似性，达到保边去噪的目的，具

  有简单、非迭代、局部处理的特点。

  ```python
  import cv2
  import numpy as np
  
  img = cv2.imread('image/opencv.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  blur = cv2.bilaterFilter(img, -1, 15, 10)
  ```

  

##### 图像增强

##### 直方图均衡化

直方图均衡化是将原图通过某种变换，得到一幅灰度直方图为均匀分布的新图像的方法。

直方图均衡化方法的基本思想是对在图像中像素个数多的灰度级进行展宽，而对像素个数少的灰度级进行缩减，从而达到清晰图像的目的。

```python
import cv2
# 灰度图像均值化
img = cv2.imread('dark.jpg', 0)
img_equal = cv2.equalizeHist(img)

# 彩色图像均值化，需要对每一个通道均值化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

result = cv2.merge((bH, gH, rH))
```

##### Gamma变换

Gamma变换是对输入图像灰度值进行的非线性操作，使输出图像灰度值通输入图像的灰度值成指数关系
$$
V_{out} = AV_{in}^r
$$
Gamma变换就是用来图像增强，其提升了暗部细节， 通过非线性变换，让图像从暴光强度的线性响应变得更接近人眼感受的响应，将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正。

#### 图像形态学

形态学，是图像处理中应用最为广泛的技术之一，主要用于从图像中提取对表达和描绘区域形状有意义的图像分量，使后续的识别工作能够抓住目标对象最为本质的形状特征，如边界和连通区域等。

结构元素： 设有两幅图像B，X。 若X是被处理的对象，而B是用来处理X的，则称B为结构元素(structure element)，又被形象地称做刷子。 结构元素通常都是一些比较小的图像。

##### 腐蚀和膨胀

图像的膨胀（Dilation）和腐蚀（Erosion）是两种基本的形态学 运算，其中膨胀类似于“领域扩张”，将图像中的白色部分进行 扩张，其运行结果图比原图的白色区域更大；腐蚀类似于“领域 被蚕食”，将图像中白色部分进行缩减细化，其运行结果图比原 图的白色区域更小。

从数学计算上可以更好的理解：结构元B与图像A进行卷积计算，扫描图像中的每一个像素点，用结构元的元素与二值话图像元素做“与”运算，比如结构元有3个元素，对应的与运算结构都为0，那么目标的像素点为0，否则为1，这就是膨胀。如果利用结构元与图像做卷积，得出B覆盖区域的像素点最小值，并用这个最小值来替代参考点的像素值，**即只有当卷积核对应的元素值均为1的时候，结果的像素值才设为1，否则为0**这就是腐蚀。

```python
import cv2	

img = cv2.imread('test.png')
# src原图像 elment腐蚀操作的内核 anchor默认为(-1, -1)，内核的中心店 iterations腐蚀的次数
erode_res = cv2.imread(src, element, anchor, iterations)


```

##### 开运算

开运算 = 先腐蚀运算，再膨胀运算（看 上去把细微连在一起的两块目标分开了）

开运算总结：

1. 开运算能够除去孤立的小点，毛刺和小桥，而总的位置和形状不变。 
2. 开运算是一个基于几何运算的滤波器。 
3. 结构元素大小的不同将导致滤波效果的不同。 
4. 不同的结构元素的选择导致了不同的分割，即提取出不同的特征。

##### 闭运算

闭运算 = 先膨胀运算，再腐蚀运算（看上 去将两个细微连接的图块封闭在一起），

闭运算总结：

1. 闭运算能够填平小湖（即小孔），弥合小裂缝，而总的位置和形状不变。
2.  闭运算是通过填充图像的凹角来滤波图像的。
3.  结构元素大小的不同将导致滤波效果的不同。
4.   不同结构元素的选择导致了不同的分割。

##### 形态学梯度

基础梯度：基础梯度是用膨胀后的图像减去腐蚀后的图像得到差值图像，也是opencv中 支持的计算形态学梯度的方法，而此方法得到梯度有称为基本梯度。

内部梯度：是用原图像减去腐蚀之后的图像得到差值图像，称为图像的内部梯度。

外部梯度：图像膨胀之后再减去原来的图像得到的差值图像，称为图像的外部梯度。

##### 顶帽和黑帽

顶帽（Top Hat）： 原图像与开运算图的区别（差值），突出原图像中比周围亮的区域

黑帽（Black Hat）： 闭操作图像 - 原图像,突出原图像中比周围暗的区域