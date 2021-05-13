#### Direction Derivative

想要理解方向导数就先要理解多元函数的偏导数

##### 偏导数

对于二元函数 $ z = f(x, y) $的偏导数来说，如果是对于x的偏导数，就相当于固定y轴，函数值z相对于x变化量，总结来说**偏导数是函数在每个位置处沿自变量的坐标轴方向上的导数（切线斜率）**

##### 方向导数

如果不是沿自变量坐标轴的方向了，而是沿着自变量所构成空间中的任意方向，对于二元函数来说就是 $xoy$ 这个平面，如果是一个三元函数就是一个 $x y z$ 空间，就是这个3维空间中的任意方向。

明白了方向导数中方向的意义，那么就剩下如何求出该方向的导数

还是利用二元函数来举例：

定义$xoy$平面上的任意一点$ (a, b)$以及表示任意方向的一个单位向量 $ \vec{u}=(cos\theta， sin\theta) $  ，在曲面$ z = f(x,y)$上，从$(a, b, f(a, b))$开始，沿着$ \vec{u} = (cos\theta, sin\theta)$ 方向走$t$单位长度后，函数值$z$变化为$ F(t) = f(a + tcos\theta, b + tsin\theta)$，则点$(a, b)$处沿$\vec{u}$的方向导数为：
$$
\begin{align*}
& \lim\limits_{t\to0}\frac{f(a+tcos\theta, b+tsin\theta)-f(a,b)}{t}\\
&= \lim\limits_{t\to0}\frac{f(a+tcos\theta, b+tsin\theta) - f(a,b+tsin\theta)}{t} + 
\lim\limits_{t\to0}\frac{f(a, b+tcos\theta)-f(a,b)}{t}\\
&=\frac{\partial f(a,b)}{\partial x}*\frac{\mathrm{d}x}{\mathrm{d}t}+\frac{\partial f(a,b)}{\partial y} * \frac{\mathrm{d}y}{\mathrm{d}t}\\
&= f_x(a,b)cos\theta + f_y(a,b)sin\theta\\
&= (f_x(a,b), f_y(a,b))\cdot(cos\theta, sin\theta)
\end{align*}
$$

##### 梯度

从上面的公式就可以看出方向导数是$(f_x(a,b), f_y(a,b))$和$(cos\theta, sin\theta)$两个向量之间的点乘，所以当两个向量的角度为0的时候即同$(f_x(a,b), f_y(a,b))$方向相同，该方向导数最大，即函数的变化率是最大的，因此定义该方向为梯度。

