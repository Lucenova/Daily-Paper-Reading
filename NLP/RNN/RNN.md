##### RNN为什么会产生梯度消失和爆炸

对于一个经典的RNN结构，假设时间序列只有3段，$S_0$为给定的初始值，神经元没有激活函数，RNN最简单的前向传播过程如下：
$$
S_1 = W_xX_1+W_sS_0+b_1 \\O_1 = W_0S_1+b_2 \\S_2 = W_xX_2 + W_sS_1 + b_1 \\O_2 = W_0S_2+b_2 \\S_3 = W_xX_3+W_sS_2+b_1 \\O_3 = W_0S_3+b_2
$$
对于一次训练的任务的损失函数$L = \sum_{t=0}^{T}L_t$，即每一时刻的损失值的累加。

在反向传播过程中，梯度的计算公式如下：
$$
\frac{\partial{L_3}}{\partial{W_0}} = \frac{\partial{L_3}}{\partial{O_3}}\frac{\partial{O_3}}{\partial{W_0}} \\\frac{\partial{L_3}}{\partial{W_x}} = \frac{\partial{L_3}}{\partial{O_3}}\frac{\partial{O_3}}{\partial{S_3}}\frac{\partial{S_3}}{\partial{W_x}} + \frac{\partial{L_3}}{\partial{O_3}}\frac{\partial{O_3}}{\partial{S_3}}\frac{\partial{S_3}}{\partial{S_2}}\frac{\partial{S_2}}{\partial{W_x}} + \frac{\partial{L_3}}{\partial{O_3}}\frac{\partial{O_3}}{\partial{S_3}}\frac{\partial{S_3}}{\partial{S_2}}\frac{\partial{S_2}}{\partial{S_1}}\frac{\partial{S_1}}{\partial{W_x}}\\\frac{\partial{L_3}}{\partial{W_s}} = \frac{\partial{L_3}}{\partial{O_3}}\frac{\partial{O_3}}{\partial{S_3}}\frac{\partial{S_3}}{\partial{W_s}} + \frac{\partial{L_3}}{\partial{O_3}}\frac{\partial{O_3}}{\partial{S_3}}\frac{\partial{S_3}}{\partial{S_2}}\frac{\partial{S_2}}{\partial{W_s}} + \frac{\partial{L_3}}{\partial{O_3}}\frac{\partial{O_3}}{\partial{S_3}}\frac{\partial{S_3}}{\partial{S_2}}\frac{\partial{S_2}}{\partial{S_1}}\frac{\partial{S_1}}{\partial{W_s}}
$$
可以看出对于$W_0$求偏导并没有长期的依赖，但是对于$W_x$和$W_s$求偏导，会随着时间序列产生长期依赖。

从而总结出任意时刻对于$W_x$和$W_s$求偏导的公式：
$$
\frac{\partial{L_t}}{\partial{W_x}}=\sum_{k=0}^{t}\frac{\partial{L_t}}{\partial{O_t}}\frac{\partial{O_t}}{\partial{S_t}}(\prod_{j=k+1}^{t}\frac{\partial{S_j}}{\partial{S_{j-1}}})\frac{\partial{S_k}}{\partial{W_x}}
$$
如果在加上激活函数，则$S_j=tanh(W_xX_j+W_sS_{j-1}+b_1)$，则$\prod^{t}_{j=k+1}\frac{\partial{S_j}}{\partial{S_{j-1}}}=\prod_{j=k+1}^{t}tanh^{\prime}W_s$

对应的$tanh$和其导数的图像如下：

![image-20211124175806728](619f58375653bb136f853f19)

可以看出$tanh^{\prime}\leq1$，所以训练过程大部分情况下导数是小于1的，因为很少情况下会出现$W_xX_j+W_sS_{j-1}+b_1=0$，如果$W_s$是一个大于0小于1的值，当$t$很大时候，$\prod^{t}_{j=k+1}tanh^{\prime}W_s$就会趋近于0，当$W_s$很大的时候，$\prod^{t}_{j=k+1}tanh^{\prime}W_s$就会趋近于无穷，这就是RNN梯度消失和爆炸的原因。

##### LSTM为什么能够缓解梯度爆炸和消失问题

LSTM中有3个门，分别是：**forget gate**，**input gate**，**output gate**，其对应的**系数**计算公式如下：
$$
f_t=\sigma(W_fX_t+b_f)\\i_t=\sigma(W_iX_t+b_i)\\o_i=\sigma(W_oX_t+b_o)
$$
那么当前的状态的计算公式为$S_t=f_tS_{t-1}+i_tX_t=\sigma(W_fX_t+b_f)S_{t-1}+\sigma(W_iX_t+b_i)X_t$

如果再加上激活函数，$S_t=tanh[(\sigma(W_fX_t+b_f)S_{t-1})+\sigma(W_iX_t+b_i)X_t]$

在RNN中引起梯度爆炸或者消失的导数项在LSTM中变成了$\prod_{j=k+1}^{t}\frac{\partial{S_j}}{\partial{S_{j-1}}}=\prod_{j=k+1}^{t}tanh^{\prime}\sigma(W_fX_t+b_f)$

其中$tanh^{\prime}\sigma(W_fX_t+b_f)$的值基本上不是0就是1，这样就解决了RNN中梯度消失的问题。