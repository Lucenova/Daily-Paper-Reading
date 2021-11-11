#### 为什么Transformer需要进行Multi-head Attention

多头注意力机制就是一种ensemble，使用多头注意力，能够综合捕捉句子中各方面的信息。Multi-head不是必须的，去掉一些仍然有不错的效果，在头足够的情况下，Multi-head已经能够有关注的句子中不同方面的信息了。

- 在论文中发现，比较一个Layer中的transformer的multi-head中所对应的隐变量$h_i$之间的距离发现：不同的$h_i$之间的方差随着所在层的增大而减少。
- Transformer底层的head方差大是因为Transformer存在梯度消失问题，也就是说并不是底层的方差大是好的，而是模型在训练过程中无法让它变好，所以合理的初始化应该减少底层头的方差，提高效果。

#### Transformer为什么$Q$和$K$使用不同的权重矩阵生成，为什么不能使用同一个值进行自身的点乘？

首先明确一下$QK^T$点积的具体的意义是求矩阵$Q$的行向量同矩阵$K^T$行向量的进行相似度的计算，两个矩阵的行向量越相近那么得到的乘积的结果就越大。

已经$Q$和$K$是用来计算注意力的，公式如下：

$$ Attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt{d_k}})V $$

如果$Q$和$K$相同那么$QK^T$的对应的结果的矩阵中的元素会是对角线中的元素值是最大的，剩下的元素值会比较小，在通过$softmax$的计算，那么所得到的结果就会是接近一个单位矩阵，在乘以$V$那么结果也不会变化，这就失去了attention的意义了。

#### 为什么进行softmax之前需要对attention进行scaled（为什么除以$d_k$的平方根），并使用公式推导

- 首先明确下softmax的性质，就是如果输入softmax的向量的值相差越大（可能性最大的值很大，其他可能性的值很小），那么softmax的计算结果就会很接近于一个单位向量。

- softmax函数的导数同softmax的输出结果相关，如果输出的结果是一个单位向量，那么softmax的导数就会是为0，这就会导致梯度消失，所以要对attention进行scaled

reference url： https://www.zhihu.com/question/339723385/answer/782509914

##### Transformer中的Position embedding详解

已经在计算self-attention中是没有考虑到word中的position信息的，那么如何实现position embedding。一般直觉则是第一个word对应为1，第二个word为2，以此类推。但是这样就会有如下的问题：

- 数据集中如果会有特别长的sentence，这时就会造成输入的数据过大，model难以训练
- model在train的过程中，数据集中不可能包括所有长度的样本，这样就会影响model的泛化能力

基于此，我们可以将word的position embedding所对应的值限制在$[0, 1]$之间，但是这样也会有一个比较重要的问题就是对于不同长度的sentence，相同position的word所对应的embedding的值是不同的。

所以一个好的位置编码方案需要满足以下几个要求：

- 能为每个时间步（也就是每个位置的word）输出独一无二的编码
- 不同句子之间，任何两个时间步之间的距离应该保持一致
- 模型应该能毫不费力地泛化到更长的sentence，它所对应的值应该是有界的
- embedding的值必须是确定性的

Transformer模型中position embedding计算方式如下：

$$\vec{p}^{(i)}_{t} := \begin{equation}\left\{\begin{array}{lr}sin(w_k\cdot{t}), & if & i=2k\\cos(w_k\cdot{t}), & if & i=2k+1\end{array}\right.\end{equation}$$

其中频率的$w_k$定义如下：

$$w_k = \frac{1}{10000^{2k/d}}$$

从定义可知，频率沿着向量的维度减少。

$$\vec{p} = [sin(w_1\cdot{t}), cos(w_2\cdot{t}), \cdots, sin(w_{d/2}\cdot{t}),cos(w_{d/2}\cdot{t})]$$

正弦曲线函数的位置编码的另外一个特点是，它能让模型毫不费力地关注相对位置信息，对于任意固定的偏移量$k$，$PE_{pos+k}$可以表示成$PE_{pos}$的线性函数。

但是传统的Transformer使用的绝对位置信息的embedding有一定的问题，如下如所示：

![image-20211020102613098](C:\Users\yue.zhu\AppData\Roaming\Typora\typora-user-images\image-20211020102613098.png)

可以看到当横坐标代表position embedding的维度从大到小，纵坐标的代表position embedding的值不同位置的差异性也越来越模糊。

![image-20211020105514136](C:\Users\yue.zhu\AppData\Roaming\Typora\typora-user-images\image-20211020105514136.png)

其次就是虽然$PE_{pos+k}$可以表示为$PE_{pos}$，也就是点积可以反映相对距离，但是它缺乏方向性，并且这种特性（相对距离）会被原始的Transformer的注意力机制给破坏。大致证明如下：

把词向量和位置向量作为输入，经过注意力层，然后因式分解会有四个部分，重点关注两个不同位置编码的公式部分，形式如下：

$$PE^{T}_{pos}W^T_{q}W_{k}PE_{pos+k}$$

即问题就转化为上面那个公式能否反应相对位置信息，先从下面这个公式入手：

$$PE^{T}_{pos}PE_{pos+k}$$

这个公式和上面因式分解的部分公式仅仅是中间多了两个矩阵相乘，可以看做是一个线性变化。

剩下证明参考：https://mp.weixin.qq.com/s/vXYJKF9AViKnd0tbuhMWgQ

##### Transformer中的relative position representations

在RNN模型中是按照时间序列的顺序input产生hidden state，所以hidden state中带有input的位置信息。在transformer中使用的是absolute position representations，虽然absolute position representations也能表示input的relative postition information，但是其表示的relative position information没有方向性，所以基于此有两种在transformer中使用了relative position representation的方案

- **Self-Attention with Relative Position Representations**

- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**

##### Self-Attention with Relative Position Representations

本篇论文提出了在transformer模型中加入了可训练的relative postition embedding，然后embedding matrix中的vectors加入到计算input中的$i$和$j$之间的attention weight和attention value中，用公式表示如下：

$$z_i=\sum\alpha_{i,j}(x_jW^{V}+a_{i,j}^{V})$$

$$e_{i,j}=\frac{x_iW^Q(x_jW^K+a_{i,j}^K)^T}{\sqrt{d_z}}$$

需要注意的是公式中的$a_{i,j}^{V}$和$a_{i,j}^{K}$是两个不同relative postition embedding matrix中的embedding vector。

下面介绍一下使用的relative position embedding matrix中的vector的含义：

举例来说，当输入的句子input为length=5的sentence的（I think therefore I am），需要学习的embedding matrix的行数就为9（index=4表示的是当前单词的位置信息，index=0到3表示当前单词的左边的单词，index=5到9就是右边的单词）。

需要注意的是，对于一个比较长的句子，如果计算$i$和$j$的相关性的时候，如果$i$和$j$相距的距离较远，那么在使用embedding matrix中的vector的时候就会有clipped，其实就是作者认为相对位置编码再超过了一定距离之后是没有必要的，并且clip最大距离可以使模型的泛化效果更好，可以更好的generalize到训练过程中没有出现过的序列长度上。



