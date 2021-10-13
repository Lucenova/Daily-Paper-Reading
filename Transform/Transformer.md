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

