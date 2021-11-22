##### 什么是Batch Norm，什么是Layer Norm，为什么NLP使用Layer Norm而不使用Batch Norm

##### Batch Norm

Batch Norm一般是在图像处理领域中使用，对于一个batch的feature map来说：$x\in{R^{N*C*H*W}}$，在一个batch中包含了$N$个样本，每个样本的通道数量为$C$，高为$H$，宽为$W$。对这个batch的feature map求均值和方差的时候，将在$N$，$H$，$W$上操作，而保留Channel的$C$的维度。具体来说就是：就是将第1个样本的第一个通道加上第2个样本的第一个通道，直到加到第N个样本的第一个通道，然后求所有样本的第1个通道之和的所有像素的均值，方差也同理，具体公式如下：
$$
\mu_c(x)=\frac{1}{NHW}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{nchw}(1)\\\sigma_c(x)=\sqrt{\frac{1}{NWH}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W}(x_{nchw}-\mu_c(x))^2}
$$
如果把$x\in{R^{N*C*H*W}}$类比一摞书，这摞书总共有$N$本，每本有$C$页，每页有$H$行，每行有$W$个字符，batch norm求均值的时候，就相当于将每本书的相同的页码分别加起来，在除以每个页码的字符的总数$N*W*H$。

##### Layer Norm

对于一个batch的feature map来说，$x\in{R^{N*C*H*W}}$，layer norm相当于保留batch size中的样本，然后对于每个样本$C$，$N$，$W$维度上的数据求均值和标准差。继续采用之前的类比，layer norm相当于把每一本书的所有字都加起来，然后再除以这本书的字符的总数$C*H*W$，即求每本书的平均字，求标准差也是同理。

##### 为什么NLP使用Layer norm

有了上面的介绍，这个问题就很好理解，Batch Norm相当于是求Channel的平均，而layer norm相当于是求样本的平均。举例来说：

如果一个batch中有4句话：

我是中国人我爱中国

武汉抗疫非常成功0

大家好才是真的好0

人工智能很火000

那么对于这个batch做batch norm的时候就相当于把4条文本位置相同的的字做归一化处理，例如：我、武、大、人，这样就破坏了一句话中内在的语义的含义。

而Layer norm则是针对于每一句话做归一化处理。