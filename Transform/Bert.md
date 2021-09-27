#### Bert模型理解

##### BPE/WordPiece算法

这两种算法其实就是分词算法，对应英文来说每个英文单词有着不同的时态，如果相同的单词的不同时态将其算为不同的单词，那么词表就会变得相当的庞大，所以如果要训练一个bert模型，则需要使用这两种算法生成token的vocab。然后在使用bert模型的时候进行fine-tuning的时候，需要利用bert模型提供的tokenizer将输入sentence进行分割，分割成对应的token，对应中文来说，这两种算法则没有什么意义，中文在训练模型来说仅仅是需要分字或者分词即可。 

BPE算法的流程大体如下：

- 准备一个corpus，确定期望的subword词表的大小
- 将corpus英文单词拆分为字母
- 统计corpus中字母组合的频数，从而不断的组合字母或者子词，从而达到设定的subword的词表的大小

WordPiece算法：同BPE算法的流程基本类似，唯一有区别的地方在于不是根据频数组合子词，而是根据组合之后子词的信息量大小的变化（选取组合之后信息熵减少最多的组合，也就是两个子词之间的互信息最大）作为评价标准。

##### embedding层

###### token embedding层

该层的参数可以随机初始化，或者使用word2vec等算法进行预训练作为初始化之，其对应的输出是文本中各个字/词融合了全文语义信息的向量表示。

###### Segment Embeddings

该层的作用就是用来区分上下句的，因为训练Bert模型的预训练任务往往是NSP，那么则需要标注一下那句话是前一句，哪句话是后一句。所以初始化的时候是根据当前输入句子的index作为对应token的embedding

###### Position Embeddings

该层的作用就是用来对模型输入sentence中token的position信息。

##### Bert中dropout（Todo）

- embedding层
- transformer层
  - 在transformer中计算attention的时候，即根据$Q$和$K$，通过softmax计算之后，会有一个dropout
  - 在transformer提取出attention feature之后，

##### Bert为什么会有效

reference paper: https://arxiv.org/pdf/1908.05620.pdf

论文通过visualizing的方法，证明了Bert模型为什么有效。

- visualization results indicate that Bert pre-training reaches **a good initial point** across downstream tasks
- pre-training obtains more **flat and wider optima**
- the lower layers of Bert are more invariant across tasks than the higher layers, which suggest that the lower layers learn **transferable representations of language**