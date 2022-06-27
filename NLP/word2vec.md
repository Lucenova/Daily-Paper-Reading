#### 文本表示

##### Bag of words model

##### TF-IDF

**字词的重要性随着它在文件中出现的次数成正比增加，但是同时会随着它在语料库中的出现的频率成反比下降，一个词语在一篇文章中出现次数越多，同时在所有文档中出现的次数越少，则越能够代表该文章。**

TF: Term Frequency IDF: Inverse Document Frequency
$$
TF=\frac{某个词在文章中出现的次数}{文章的总次数} \\
IDF=log\frac{语料库的文档总数}{包含该词的文档数+1} \\
TF-IDF=词频（TF）*逆文档频率（IDF）
$$
代码使用：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese", "Dalian Chinese"]

tv = TfidfVectorizer()
tv_fit = tv.fit_transform(train)
```

##### word2vec

cbow和skip_gram实现

```python
import torch
import torch.nn as nn

from torchtext.vocab import build_vocab_from_iterator

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_norm):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_norm=self.max_norm
        )

        self.linear = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.vocab_size
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

class skip_gram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_norm):
        super(skip_gram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_norm=self.max_norm
        )
        self.linear = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.vocab_size
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)

        return x
    
def build_vocab(data_iter, tokenizer):
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=50
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab   

def collate_cbow(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_token_ids = text_pipeline(text)

        if len(text_token_ids) < 4 * 2 + 1:
            continue

        text_token_ids = text_token_ids[:256]

        for idx in range(len(text_token_ids) - 9 * 2):
            token_id_sequence = text_token_ids[idx:(idx + 9 + 1)]
            output = token_id_sequence.pop(4)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)

        return batch_input, batch_output

def collate_skipgram(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_token_ids = text_pipeline(text)

        if len(text_token_ids) < 9:
            continue

        text_token_ids = text_token_ids[:256]

        for idx in range(len(text_token_ids) - 9 * 2):
            token_id_sequence = text_token_ids[idx, (idx + 9 + 1)]
            input_ = token_id_sequence.pop(4)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)

        return batch_input, batch_output
```

如果面对vocab size很大的情况的时候，训练过程就会很慢，为此就有了negative sample这种方法，将多分类问题转换为几个二分类问题。

```python
# model init
def __init__(self, emb_size, emb_dimension):
    """Initialize model parameters. Args: emb_size: Embedding size. emb_dimention: Embedding dimention, typically from 50 to 500. """
    super(SkipGramModel, self).__init__()
    self.emb_size = emb_size
    self.emb_dimension = emb_dimension
    self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
    self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
    self.init_emb()
def init_emb(self):
    initrange = 0.5 / self.emb_dimension
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.v_embeddings.weight.data.uniform_(-0, 0)

# negative sample implementation
def forward(self, pos_u, pos_v, neg_v):
    """Forward process. As pytorch designed, all variables must be batch format, so all input of this method is a list of word id. Args: pos_u: list of center word ids for positive word pairs. pos_v: list of neibor word ids for positive word pairs. neg_v: list of neibor word ids for negative word pairs. """
    losses = []
    emb_u = self.u_embeddings(Variable(torch.LongTensor(pos_u)))
    emb_v = self.v_embeddings(Variable(torch.LongTensor(pos_v)))
    score = torch.mul(emb_u, emb_v).squeeze()
    score = torch.sum(score, dim=1)
    score = F.logsigmoid(score)
    losses.append(sum(score))
    neg_emb_v = self.v_embeddings(Variable(torch.LongTensor(neg_v)))
    neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    neg_score = F.logsigmoid(-1 * neg_score)
    losses.append(sum(neg_score))
    return -1 * sum(losses)
    
```
reference: word2vec Parameter Learning Explained
