#### Pytorch笔记

##### tensor创建

```python
torch.tensor(data, dtype=torch.float32) data可以是list, tuple, numpy array, scalar
特定从numpy创建 torch.from_numpy(ndarry)
```

##### tensor转换

- Int与tensor转换

  ```python
  A = torch.tensor(1)
  b = A.item()
  ```

- list与tensor转换

  ```
  a = [1, 2, 3]
  b = torch.tensor(a)
  c = b.numpy().tolist()
  ```

##### tensor常用操作

```python
# 获取形状
a = torch.tensor(2, 3, 5)
a.size()
a.shape

# 维度转换
# 维度变换之后，tensor在内存中存储的位置就会变得不连续
x = torch.randn(2, 3, 5)
x.shape # torch.Size([2, 3, 5])
x = x.permute(2, 1, 0) #传入的是维度的序号
x.shape # torch.Size([5, 3, 2])

x = torch.randn(2, 3, 4)
x.shape # torch.Size([2, 3, 4])
x = x.transpose(1, 2)
x.shape # torch.Size([2, 4, 3])

# 形状变换
x = torch.randn(4, 4)
x.shape # torch.Size([4, 4])
x = x.view(2, 8)
x.shape # torch.Size([2, 8])
x = x.permute(1, 0)
x.shape # torch.Size([8, 2])
x.view(4, 4) # error view不能处理不连续结构的Tensor的结构
x = x.reshape(4, 4) # reshape相当于进行了两步操作，让tensor在内存中储存变得连续，在进行view操作
x.shape # torch.Size([4, 4])

# 增减维度
x = torch.rand(2, 1, 3)
x.shape # torch.Size([2, 1, 3])
y = x.squeeze(1)
y.shape # torch.Size([2, 3])
z = y.squeeze(1)
z.shape # torch.Size([2, 3]) # squeeze所指定的维度如果是1，就相当于把那个维度给去掉，如果不为1，啥也不做

x = torch.randn(2, 1, 3)
y = x.unsqueeze(2)
y.shape #torch.Size([2, 1, 1, 3]) #unsqueeze相当于给指定维度index，增加一个维度

# 连接操作
# cat只是维度上的拼接，拼接出来的结果的维度不会变化
# tensors若干个准备拼接的Tensor
# dim拼接的维度
torch.cat(tensors, dim=0, out=None)
torch.cat((A, B), dim=1)
# stack是把多个Tensor堆叠，那么肯定就会多出一个维度，且维度等于Tensor的个数
# inputs表示要堆叠Tensor，dim表示新建立维度的方向
torch.stack(inputs, dim=0)

#切分操作
# chunk是先求出每份的个数（如果不是整除就向上取整），然后均分, 若chunks大于Tensor维度上的元素，则相当于将每个元素都作为1个chunks
# input准备切分的Tensor
# chunks切分的份数
# 返回包含每份的元组
torch.chunk(input, chunks, dim=0)
A = torch.ones(6, 4, 2)
A.shape # torch.Size([6, 4, 2])
B = torch.chunk(A, 2, dim=1) # B为长度为chunk=2的元组
B[0].shape # torch.Size([6, 2, 2])
# split按照每份按照确定的大小，split如果发现不能完全按照split_size_or_sections分割tensor，则先按照size大小分割，剩下的多少就是多少
# tensor待切分的Tensor
# split_size_or_sections为整数时，表示将tensor按照每块大小为这个整数的数值来切割，当这个参数为列表的时候，则表示将此tensor切成和列表中元素一样大小的块
torch.split(tensor, split_size_or_sections, dim=0)
# input表示待处理的tensor, dim表示维度的方向
torch.unbind(input, dim=0)
A = torch.arange(0. 16).view(4, 4)
# tensor([[ 0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9, 10, 11], [12, 13, 14, 15]])
b=torch.unbind(A, 0) # (tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7]), tensor([ 8, 9, 10, 11]), tensor([12, 13, 14, 15]))
b=torch.unbind(A, 1) #(tensor([ 0, 4, 8, 12]), tensor([ 1, 5, 9, 13]), tensor([ 2, 6, 10, 14]), tensor([ 3, 7, 11, 15]))

#索引操作
# index表示从dim维度中哪些位置选择数据，其实就是选定某个特定维度之后，在选择的index
torch.index_select(tensor, dim, index)
A=torch.arange(0,16).view(4,4) #tensor([[ 0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9, 10, 11], [12, 13, 14, 15]])
B=torch.index_select(A,0,torch.tensor([1,3])) # tensor([[ 4, 5, 6, 7], [12, 13, 14, 15]])
C=torch.index_select(A,1,torch.tensor([0,3])) # tensor([[ 0, 3], [ 4, 7], [ 8, 11], [12, 15]])
# mask_select通过一些判断条件进行选择，比如提取深度学习网络中某一层大于0的参数
# mask和input维度必须完全相同，一一对应
torch.mask_select(input, mask, out=None)
A = torch.rand(5) # tensor([0.3731, 0.4826, 0.3579, 0.4215, 0.2285])
B = A > 0.3 # tensor([ True, True, True, True, False])
C=torch.masked_select(A, B) # tensor([0.3731, 0.4826, 0.3579, 0.4215])
```

##### 数据读取

```Python
# Dataset类
# 无论是使用自定义的数据集，还是官方为我们封装好的数据集，本质都是继承了Dataset类，在继承Dataset类必须重写以下几个方法
# __init__():构造函数，可以自定义数据读取方法以及进行数据预处理
# __len__(): 返回数据集大小
# __getitem__(): 索引数据集中的某一个数据
```

##### 模型保存与加载

```python
# 方式一：只保存训练好的参数
torch.save(model.state_dict(), './linear_model.pth')

# 加载模型方式
linear_model = LinearModel()
linear_model.load_state_dict(torch.load('./linear_model.pth'))

# 方式二：保存网络结构和参数（相比第一种方式，这种方式在加载模型的时候，不需要加载网络结构了）
torch.save(model, './linear_model_with_arc.pth')
# 加载模型，不需要创建模型了
linear_model_2 = torch.load('./linear_model_with_arc.pth')
```

##### 分布式训练

两个问题：

- 谁分布了： 数据和模型
- 怎么分布：单机多卡与多级多卡

```Python
# 单机多卡
# model就是定义的模型
# device_ids为训练用到的GPU设备号
# output_device表示输出结果的device，
torch.nn.DataParallel(model, device_ids=None, output_device=None, dim=0)
# 可以使用nvidia-smi命令查看GPU使用情况
```

