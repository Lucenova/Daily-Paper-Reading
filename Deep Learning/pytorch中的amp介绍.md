#### pytorch的amp介绍

##### AMP介绍

神经网络框架的计算核心是tensor，也就是从scaler -> array -> matrix -> tensor 维度一路增加过来的tensor，在pytorch中创建tensor的方法为：

```python
import torch


gemfield = torch.zeros(70, 30)
# torch.FloatTensor
gemfield.type()
```

可以看到默认创建的tensor的类型为FloatTensor类型，在pytorch中，一共有10种类型的tensor：

- torch.FloatTensor (32-bit floating point)
- torch.DoubleTensor (64-bit floating point)
- torch.HalfTensor (16-bit floating point 1)
- torch.BFloat16Tensor (16-bit floating point 2)
- torch.ByteTensor (8-bit integer (unsigned))
- torch.CharTensor (8-bit integer (signed))
- torch.ShortTensor (16-bit integer (signed))
- torch.IntTensor (32-bit integer (signed))
- torch.LongTensor (64-bit integer (signed))
- torch.BoolTensor (Boolean)

amp（自动混合精度）关键词有两个**自动**、**混合**，通过torch.cuda.amp这个模块来实现。

混合精度使用了torch.FloatTensor和torch.HalfTensor这两种精度。

自动表明Tensor的类型dtype会自动变化，也就是框架按需自动调整tensor的dtype。（但实际上也不是完全自动的，有些地方也需要手动干预）

**torch.cuda**.amp 的名字意味着这个功能只能在cuda上使用，只有支持Tensor core的CUDA硬件才能享受到AMP的好处，Tensor Core是一种矩阵乘累加的计算单元，每个Tensor Core每个时钟执行64个浮点混合精度操作（FP16矩阵相乘和FP32累加），英伟达宣称使用Tensor Core进行矩阵运算可以轻易的提速，同时降低一半的显存访问和存储。

PyTorch中，当我们提到自动混合精度训练，我们说的就是在NVIDIA的支持Tensor core的CUDA设备上使用torch.cuda.amp.autocast （以及torch.cuda.amp.GradScaler）来进行训练。

#### 为什么需要自动混合精度？

torch.HalfTensor的优势就是存储小、计算快、更好的利用CUDA设备的Tensor Core。因此训练的时候可以减少显存的占用（可以增加batchsize了），同时训练速度更快；

torch.HalfTensor的劣势就是：数值范围小（更容易Overflow / Underflow）、舍入误差（Rounding Error，导致一些微小的梯度信息达不到16bit精度的最低分辨率，从而丢失）。

可见，当有优势的时候就用torch.HalfTensor，而为了消除torch.HalfTensor的劣势，我们带来了两种解决方案：

1，梯度scale，这正是上一小节中提到的torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度的underflow（这只是BP的时候传递梯度信息使用，真正更新权重的时候还是要把放大的梯度再unscale回去）；

2，回落到torch.FloatTensor，这就是混合一词的由来。那怎么知道什么时候用torch.FloatTensor，什么时候用半精度浮点型呢？这是PyTorch框架决定的，在PyTorch 1.6的AMP上下文中，如下操作中tensor会被自动转化为半精度浮点型的torch.HalfTensor：

1. __matmul__
2. addbmm
3. addmm
4. addmv
5. addr
6. baddbmm
7. bmm
8. chain_matmul
9. conv1d
10. conv2d
11. conv3d
12. conv_transpose1d
13. conv_transpose2d
14. conv_transpose3d
15. linear
16. matmul
17. mm
18. mv
19. prelu

#### 如何在PyTorch中使用自动混合精度？

##### autocast

```python
from torch.cuda.amp import autocast

model = Net().cuda()
optimizer = optim.SGD(model.params(), ...)


for input, target in data:
    optimizer.zero_grad()

    # 前向过程(model + loss)开启 autocast
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    # 反向传播在autocast上下文之外
    loss.backward()
    optimizer.step()
```

当进入autocast的上下文后，上面列出来的那些CUDA ops 会把tensor的dtype转换为半精度浮点型，从而在不损失训练精度的情况下加快运算。刚进入autocast的上下文时，tensor可以是任何类型，你不要在model或者input上手工调用`.half()` ，框架会自动做，这也是自动混合精度中“自动”一词的由来。

另外一点就是，autocast上下文应该只包含网络的前向过程（包括loss的计算），而不要包含反向传播，因为BP的op会使用和前向op相同的类型。

##### GradScaler

```python
from torch.cuda.amp import autocast as autocast

# 创建model，默认是torch.FloatTensor
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# 在训练最开始之前实例化一个GradScaler对象
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # 前向过程(model + loss)开启 autocast
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss. 为了梯度放大.
        scaler.scale(loss).backward()

        # scaler.step() 首先把梯度的值unscale回来.
        # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
        # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        scaler.step(optimizer)

        # 准备着，看是否要增大scaler
        scaler.update()
```

scaler的大小在每次迭代中动态的估计，为了尽可能的减少梯度underflow，scaler应该更大；但是如果太大的话，半精度浮点型的tensor又容易overflow（变成inf或者NaN）。所以动态估计的原理就是在不出现inf或者NaN梯度值的情况下尽可能的增大scaler的值——在每次scaler.step(optimizer)中，都会检查是否又inf或NaN的梯度出现：

1，如果出现了inf或者NaN，scaler.step(optimizer)会忽略此次的权重更新（optimizer.step() )，并且将scaler的大小缩小（乘上backoff_factor）；

2，如果没有出现inf或者NaN，那么权重正常更新，并且当**连续多次**（growth_interval指定）没有出现inf或者NaN，则scaler.update()会将scaler的大小增加（乘上growth_factor）。
