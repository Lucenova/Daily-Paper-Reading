# 论文深度解读：How do language models learn facts? Dynamics, curricula and hallucinations

## 论文信息
- **标题**：How do language models learn facts? Dynamics, curricula and hallucinations
- **作者**：Nicolas Zucchet, Jörg Bornschein, Stephanie Chan, Andrew Lampinen, Razvan Pascanu, Soham De
- **机构**：ETH Zürich, Google DeepMind
- **版本**：arXiv:2503.21676v2, 2025-07-24

---

## 一、Problem：定义问题与范围

### 1.1 这篇论文到底在问什么？
这篇论文的核心问题不是“语言模型能不能记住事实”，而是一个更基础、更机制化的问题：

> **语言模型在预训练过程中，到底是如何把外部训练数据中的事实压缩进参数里的？这种知识获得的动力学是什么？为什么会出现平台期？为什么后训练注入新知识时会遗忘并伴随幻觉？**

作者把这个问题拆成三个子问题：
1. **知识是如何在训练中出现的？**
2. **数据分布如何影响知识学习速度？**
3. **为什么 fine-tuning 注入新知识很难，并且会导致旧知识被破坏？**

### 1.2 他们如何界定“知识”？
论文区分了两个概念：
- **knowledge（知识）**：模型能够脱离原始句式，以更抽象、更灵活的方式调用训练数据中的事实。
- **memorization（记忆）**：模型只是在复现见过的具体训练句子或表面模式。

这是本文非常关键的理论起点。作者不是把“训练集上答对了”直接视为知识，而是要求模型在**未见模板**上也能正确回忆同一事实，借此排除纯表面记忆。

### 1.3 研究范围的边界
这篇论文刻意缩小研究范围，不研究开放域复杂知识，而聚焦于一种**synthetic factual recall**：
- 每个“人”有若干原子事实：出生地、出生日期、大学、专业、公司、居住地；
- 输入是这类合成 biography；
- 模型要预测属性值 token；
- 事实之间彼此独立，不能通过推理互相推出。

因此，这篇论文研究的不是“复杂推理中的知识使用”，而是更纯粹的：

> **参数化事实记忆是如何形成、调度和被破坏的。**

### 1.4 问题设定的优点与代价
**优点**：
- 控制变量极强；
- 可以精确定义“无知识基线”；
- 可以跟踪整个训练轨迹，而不是只看最终模型；
- 容易做机制干预（attention patching）。

**代价**：
- 任务过于干净，真实 web-scale 预训练里知识、推理、语言统计和多任务学习是纠缠的；
- “名字→属性”的关联是理想化关联记忆，不等于真实大模型中的全部知识形态；
- 外推到多模态和复杂开放世界任务时，需要谨慎。

---

## 二、Methodology：思路与步骤

这篇论文的方法链条很清晰，可以概括为：

> **构造一个能精确测知识的可控环境 → 在线跟踪学习动力学 → 做机制干预 → 改变数据分布 → 研究后训练注入新知识。**

### 2.1 合成 biography 数据集：把“知识学习”做成可测问题
作者基于 synthetic biographies 构造训练数据。每个个体都有唯一姓名和 6 个属性，生成 biography 时：
1. 先采样一个个体；
2. 对每个属性采样一个模板；
3. 用该个体的属性值填充模板；
4. 将 6 个属性句子随机打乱并拼接成 biography。

这里有三个关键设计：

#### 设计1：事实是原子的
每个事实彼此独立，避免模型靠逻辑推理“猜”出来。

#### 设计2：模板多样、顺序随机
每种属性有 25 个模板，并且随机排列，避免模型只记住固定顺序或固定句式。

#### 设计3：评测用未见模板
每个个体训练时只见 20 个模板，评测用剩余 5 个模板，因此评测更接近“知识迁移”，而非“句式复现”。

### 2.2 度量设计：attribute loss 与 no-knowledge baseline
作者不使用传统问答形式，而是把 attribute value token 的预测视为 factual recall。

核心指标有两个：
- **attribute loss**：属性值 token 上的平均交叉熵损失；
- **attribute accuracy**：一个属性值的所有 token 都预测正确才算对。

更重要的是，他们提出了一个理论参照：
- **no-knowledge baseline**：如果模型完全不知道个体级信息，只知道属性值总体分布，那么它能达到的最优损失就是属性值分布的熵。

这使得“模型是否真的学到个体知识”变成了可以定量判断的问题：
- 如果性能只是达到熵基线，说明模型只学会了总体统计；
- 如果明显优于熵基线，才说明模型掌握了个体事实。

### 2.3 学习动力学：三阶段结构
这是全文最核心的发现之一。作者发现知识学习不是平滑的，而是三阶段：

#### 阶段1：generic statistics learning
模型快速学会属性总体分布，例如“出生地像城市名，大学像学校名”。此时达到 no-knowledge baseline。

#### 阶段2：plateau
模型长时间停在熵基线附近，几乎没有个体知识准确率提升。这个平台期长度几乎与个体数成线性关系。

#### 阶段3：knowledge emergence
模型开始真正学会“某个名字对应某个属性值”的关联，知识准确率显著上升。

这说明知识学习并不是“遇到一条知识就连续积累一点”，而是在很多设置下带有明显相变结构。

### 2.4 机制解释：attention-based recall circuits 在平台期形成
作者提出一个很强的机制解释：

> **平台期本质上是 recall circuit 尚未形成的时期。**

具体图景如下：
1. 前几层 attention 先把名字 token 聚合成“这个人是谁”的表示；
2. MLP / feed-forward 层像 key-value memory 一样存储个体相关信息；
3. 最后层 attention 根据当前属性类型，把对应信息抽取出来供输出 token 预测。

在这个电路没有建好之前，属性值预测的梯度无法精准回传到名字 token，而会扩散到不相关 token 上，导致 credit assignment 很差，所以 loss 卡在平台期。

### 2.5 关键机制实验：attention patching
这篇论文最有说服力的机制实验，是 **attention patching**。

做法是：
- 先训练一个 reference model；
- 保存它在不同训练步的 attention patterns；
- 再从头训练一个 modified model，但把 attention pattern 替换成 reference model 某一步的 attention；
- modified model 只学习 token-wise feed-forward 变换。

结果非常关键：
- 如果提供的是**平台后**的 attention patterns，平台期几乎消失，学习大幅加快；
- 如果提供的是早期尚未形成的 attention patterns，效果反而差。

这相当于提供了因果证据：
- attention pattern 的成熟程度，直接决定知识学习速度；
- 平台期不是“什么都没发生”，而是在形成 recall circuit。

### 2.6 数据分布分析：不均衡分布能缩短平台期
作者进一步研究：如果不是均匀采样个体，而是使用 inverse power law / Zipf 分布，会怎样？

发现非常有意思：
- **适度不均衡**：平台期更短；
- **过度不均衡**：会过拟合，长尾知识学得更差；
- 因此存在 trade-off：
  - 不均衡分布更有利于 recall circuit 形成；
  - 均匀分布更有利于整体覆盖与最终知识获取。

这说明重复并非纯粹浪费。恰当的高频样本可以更快让模型识别“哪些相关性值得建电路”。

### 2.7 Warm-up scheduler：先少样本、后全样本
基于上述 trade-off，作者提出 warm-up：
1. 前期只在一部分个体上训练；
2. 后期再切回全部个体、均匀分布。

这样做的直觉很简单：
- 前期用低多样性、强化重复，先把 recall circuit 建出来；
- 后期用高多样性，提升整体知识覆盖。

实验表明 warm-up 比始终 uniform 更好，也常优于静态 Zipf 分布。

### 2.8 后训练：幻觉与知识注入问题
作者最后研究 fine-tuning 注入新知识时发生什么。

#### 发现1：幻觉与知识同时出现
模型一旦开始掌握个体知识，就会对未见个体做出高置信错误预测。也就是说：

> **hallucination 不是“模型不会时乱答”，而是“模型开始会时，也开始对不会的东西过度自信”。**

#### 发现2：fine-tuning 新个体会快速破坏旧知识
在新个体上微调时：
- 旧知识在最初几百步内迅速塌陷；
- 新知识的获得却更慢；
- replay 可以缓解最终下降，但很难阻止最初坍塌。

#### 发现3：attention 不是主要罪魁
作者跟踪 fine-tuning 期间的 attention patterns，发现它们相当稳定，因此旧知识崩塌更可能来自：
- **feed-forward associative memories 被新 key-value 干扰。**

#### 发现4：MLP toy model 复现了同样现象
他们训练了一个没有 attention 的单隐层 MLP，在 toy associative recall 上也观察到类似 fine-tuning 遗忘行为，这进一步支持：

> **新知识注入的核心冲突位于 FF associative memory，而非 attention 电路。**

---

## 三、Discussion：经济性、延展性、证明完备性

### 3.1 经济性：这篇论文值不值得投入工程资源？
从工程角度看，这篇论文最大的价值不是直接给出一个可立刻大规模部署的 SOTA 技巧，而是给出了一个关于训练浪费的结构性诊断：

> **很多训练步数可能都花在“电路尚未形成”的平台期里。**

这意味着：
- 训练效率不只是优化器和硬件问题，也可能是数据分布问题；
- 合理的数据调度可能比单纯增大模型更高性价比；
- 早期训练数据如果无法被有效吸收，等于算力浪费。

但这篇论文离真实大模型训练仍有距离：
- 实验模型是 44M 左右，默认 8 层 Transformer；
- 任务是单一合成 factual recall；
- 真实训练中不容易像本文这样精确观测某个子任务是否处于平台期。

所以其**工程启发很强，直接工程可用性中等**。

### 3.2 延展性：能推广到真实 LLM / MLLM 吗？
我认为这篇论文最可迁移的不是具体设置，而是三个“结构性命题”：

#### 命题1：知识学习可能具有相变结构
在真实大模型中，许多能力提升可能不是平滑出现，而是某些 recall / routing / extraction circuit 成熟后才突然显现。

#### 命题2：数据分布决定电路形成速度
并不是所有样本都同等有助于建立关键电路。重复、低多样性、高相关局部统计，可能在早期更重要。

#### 命题3：后训练注入新知识的瓶颈可能在 FF memory 干扰
这对 continual pretraining、知识编辑、LoRA 注入、RAG 替代 FT 等路线都很有启发。

对多模态来说，这可以自然映射到：
- 视觉 token → 实体 token → 答案 token 的 recall / extraction circuit；
- 视觉事实学习是否也有平台期；
- 视频或多帧任务里是否也存在“先建抽取电路，再写入 FF memory”的两阶段。

### 3.3 证明完备性：证据链强在哪里？弱在哪里？

#### 强项
1. **问题定义干净**：知识与记忆区分得很明确；
2. **无知识基线非常漂亮**：使平台期可解释；
3. **attention patching 提供近因果证据**；
4. **fine-tuning 行为用 toy MLP 复现**，说明机制不是纯 Transformer 特有巧合；
5. **多种消融显示三阶段现象较稳健**。

#### 弱项
1. **attention pattern 分析仍然不完整**：作者自己也承认，这种分析忽略 values，本质上仍有相关性而非完全机制证明；
2. **真实语料中的知识并不都是原子独立的**，而本文假设了非常干净的原子事实；
3. **平台期的“统计解释”与“优化解释”并未被完全分离**；
4. **warm-up 只做了有限网格搜索**，离最优调度策略还有距离；
5. **scale gap**：44M 到真实大模型之间仍有巨大鸿沟。

### 3.4 我对论文最核心的批判
如果我要用一句话概括这篇论文最值得肯定、但也最该谨慎的地方，那就是：

> 它抓住了“知识学习的机制瓶颈”这一真正重要的问题，但它是在一个极度理想化的环境中抓住的。

因此，我们应该把它看作：
- 不是“真实 LLM 已被完全解释”；
- 而是“提出了一个非常强、非常可检验的机制假说”。

这是好论文的典型特征：
- 不一定完全真实；
- 但它给出了足够清晰的假说，让后来者可以在更真实的场景里验证、反驳、扩展。

---

## 四、启发：必备概念、反直觉洞见、行动指南

## 4.1 必备概念

### 概念1：No-knowledge baseline
它把“模型只学到总体统计而未学个体知识”转成一个可计算熵阈值。今后你分析知识学习时，最好都找一个类似的理论下界或无知识参照。

### 概念2：Plateau ≠ nothing happens
性能停滞不代表网络内部没有变化。平台期可能对应关键 recall circuit 的形成阶段。

### 概念3：Recall circuit
知识回忆不是一个单点操作，而是一条链：
- key 构造（名字/实体聚合）
- memory 查询（FF associative memory）
- 信息抽取（最后 attention routing）

### 概念4：Data distribution as algorithm selector
数据分布不只是影响收敛速度，还可能决定模型先形成哪类算法/电路。

### 概念5：Hallucination as byproduct of parametric recall
幻觉不一定只是对齐失败，也可能是参数化 recall 机制天然的副产物。

### 概念6：FF associative memory interference
新知识注入失败，很可能不是注意力不会看，而是写入新的 key-value 时覆盖了旧的 memory basin。

---

## 4.2 反直觉洞见

### 洞见1：越“重复”的数据，早期反而可能越有价值
常见直觉是“重复样本浪费训练预算”，但论文表明，在电路形成前，重复能增强信噪比，帮助模型更快找到关键相关性。

### 洞见2：模型一学会知识，就会同时更会胡说
知识与幻觉并不是先后关系，而可能是同一 recall 机制的两面。

### 洞见3：fine-tuning 注入新知识之所以难，未必是 attention 出问题
更深层瓶颈可能是 FF associative memory 容量/干扰，而不是 token routing。

### 洞见4：模型大小不一定是平台期的决定因素
作者讨论指出，平台期长度更多由数据分布决定，而不只是模型规模。这对“只靠 scale 解决一切”的直觉是个修正。

### 洞见5：早期训练数据可能根本没有被真正“留下”
如果平台期前样本主要用于建电路而不形成可保持知识，那么那部分训练计算的“知识保留率”可能很低。

---

## 4.3 行动指南

### 指南1：为你自己的任务定义“无知识基线”
无论是多模态目标识别、属性抽取、bbox 生成还是 structured output，都应该先定义一个“只靠总体统计能做到什么程度”的基线。

### 指南2：不要只盯 loss，要盯机制 proxy
在训练中同时跟踪：
- 对关键实体 token 的注意力比例；
- attention entropy / sharpness；
- 关键位置梯度是否能回传到 key token；
- FF 层表征相似性或 key-value separation。

### 指南3：尝试两阶段数据调度
对你关心的子任务，先低多样性、高重复、强相关；再逐步扩大覆盖面。这在视觉事实学习、长尾目标识别、视频对象持续跟踪中都值得测试。

### 指南4：不要盲目用纯 fine-tune 注入新知识
尤其是当新知识与旧知识共享同一参数子空间时。更稳妥的路线包括：
- replay 混合；
- 局部参数注入；
- 显式 memory 模块；
- 检索增强；
- 子空间隔离/正交化写入。

### 指南5：专门做“未见实体”校准集
不要只测 seen data accuracy。应额外固定一批 held-out entities，持续监控：
- max probability；
- entropy；
- 是否超过 no-knowledge baseline；
- confidence gap。

---

## 五、研究灵感：3个新问题 + 动机 + 原型 + 风险

## 灵感1：自适应平台检测与数据调度

### 新问题
能否利用机制信号自动检测“recall circuit 是否成形”，并据此自适应切换训练数据分布，而不是手工设定 warm-up 步数？

### 动机
论文证明 warm-up 有效，但它是固定时长、固定子集的粗粒度策略。真实训练中，不同任务、不同模型、不同模态的“平台结束时刻”并不一致。

### 原型
做一个两阶段或多阶段自适应 scheduler：
1. 选一个可在线评估的 factual recall 子任务；
2. 定义 circuit proxy：
   - attribute token 对 key/entity token 的 attention mass；
   - attention sharpness 拐点；
   - 梯度回流到 key token 的比率；
3. 当 proxy 长期停滞且低于阈值时，减少数据多样性；
4. 当 proxy 明显上升后，恢复或增加多样性。

### 风险
- proxy 可能和真实能力不一致；
- 多任务设置中，一个任务的调度会伤害另一个任务；
- 在线观测增加计算开销。

### 研究价值
即便失败，也能回答一个重要问题：

> **平台期到底能否被可靠地在线检测？**

这本身就很有发表价值。

---

## 灵感2：知识出现时的幻觉抑制机制

### 新问题
能否在不显著损害 seen knowledge 的前提下，抑制“知识涌现同时伴随幻觉”的现象？

### 动机
论文显示 hallucination 与 knowledge emerge 几乎同步，这说明幻觉并不是训练末期偶发噪声，而是参数化 recall 的结构性副产物。

### 原型
在训练中加入“未见实体校准项”：
1. 保留一组 held-out entities；
2. 对这些实体，不要求模型回答正确，但要求其输出接近无知识先验；
3. 可以使用：
   - entropy regularization；
   - KL 到 attribute prior；
   - uncertainty head / unknown token；
   - representation density threshold。

### 风险
- 可能伤害长尾 seen entities 的记忆；
- “高熵输出”不等于真正不幻觉；
- 在复杂真实数据上，未见实体分布难构造。

### 研究价值
这条路线可以把“知识—幻觉耦合”从描述性结论推进到干预性研究，非常值得做。

---

## 灵感3：面向持续知识注入的 FF 记忆隔离机制

### 新问题
既然 fine-tuning 注入新知识时主要问题来自 FF associative memory 干扰，那么能否设计一种“低成本的 FF 写入隔离机制”，降低旧知识破坏？

### 动机
论文的强证据是：
- fine-tuning 时 attention 很稳定；
- toy MLP 也复现遗忘；
- 因此问题更像是 key-value memory 冲突，而不是 token routing 失效。

### 原型
非常适合做成参数高效路线：
1. 只在 FFN / MLP 层添加 LoRA / adapter 写入通道；
2. 对不同知识批次、不同 topic 的写入通道施加子空间正交约束；
3. 用门控机制控制何时调用新 memory 通道；
4. 用少量 replay 样本稳定边界。

### 风险
- 会增加参数与推理路径复杂度；
- topic 边界并不总是清晰；
- residual 混合仍可能造成间接干扰。

### 研究价值
这条路线同时连接：
- continual learning
- PEFT
- factual editing
- memory modularization

并且很容易做出有说服力的 ablation。

---

## 六、我的总评

这篇论文的真正价值，不在于它回答了“LLM 如何学习所有知识”，而在于它提出了一个非常强的、可检验的中层机制框架：

> **知识学习存在三阶段动力学；平台期对应 recall circuit 的形成；数据分布决定平台长短；而后训练注入新知识的瓶颈主要在 feed-forward associative memory 的干扰。**

我认为它最值得认真对待的地方有三点：
1. 它把“知识学习动力学”从经验现象推进成了机制假说；
2. 它把“数据分布”从静态数据工程问题提升成了“电路形成调度问题”；
3. 它给“为什么 fine-tuning 注入知识经常不理想”提供了比经验主义更扎实的解释。

如果把这篇论文放在更大的研究脉络里，它最像一块“桥梁”：
- 一头连接 mechanistic interpretability；
- 一头连接 curriculum / data scheduling；
- 另一头连接 continual learning 与 hallucination。

它不是终点，但非常适合作为下一批研究的起点。

---

## 七、适合继续推进的实验方向（简表）

| 方向 | 最小实验 | 关键指标 | 最大风险 |
|---|---|---|---|
| 自适应 scheduler | 用 attention mass/entropy 触发数据切换 | plateau 长度、最终 loss、sample efficiency | proxy 不可靠 |
| 幻觉抑制 | unseen entities 上加高熵/unknown 正则 | seen acc、held-out calibration、hallucination gap | 长尾知识受损 |
| FF memory 隔离 | 仅在 MLP 层做 LoRA/adapter + 正交约束 | old knowledge retention、新知识学习率 | 参数子空间仍耦合 |

---

## 八、一句话总结

这篇论文最重要的启发是：

> **知识不是连续、均匀地被写入模型的；模型往往先花很长时间搭建“如何回忆知识”的电路，真正的事实写入与泛化则在此之后才开始，而幻觉与遗忘正是这个过程的阴影。**

