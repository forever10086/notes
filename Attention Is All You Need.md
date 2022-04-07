		注意力机制”实际上就是想将人的感知方式、注意力的行为应用在机器上，**让机器学会去感知数据中的重要和不重要的部分**。

​		所谓的"注意力机制"也就是当机器在做一些任务，比如要识别下面这张图片是一个什么动物时，我们让机器也存在这样的一个注意力侧重，最重要该关注的地方就是图片中动物的面部特征，包括耳朵，眼睛，鼻子，嘴巴，而不用太关注背景的一些信息，核心的目的就在于希望机器能在很多的信息中注意到对当前任务更关键的信息，而对于其他的非关键信息就不需要太多的注意力侧重。同样的如果我们在机器翻译中，我们要让机器注意到每个词向量之间的相关性，有侧重地进行翻译，模拟人类理解的过程。

<img src="F:\论文\image-20220223145214682.png" alt="image-20220223145214682" style="zoom: 33%;" />

### encoder与decoder		

​		Transformer的单次编码过程：不改变输入矩阵的形状，而只是根据句子中不同token的交互更新输入数据，这样的编码过程会串行重复N次（encoder模块重复N次）。

​		下面介绍Transformer的解码过程：利用输出句子的每个单词的编码对输入句子数据查询和利用。Transformer的解码过程同样串行重复N次，计算方式与编码过程类似，公式与编码时相同。

<img src="F:\论文\image-20220212200753450.png" alt="image-20220212200753450" style="zoom:50%;" />

### attention机制

​		Value承载着被分为多个部分的目标信息，Key则是这些目标信息的索引，Query代表着注意力的顺序，比如说我们在看一幅画时视线扫过画中各个区域的顺序。注意力的计算过程就是通过Query序列去检索Key，以取得合适的Value信息。

​		dot-product attention 一般用矩阵运算，Q K V 分别是三个矩阵，均表示一组向量，dot-product attention想做的是如何用V中的向量表示Q，Q一般指的是要表示的目标，K要和Q建立联系，计算相关性，以计算出的相关性为权重，加权叠加矩阵V中的向量。下图是Transformer中用的dot-product attention，$\sqrt{d_k}$作用是缩放，一般的dot-product attention可以不用缩放。


$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V,\\
$$
​		对于较小的$\sqrt{d_k}$值，加法注意与点积注意类似，对于较大的$\sqrt{d_k}$值，加法注意优于点积注意,所以要对点积注意进行放缩操作

​		其中$QK^T$正是注意力中最为重要的交互步骤：Q代表着需要编码的词的信息，K代表着句子中其它词的信息，相乘后得到句子中其它词的权重值attention score； V代表着每个位置单词蕴含的语义信息，在被加权求和后作为待编码词的编码。我们会发现，此时的注意力机制将输入矩阵编码成大小为$S\times d$的矩阵。所谓多头注意力机制，就是将n个这样得到的编码进行拼接，得到大小为$S\times[d,\dots,d]$(其中 d的个数为 n)的矩阵。在原文中，$s\times d$的结果刚好等于D，两者不相等也没有关系，因为其后还有一个$W_O$矩阵，用于将编码从$n\times d$维转化为D维。

​		此处对于计算得到的Q、K、V，可以理解为这是对于同一个输入进行3次不同的线性变换来表示其不同的3种状态。在计算得到Q、K、V之后，就可以进一步计算得到权重向量。

### masked self-attention

​		让decoder不看到未来的结果，举个例子：将“machine learning”翻译成“机器学习”时，将machine learning的encoder结果与“machine”放入decoder中，我们应该看到的是“machine”的结果“机器”，而不应该有“学习”，所以mask的操作就是先把“学习”给掩藏起来，而如何做到mask操作呢？把需要掩藏的attention值设为一个比较大的负数值，那么经过softmax后，值就无限接近0，那么就相当于掩藏起来了。

### multi-head self-attention

<img src="F:\论文\image-20220223150133320.png" alt="image-20220223150133320" style="zoom:50%;" />

1. 因为scaled dot-product attention可学习的参数少，所以有了多头注意力机制，在这里面，我们要学习的有W~i~^Q^,W~i~^K^,W~i~^V^。
2. 使用多头注意力机制是为了让模型从多个维度学习，给予注意力层的输出包含有不同子空间中的编码表示信息，从而增强模型的表达能力。
3. 自注意力机制的缺陷就是：模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置， 因此作者提出了通过多头注意力机制来解决这一问题。
4. 最终的多头注意力值需要将每个头的注意力值concatenation之后，做一个线性操作。$MultiHead(Q,K,V)=Concatenation(head_1,\dots,head_h)W^O$

### position encoding

​		attention机制没有使用到时序信息，query与key之间的相似度决定了权值，以此来加权value，但是key与value对在序列里面的顺序并没有纳入学习范围。就好比如在机器翻译里面，英文单词顺序调换对中文的翻译结果没有影响。==将输入的一段时序数据例如句子中的单词embedding成向量之后，再拼接上单词的位置信息。==

```
在注意力中，有两个token（x 和 y），第一个通过Q，第二个通过K，并通过它们比较结果查询和键向量的相似程度点积。所以，基本上，我们想要 Qx 和 Ky 之间的点积，我们写成：(Qx)'(Ky) = x' (Q'Ky)。所以等效地，我们只需要学习一个联合 Query-Key 转换 (Q'K)。

分别将position encoding e 和 f 添加到 x 和 y，我们基本上将点积更改为(Q(x+e))' (K(y+f)) = (Qx+Qe)' (Ky+Kf) = (Qx)' Ky + (Qx)' Kf + (Qe)' Ky + (Qe )' Kf = x' (Q'Ky) + x' (Q'Kf) + e' (Q'Ky) + e' (Q'K f)，其中除了原来的x' (Q'Ky)术语，它询问“给定单词 y，我们应该对单词 x 给予多少关注”，我们还有 x' (Q'Kf) + e' (Q'Ky) + e' (Q'K f)，其中问额外的问题，“给定单词 y 的位置 f，我们应该对单词 x 给予多少关注”，“给定单词 x 的位置 e，我们应该对 y 给予多少关注”，以及“我们应该给予多少关注给定单词 y 的位置 f 到单词 x 的位置 e"。

本质上，具有位置编码的学习变换矩阵 Q'K 必须同时完成所有这四个任务。这是可能看起来效率低下的部分，因为直观地说，应该在 Q'K 同时很好地完成四个任务的能力上进行权衡。

然而当我们强制 Q'K 完成所有这四个任务时，可能实际上并没有权衡取舍，因为在高维空间中满足了一些近似正交条件，因为随机选择的高维向量几乎总是近似正交的，没有理由认为词向量和位置编码向量以任何方式相关。如果词嵌入形成了一个较小维度的子空间，而位置编码形成了另一个较小维度的子空间，那么这两个子空间本身可能是近似正交的，因此推测这些子空间可以转换，通过相同的学习 Q'K 变换独立地进行（因为它们基本上存在于高维空间的不同轴上）。

如果为真，这将解释为什么sum位置编码而不是concatenation基本上没问题：concatenation将确保位置维度与单词维度正交，但我的猜测是，因为这些嵌入空间的维度非常高，即使sum这种方式也可以免费获得近似正交性，而无需像concatenation那样需要额外的连接成本（更多参数学习）。
```

---

​		Transformer的设计最大的带来性能提升的关键是==任意两个单词的距离是1==，这对解决NLP中棘手的长期依赖问题是非常有效的。

---

---

​		Decoder的masked Self-Attention和Encoder的Self-Attention非常像，只不过当要Decoder第二个词时，用黑框蒙住了第三、四个及之后词的运算（设置值为-1e9，近似看作是负无穷大，为了在softmax过程中不起作用）。因为对于机器翻译来说，**Encoder时能看到源句子所有的词，但是翻译成目标句子的过程中，Decoder只能看到当前要翻译的词之前的所有词，看不到之后的所有词**，所以要把之后的所有词都遮住。这也说明==Transformer只是在Encoder阶段可以并行化，Decoder阶段依然要一个个词顺序翻译，依然是串行的。==

---



### Self Attention与传统的Attention机制非常的不同：

​		传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是**源端的每个词与目标端每个词之间的依赖关系**。但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，**捕捉source端或target端自身的词与词之间的依赖关系**；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，self Attention比传统的Attention mechanism效果要好，主要原因之一是，**传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系**，相对比，self Attention可以不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系。

<img src="https://upload-images.jianshu.io/upload_images/4155986-d6214fe17fa1ee46" alt="img" style="zoom:50%;" />

### 注意力机制如何实现，以及注意力机制的分类

​		简单来说就是对于模型的每一个输入项，可能是图片中的不同部分，或者是语句中的某个单词分配一个权重，这个权重的大小就代表了我们希望模型对该部分一个关注程度。这样一来，通过权重大小来模拟人在处理信息的注意力的侧重，有效的提高了模型的性能，并且一定程度上降低了计算量。

深度学习中的注意力机制通常可分为三类：软注意（全局注意）、硬注意（局部注意）和自注意（内注意）

1. Soft/Global Attention(软注意机制)：对每个输入项的分配的权重为0-1之间，也就是某些部分关注的多一点，某些部分关注的少一点，因为对大部分信息都有考虑，但考虑程度不一样，所以相对来说计算量比较大。
2. Hard/Local Attention(硬注意机制)：对每个输入项分配的权重非0即1，和软注意不同，硬注意机制只考虑那部分需要关注，哪部分不关注，也就是直接舍弃掉一些不相关项。优势在于可以减少一定的时间和计算成本，但有可能丢失掉一些本应该注意的信息。 可以用于*图像裁剪（image cropping）*，**强注意力是一个不可微的注意力，训练过程往往是通过增强学习(reinforcement learning)来完成的，想要学习模型参数的话，就必须使用分数评估器（score-function estimator）。**
3. Self/Intra Attention（自注意力机制）：==对每个输入项分配的权重取决于输入项之间的相互作用==，即通过输入项内部的"表决"来决定应该关注哪些输入项。和前两种相比，在处理很长的输入时，具有**并行计算**的优势。



### reference

- https://jalammar.github.io/illustrated-transformer/	对transformer讲解很详细
- [深度学习attention机制中的Q、K、V分别是从哪来的？ - lllltdaf的回答 - 知乎](https://www.zhihu.com/question/325839123/answer/1903376265)   讲解Q、K、V的存在意义

```
假如一个男生B，面对许多个潜在交往对象B1，B2，B3...，他想知道自己谁跟自己最匹配，应该把最多的注意力放在哪一个上。那么他需要这么做：
1、他要把自己的实际条件用某种方法表示出来，这就是Value；
2、他要定一个自己期望对象的标准，就是Query；
3、别人也有期望对象标准的，他要给出一个供别人参考的数据，当然不能直接用自己真实的条件，总要包装一下，这就是Key；
4、他用自己的标准去跟每一个人的Key比对一下（Q*K），当然也可以跟自己比对，然后用softmax求出权重，就知道自己的注意力应该放在谁身上了，也有可能是自己。
```

- https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/?utm_source=share&utm_medium=web2x&context=3  position encoding为什么用sum而不用concatenate？
- https://blog.csdn.net/yujianmin1990/article/details/85221271
- [Transformer Architecture: The Positional Encoding - Amirhossein Kazemnejad's Blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [CS224N（2.21）Transformers and Self-Attention For Generative Models | bitJoy](https://bitjoy.net/2020/03/04/cs224n（2-21）transformers-and-self-attention-for-generative-models/)
- [Self-Attention和Transformer - machine-learning-notes (gitbook.io)](https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer#mo-xing-de-si-xiang)
- [NLP中 batch normalization与 layer normalization - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/74516930)这解释了在transformer中为什么采用layer normalization

```latex
	如果我们将一批文本组成一个batch，那么BN的操作方向是，对每句话的第一个词进行操作。但语言文本的复杂性是很高的，任何一个词都有可能放在初始位置，且词序可能并不影响我们对句子的理解。而BN是针对每个位置进行缩放，这不符合NLP的规律。
	而LN则是针对一句话进行缩放的，且LN一般用在第三维度，词向量的维度，或者是RNN的输出维度等等，这一维度各个特征的量纲应该相同。因此也不会因为特征的量纲不同而导致的缩放问题。
```

- [transformer的细节到底是怎么样的？ - 月来客栈的回答 - 知乎](https://www.zhihu.com/question/362131975/answer/2182682685)

- [NLP中的Transformer架构在训练和测试时是如何做到decoder的并行化的？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/307197229)这解释了encoder无论是训练还是测试都是可并行的，而**Decoder的并行化仅在训练阶段，在测试阶段，因为我们没有正确的目标语句，t时刻的输入必然依赖t-1时刻的输出，这跟之前的seq2seq就没什么区别了。**
