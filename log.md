# 实验记录

漏了两周的记录
现在开始写

# 2024年3月
## 21日
为了测试工作记忆长度，尝试了翻转和复制任务。<br>
用llama2-7b-chat试了，不进行微调的情况下，只进行few-shot<br>
翻转任务的情况很差，复制任务表现较好<br>
但是复制任务表现好也是有要求的<br>
~~1.不能使用<bos>和<eos>作为表示符，可能是因为他们在训练时有特殊的含义(好像也没有明显差异)~~<br>
~~2.fewshot数量要足够，大概四个，并不是越多样本效果就越好，看上去到达四个之后就变得随机起来了，或许序列本身也有影响<br>~~
3.或许可以只测后一半，前面一半相当于prompt的一部分，这样zero-shot的效果也挺好(其实前面一半也相当于很多个演示了)<br>
4.无意义的token似乎更难记一些, 可以用有意义的试试<br>

复制任务,在窗口内<br>
llama2-7b-chat后一半的正确率在70%~90%左右,无意义的token<br>

试试在PG-19上训练的模型<br>

## 22日
复制任务,在窗口内<br>
对于有意义的token序列，llama2-7b-chat后一半的正确率大部分都是100%， 这么一看，这个设置比较适合测试工作记忆长度<br>

对于在PG-19上从头训练一个epoch的小模型（80M）来说<br>
记忆随机序列的正确率为接近0%, 记忆有意义的序列的在一段区间内为100%<br>
模型在2048的segment上训练，这个区间为4~32，之后就慢慢下降，远远少于训练时的区间<br>
llama2-7b-chat的较高正确率有可能源于语言建模的任务，而这样随机的token序列设置更接近“死记硬背”的工作记忆长度<br>

补充一下，我为什么要执着于不用copy task训练，因为即使在copy task能测出模型工作记忆长度的潜力，但不代表它在正常任务下仍能够有这么长的工作记忆<br>

可以计算语言建模的准确率对比一下，看看记忆（history）的增益<br>
如果前面有没有这个信息准确率都不变，说明根本没在记忆<br>

ok,语言建模的准确率和复制的准确率在训练窗口内有明显变化, 在训练窗口外差距很小。<br>
唯一可惜的是由于参数量较少，并没有达到100%的准确率。<br>

## 25日
配置：Adam，lr=0.001，betas=(0.9, 0.999), eps=1e-08<br>
固定只训练一个epoch（预训练场景），测试了一些超参，batch size影响比较大，或者说更新步数影响比较大。<br>
| data_type | seg_len | seg_num | batch_tokens | params(emb) | valid_ppl | test_ppl | 4x3090_hours |
|-----------|---------|---------|--------------|--------------|------------|----------|--------------|
| shuffle   | 1024    | 32      | 0.5M         | 83M(32M)     | 23.63      | 20.85    | 9.77         |
| shuffle   | 1024    | 32      | 0.25M        | 83M(32M)     | 17.89      | 15.79    | 9.92         |
| shuffle   | 1024    | 32      | 128k         | 83M(32M)     | 17.55      | 15.49    | 10.21        |
| shuffle   | 1024    | 32      | 64k          | 83M(32M)     | 17.4       | 15.35    | 10.77        |

测试了500k,250k,128k,64k四个batch size, 越小效果越好。<br>
时间会越来越长，应该是因为同步梯度的次数变多了<br>
但是考虑到之后的Curriculum Learning场景，还是保留较大的梯度累积步数为好<br>

选择250k tokens(梯度累积步数为8)作为固定的batch size,继续探索学习率的影响<br>


## 27日
xformers 居然比sdpa快1个小时左右，10%


## 28日

WSD调度器并没有在decay阶段出现loss的剧降，看来这个东西没有那么通用。不过确实使ppl降低了很多，但这是相对于没有使用scheduler的情况。
从趋势看，越早使用decay越好。或许使用余弦调度器能得到更好的结果。

| data_type | seg_len | seg_num | batch_tokens | params(emb) | lr       |
|-----------|---------|---------|---------------|-------------|----------|
| shuffle   | 1024    | 32      | 0.25M         | 83M(32M)    | 1.00E-03 |


| type  | warmup_steps | stable_ratio | factor | patience | valid_ppl | test_ppl | 4x3090_hours | 备注          |
|-------|--------------|--------------|--------|----------|-----------|----------|--------------|--------------|
| WSD   | 500          | 0.5          |        |          | 15.6      | 13.78    | 8.46         | 使用xformers  |
| WSD   | 500          | 0.7          |        |          | 15.64     | 13.81    | 9.9          |               |
| WSD   | 500          | 0.9          |        |          | 15.93     | 14.07    | 8.39         | 使用xformers  |

这下不得不补一下余弦调度器的实验了。<br>

还想试一下fp16和bf16混合精度训练的，但是快中期答辩了<br>

最后试一下ReduceLROnPlateau，和CosineAnnealingLR的<br>
吐槽一下,ReduceLROnPlateau几乎步步判定没有improved，几乎退化成MultiStepLR了<br>
而CosineAnnealingLR或许因为T_max太大了(12,000)，看上去挺直的，感觉和LinearLR的差别不是很大。


## 29日

加入KeyNorm, scale改为1.0<br>
开始的时候$||k||\approx3.5$左右，到后面才变成9左右<br>
~~啧，端口冲突也会影响正在跑的程序，下次还是耐心点。<br>~~
照常算的话$\sqrt{d}=8$，$\frac{1}{3.5}=\frac{2.1}{8}$<br>
应该没问题<br>
先把余弦调度器的结果跑出来,确定好超参数再搞模型结构<br>

检查代码commit<br>

## 31日

让梯度跨越多个segment好像没有这么简单, 使用backward之后计算图就没了。<br>
啊，用retain_graph=True就行<br>
https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
感觉还是把所有loss加起来方便，用retain_graph=True有点担心计算图有的地方没清完。<br>
反向传播这部分还是有点神秘。<br>
x1->h1->h2->h3->loss1, loss1.backward(retain_graph=True) 计算图保留<br>
x2,h2 -> z -> loss2, loss2.backward() 计算图清除<br>
x2,h2 -> z -> loss2 被清除，因为h2能反向传播x1->h1->h2也被清除<br>
那h3->loss1会不会被清除呢?<br>
计算图有两个根节点真是麻烦<br>

补习:https://pytorch.org/docs/stable/notes/autograd.html<br>
看来就地操作真的会影响自动微分，训练时不用index_copy_是正确的

测试了一下h3->loss1是不会被清除的。

```
import torch

# 定义第一个计算图的变量
x1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
h1 = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
y = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

# 构建第一个计算图
h2 = h1 ** x1 ** y
z = h2 ** 2
l = z ** 2
loss1 = l.mean()
print("-"*50)
print(h2.grad_fn._saved_self,z.grad_fn._saved_self,l.grad_fn._saved_self)
# 执行第一个计算图的backward并保留计算图
loss1.backward(retain_graph=True)# retain_graph=True
print("-"*50)
print(h2.grad_fn._saved_self,z.grad_fn._saved_self,loss1.grad_fn._saved_self)

# 定义第二个计算图的变量
x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 构建第二个计算图
h3 = h2 * x2  # 注意，这里h2来自第一个计算图
loss2 = h3.sum()

# 执行第二个计算图的backward，不保留计算图
loss2.backward()

print("-"*50)
# print(h2.grad_fn._saved_self)

print(z.grad_fn._saved_self,loss1.grad_fn._saved_self)

```

确定z->loss1不会被清除之后还有一个问题，就是loss1的引用计数为0后，h3->loss1是否会被清除。<br>
del loss1后 z的.grad_fn._saved_self还是能打印出来。<br>
换个角度思考：<br>
整个计算图能保留下来是因为根节点（loss）保留了中间变量的引用。<br>
如果根节点引用没了的话, 计算出根节点的节点的引用也会-1,如果该节点没有其他引用,应该也会释放资源。<br>

试了一下，显存消耗不会增加
```
from tqdm import tqdm
import torch

shape = (320,512,512)
# 定义第一个计算图的变量
x1 = torch.randn(shape, requires_grad=True,device='cuda')
# 定义第二个计算图的变量
x2 = torch.randn(shape, requires_grad=True ,device='cuda')
for i in tqdm(range(100000000000)):

    h1 = torch.randn(shape,device='cuda')
    y = torch.randn(shape,device='cuda')

    # 构建第一个计算图
    h2 = h1 ** x1 ** y
    z = h2 ** 2
    l = z ** 2
    loss1 = l.mean()
#     print("-"*50)
#     print(h2.grad_fn._saved_self,z.grad_fn._saved_self,l.grad_fn._saved_self)
    # 执行第一个计算图的backward并保留计算图
    loss1.backward(retain_graph=True)# retain_graph=True
#     print("-"*50)
#     print(h2.grad_fn._saved_self,z.grad_fn._saved_self,loss1.grad_fn._saved_self)



    # 构建第二个计算图
    h3 = h2 * x2  # 注意，这里h2来自第一个计算图
    loss2 = h3.sum()

    # 执行第二个计算图的backward，不保留计算图
    loss2.backward()
```

那就不管了。只要不爆显存，这个计算图留着也不会改变梯度的传递。<br>
话说zero_grad()也可以间接清除计算图。<br>
呃，不行，zero_grad()只影响叶子节点, 中间变量还是不会影响<br>
只能等垃圾回收了<br>

不过现在的代码还是兼容stop_gradient_step==1的(像Transformer-XL一样每个segment都禁止向前传播梯度)，之后设为其他再将代码改为retain_graph=True吧<br>

做好不能长期使用GPU的准备。<br>
写个断点重训的函数吧<br>

要不把代码搬下来，用自己的4090训得了。<br>


# 4月
## 4日
### 理论感受野不等于实际感受野
类似于Transformer-XL的架构跑了<br>
copy task 和 lm task的图很有意思<br>
在work_size(q_len)+cache_size(kv_cache_len)内,表现copy task明显高于lm task<br>
但是一到窗口之外，copy task就和lm task的效果一样了,甚至连方差都很近<br>
因为xl的理论感受野是layer_num * work_size的<br>
但这个结果和截断推理没什么区别<br>
因为Mistral也是用类似的方法外推，所以应该也有这个现象，而且还真有：https://blog.csdn.net/v_JULY_v/article/details/136656918 <br>
这应该算是kv只存储自己token/下一个token信息的力证了吧(query dependent compression)<br>
从这个现象来看的话，之后实验的稀疏方法只能治标，而【Mem】Token能治本<br>



代码检查完毕，commit!<br>
~~评估的代码有点向屎山的方向前进了~~