---
title: LLM分布式训练3---模型并行
date: 2025-04-24 00:00:00
updated: 2026-10-26 00:00:00
tags:
  - LLM
  - 分布式训练
  - 张量并行
  - 流水线并行
abbrlink: 212a16ef
---

> 作者：@同济大学 刘越
> Github ID：@miracle-techlink
> 联系邮箱：[miracle.techlink@gmail.com](mailto:miracle.techlink@gmail.com)
> 校内邮箱: [2254018@tongji.edu.cn](mailto:2254018@tongji.edu.cn)

**本篇在系列中的位置**。[第二篇](/posts/3a50363.html)的 ZeRO 解决了参数状态的分片,但它有两个盲区:**单层矩阵大到一张卡算不动**(比如 4096×16384 的 FFN 矩阵,bf16 就是 128MB,加上中间激活,单卡算不动),以及**激活显存随 batch/seq 增长**没人管。这时候就要切模型本身——这就是模型并行。

本篇讲两条切法:**张量并行**(横着切矩阵,每卡算矩阵的一部分)、**流水线并行**(竖着切层,每卡算几层)。前者通信频繁细粒度,必须住 NVLink 域;后者通信稀疏粗粒度,可以跨机。最后给出它们与数据并行组合的 3D 布局,这是千卡训练的标准答案。

## 一、张量并行:把一个矩阵乘拆开

### 1.1 问题:单层矩阵太大

Transformer 的 FFN 层:$Y = \text{GELU}(XW_1)W_2$,其中 $W_1 \in \mathbb{R}^{d \times 4d}$,$W_2 \in \mathbb{R}^{4d \times d}$。

对 8B 模型($d=4096$):$W_1$ 是 $4096 \times 16384$,bf16 存 128MB。前向时中间激活 $\text{GELU}(XW_1)$ 形状 $B \times L \times 16384$,batch=4、seq=4096 时:$4 \times 4096 \times 16384 \times 2\text{B} = 512$MB。

加上反向的梯度、优化器状态,单卡算这一层就要 2GB 显存。如果模型有 32 层,光 FFN 就 64GB——超 H100 显存了。

**解决思路**:把 $W_1$、$W_2$ 切开,每卡只算一部分。

### 1.2 列切(Column Parallel)

把 $W$ 按列切成两半:

$$
W = [W_1, W_2], \quad W_i \in \mathbb{R}^{k \times n/2}
$$

前向:

$$
Y = XW = X[W_1, W_2] = [XW_1, XW_2]
$$

**性质**:
- 卡 $i$ 只需完整输入 $X$ 和自己的 $W_i$,产出 $Y$ 的第 $i$ 半列($Y_i \in \mathbb{R}^{m \times n/2}$);
- 输出在**列维上是切的**($Y_1, Y_2$ 分属两卡),下游若直接接逐元素运算(如 GELU),无需通信;
- 通信点在前一个算子:每卡都要完整的 $X$。

**代码示例**:

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        world_size = dist.get_world_size()
        # 每卡持有一半的输出维度
        self.out_features_per_partition = out_features // world_size
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_partition)
        )
    
    def forward(self, x):
        # x: [batch, in_features](完整输入)
        # weight: [in_features, out/2](列切)
        y_partial = torch.matmul(x, self.weight)  # [batch, out/2]
        return y_partial  # 各卡返回自己的一半

# 使用
layer = ColumnParallelLinear(4096, 16384).cuda()
x = torch.randn(4, 1024, 4096).cuda()  # [batch, seq, d]
y = layer(x)  # [batch, seq, 8192](每卡一半)
```

### 1.3 行切(Row Parallel)

把 $W$ 按行切、$X$ 按列切:

$$
W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}, \quad X = [X_1, X_2]
$$

前向:

$$
Y = XW = [X_1, X_2]\begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = X_1W_1 + X_2W_2
$$

**性质**:
- 卡 $i$ 持 $X_i$(输入的列片,$\in \mathbb{R}^{m \times k/2}$)与 $W_i$(权重的行片,$\in \mathbb{R}^{k/2 \times n}$),各算出一个部分和 $X_iW_i$;
- 部分和需要相加:一次 **AllReduce** 得到完整 $Y$;
- 输出在行维上是**完整的**(每卡得到全量 $Y$ 的同一份)。

**代码示例**:

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        world_size = dist.get_world_size()
        # 每卡持有一半的输入维度
        self.in_features_per_partition = in_features // world_size
        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_partition, out_features)
        )
    
    def forward(self, x_partial):
        # x_partial: [batch, in/2](列切输入)
        y_partial = torch.matmul(x_partial, self.weight)  # [batch, out]
        # 关键:AllReduce 把部分和加总成完整输出
        dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
        return y_partial  # [batch, out](完整输出)

# 使用(接在 ColumnParallelLinear 后)
layer = RowParallelLinear(16384, 4096).cuda()
x_partial = ...  # 来自 ColumnParallelLinear 的输出 [batch, seq, 8192]
y = layer(x_partial)  # [batch, seq, 4096](完整输出)
```

### 1.4 Megatron 的 Transformer 块:列切与行切共轭

Megatron-LM(Shoeybi et al. 2019)的核心设计:把 FFN 的两个矩阵分别做成列切和行切,使一个 Transformer 块内**只需要两次前向 AllReduce**:

```
输入 X [batch, seq, d](完整)
    ↓
ColumnParallelLinear: XW₁,列切
    ↓
[batch, seq, 4d/2](每卡一半)
    ↓
GELU(逐元素,无通信)
    ↓
RowParallelLinear: (GELU 输出)W₂,行切
    ↓ AllReduce ← ①
[batch, seq, d](完整输出)
```

注意力层同理:QKVO 四个投影按头列切(每卡持有部分头),输出投影行切。

**通信量计算**:

每层前向 2 次 AllReduce,反向对称 2 次,共 4 次。每次通信量(TP 度 $t$):

$$
V_{\text{layer}} = 2 \times (\text{激活大小}) \times \frac{t-1}{t}
$$

8B 模型、batch=4、seq=4096、$t=8$:激活 $4 \times 4096 \times 4096 \times 2\text{B} = 134$MB,每层 4 次 AllReduce,总通信 $4 \times 134 \times 7/8 = 469$MB。

### 1.5 TP 的适用半径:为什么必须住 NVLink

TP 的通信特征:**每层 4 次、逐层发生、细粒度**。这决定了它必须放在**卡内 NVLink/NVSwitch 域**:

| 介质 | 单向带宽 | 每层 4 次 AllReduce 时间(8B 模型) |
| --- | --- | --- |
| NVLink 4(H100) | 450 GB/s | 469MB / 450GB/s ≈ 1ms |
| PCIe 4.0 x16(4090) | 32 GB/s | 469MB / 32GB/s ≈ 15ms |
| IB 400G(跨机) | 25 GB/s | 469MB / 25GB/s ≈ 19ms |

如果 TP 跨机,每层通信 19ms,32 层就是 608ms——比计算时间(约 100ms)还长 6 倍,完全不可行。

**结论**:TP 度一般取 2/4/8(单机卡数),**绝不跨机**。

### 1.6 TP 的显存红利

TP 不仅切参数,还切激活:

- **参数**:每卡持 $1/t$ 的权重、梯度、优化器状态(与 ZeRO 正交,可以叠加);
- **激活**:每卡只持自己那份头的 KV cache 与 FFN 中间激活,显存降到 $1/t$。

8B 模型、$t=8$:FFN 中间激活从 512MB 降到 64MB,这对大 batch 长序列是显著的显存红利。

### 1.7 序列并行(SP):把 TP 的"其余激活"也切了

TP 块内的 LayerNorm/Dropout 作用于完整序列,激活不切,成为 TP 下的显存残留。

Megatron-SP(Korthikanti et al. 2022)把 LayerNorm 与 Dropout 的激活沿**序列维**再切:

- 进 TP 块前 **ReduceScatter**(把完整序列散成序列片,每卡持 $L/t$);
- 出 TP 块后 **AllGather**(把序列片拼回完整序列);
- 两次操作恰好**替换**掉原来的两次 AllReduce(通信量不变),LayerNorm 激存降到 $1/t$。

**代码示意**:

```python
class TensorParallelWithSP(nn.Module):
    def forward(self, x):
        # x: [batch, seq, d](完整序列)
        
        # ReduceScatter:序列维切分
        x_chunk = x.chunk(world_size, dim=1)[rank]  # [batch, seq/t, d]
        
        # LayerNorm(只在序列片上算,激活显存降到 1/t)
        x_norm = self.layernorm(x_chunk)
        
        # TP 块(列切 + 行切)
        x_tp = self.tp_block(x_norm)
        
        # AllGather:序列维拼回
        dist.all_gather(x_full, x_tp, dim=1)
        
        return x_full  # [batch, seq, d](完整序列)
```

实测(530B 模型、batch=1、seq=2048):TP+SP 使激活显存从 108GB 降到 13.5GB,再叠加选择性重计算,降到 5.4GB——进入可行区间。

## 二、流水线并行:把模型切成段

### 2.1 基本形态与气泡

把 $L$ 层模型按顺序切成 $p$ 段,第 $i$ 段放第 $i$ 组卡;数据以 **micro-batch**(共 $m$ 个)粒度流入。各段像流水线一样接力:段间传输的是激活(前向)与梯度(反向)。

**通信量**:段间传输激活 $O(B \times L \times d)$,**粒度粗、频率低**(每 micro-batch 每段边界 1 次),可以跨机走 IB/RoCE——这与 TP 恰好互补。

**气泡**:流水线的固有空闲。朴素调度(填满-排空,GPipe):所有 micro-batch 先全部前向、再全部反向,每段耗时 $t_f + t_b$(单 micro-batch),总时间:

$$
T_{\text{GPipe}} = (m + p - 1)(t_f + t_b)
$$

完美流水的时间是 $m(t_f+t_b)$,所以**气泡时间**:

$$
T_{\text{bubble}} = (p-1)(t_f + t_b), \quad
\text{气泡率} = \frac{p-1}{m + p - 1}
$$

**工程含义**:提高 $m$(micro-batch 数)能把气泡摊薄——$m \gg p$ 时气泡率 $\to (p-1)/m$。

### 2.2 一个具体例子

8 段流水线、16 个 micro-batch、单 micro-batch 前向 10ms 反向 20ms:

$$
T_{\text{GPipe}} = (16 + 8 - 1) \times 30 = 690\text{ms}
$$

完美流水:$16 \times 30 = 480$ms,气泡 $210$ms,气泡率 $210/690 = 30\%$。

如果 $m=64$:气泡率 $(8-1)/(64+7) = 9.9\%$——micro-batch 数翻倍,气泡率砍半。

### 2.3 1F1B:同样的气泡,省一半激活驻留

GPipe 要保留全部 $m$ 份激活等反向,显存峰值 $O(m)$。

PipeDream-Flush(1F1B)在稳态交替执行一前向一反向:前向填充到第 $p$ 个 micro-batch 后,每来一个新前向就配一个反向,激活驻留从 $m$ 份降到约 $p$ 份。

**对比**:

| 调度 | 气泡时间 | 激活峰值 |
| --- | --- | --- |
| GPipe | $(p-1)(t_f+t_b)$ | $m \times M_B$ |
| 1F1B | $(p-1)(t_f+t_b)$(相同) | $p \times M_B$(降 $m/p$ 倍) |

8 段、64 micro-batch:GPipe 激活峰值 64 份,1F1B 只有 8 份——**显存降 8 倍,气泡不变**。

代价:1F1B 调度依赖更紧,完成时间通常比 GPipe 慢约 3%(中间结果与缓冲区的显存峰值从 2.57GB 降到 0.97GB,数据来自《AI Infra》书稿的 Qwen3-8B 算例)。

### 2.4 零气泡:把反向拆成 B 和 W

反向的输入梯度 $dX$(记 B)必须尽早传给上一段(在关键路径上);权重梯度 $dW$(记 W)只用于本段的优化器累积,**可以延后**。

设每段前向、B、W 耗时 $T_F, T_B, T_W$:

$$
\text{1F1B 气泡} = (p-1)(T_F + T_B + T_W)
$$

$$
\text{ZB-H1: 把 } W \text{ 填进空隙} \Rightarrow (p-1)(T_F + T_B - T_W)
$$

$$
\text{ZB-H2: 再提前更多前向} \Rightarrow (p-1)(T_F + T_B - 2T_W)
$$

$T_B = T_W$ 时 ZB-H1 的气泡恰好是 1F1B 的三分之一、ZB-H2 下界为零。

**代价**:延后的 $dW$ 迫使靠后阶段把激活多留几个 micro-batch,显存回升——**时间与容量两根轴没有同时占优的调度**。

实测(4 段、8 micro-batch、$T_F=T_B=T_W=10$ms):

| 调度 | 完成时间 | 每卡空闲 | 阶段 0 峰值 |
| --- | --- | --- | --- |
| GPipe | 337ms | 90ms | 921 MiB |
| 1F1B | 347ms | 106ms | 921 MiB |
| ZB-H1 | 283ms | 42ms | 921 MiB(阶段 3 为 1226 MiB) |

(数据来自《AI Infra》书稿表 10-4)

### 2.5 两个工程化变体

**交错式 1F1B**:每卡持 $v$ 个不连续层块,气泡率再除以 $v$,代价是跨界通信次数升到 $2p-1$ 次。

**DualPipe**(DeepSeek-V3):从流水线两端相向注入,bubble 为 $(PP/2-1)(F\&B + B - 3W)$,代价是每卡参数翻倍。

### 2.6 PP 的适用半径:可以跨机

PP 的通信特征:**每 micro-batch 每段边界 1 次、粗粒度、低频**。8B 模型、micro-batch=4、seq=4096:段间激活 134MB,IB 25GB/s 下 5.4ms。

如果计算时间 100ms,通信占比 5%——完全可以跨机。这就是为什么 PP 是跨机并行的首选。

## 三、3D 并行:组合的艺术

### 3.1 三条原则

单一大模型训练同时用三种并行,总卡数 $= d_{\text{dp}} \times t_{\text{tp}} \times p_{\text{pp}}$。布局由各并行的通信特征决定:

| 并行 | 通信特征 | 量级 | 放哪里 |
| --- | --- | --- | --- |
| TP | 每层 4 次 AllReduce,细粒度高频 | 最高带宽需求 | 卡内 NVLink 域(单机 8 卡) |
| PP | 每层边界 1 次点对点,粗粒度低频 | 中 | 可跨机(IB/RoCE) |
| DP/ZeRO | 每步 $1{\sim}1.5\times$ 梯度/参数集合通信 | 随卡数增长 | 最外层,规模可巨大 |

**三条经验原则**:

1. **TP 贴 NVLink**:$t$ = 单机卡数(8),绝不跨机;
2. **PP 跨机**:段间通信稀疏,吃得住网络时延;
3. **DP 最外层兜规模**:剩下的卡全给数据并行,配 ZeRO-1/2 把状态分片摊掉。

### 3.2 一个完整的布局案例

**任务**:512 卡 H100 训练 70B 模型。

**约束分析**:
- 参数状态:$16 \times 70\text{B} = 1120$GB,单卡 80GB 装不下,必须分片;
- 单层矩阵:$d=8192$,FFN $8192 \times 32768$,bf16 512MB,单卡算得动但显存紧张;
- 激活:batch=8、seq=8192 时,$8 \times 8192 \times 8192 \times 34 \times 80 = 1.4$TB,必须切。

**布局**:
- TP=8(机内):切单层矩阵,激活降到 $1/8$;
- PP=8(跨 8 机):切 80 层成 8 段,每段 10 层;
- DP=8(64 机):数据并行,配 ZeRO-1 分片优化器状态。

**显存核算**(每卡):
- 参数状态:$16 \times 70\text{B} / (8 \times 8) = 17.5$GB(TP 切参数,ZeRO-1 切优化器);
- 激活:$1.4\text{TB} / (8 \times 8) = 21.9$GB(TP 切激活,PP 每段只存自己 10 层);
- 合计 39.4GB < 80GB,可行。

**通信核算**:
- TP:每层 4 次 AllReduce,469MB/层,NVLink 450GB/s,1ms/层,10 层 10ms;
- PP:每 micro-batch 段间 134MB,IB 25GB/s,5.4ms,10 段 54ms;
- DP:每步梯度 AllReduce,ZeRO-1 后 $2 \times 70\text{B}/8 \times 7/8 = 15.3$GB,IB 25GB/s,612ms。

计算时间(70B、batch=8、seq=8192、6ND):$6 \times 70\text{B} \times 8192 / (512 \times 2000\text{TFLOPS} \times 0.4) = 840$ms。

通信占比:$(10 + 54 + 612) / 840 = 80\%$——**太高了,必须把 DP 的梯度 AllReduce 藏起来**(梯度分桶 + 通信重叠,能隐藏 80%,实际占比降到 16%)。

### 3.3 为什么是这个布局

**TP=8**:H100 单机 8 卡 NVLink,TP=8 恰好用满机内带宽;

**PP=8**:80 层切 8 段,每段 10 层,段间通信稀疏,跨机可接受;

**DP=8**:剩余 64 机做数据并行,ZeRO-1 把优化器状态降到 $1/8$,梯度 AllReduce 用梯度桶隐藏。

如果换成 TP=4、PP=16、DP=4:TP 只用一半机内带宽浪费资源,PP=16 段间通信翻倍,DP=4 梯度通信量翻倍——更差。

## 四、激活重计算:用计算换显存

### 4.1 原理

前向时**不保存**中间激活,反向时**重新前向**算一遍。约 33% 额外计算换激活显存降一个量级。

### 4.2 选择性重计算

不是全部重算,只重算"大头"(注意力、FFN),保留"小头"(LayerNorm、Dropout)。

Megatron 的选择性重计算:530B 模型、batch=1、seq=2048,激活从 13.5GB(TP+SP 后)降到 5.4GB,代价是约 30% 额外计算。

### 4.3 一个具体的权衡

8B 模型、batch=4、seq=4096:

- **不重算**:激活 7.4GB,计算时间 100ms;
- **全重算**:激活 0.2GB(只存输入/输出),计算时间 133ms(多 33%);
- **选择性重算**:激活 2.1GB,计算时间 115ms(多 15%)。

如果显存紧张(24GB 卡),选择性重算是甜点;如果显存充裕(H100 80GB),不重算更快。

## 小结

- **张量并行**:列切免通信进逐元素算子、行切部分和 AllReduce;Megatron 块每层前向 2 次 + 反向 2 次 AllReduce,必须住 NVLink 域;SP 用序列切分消掉 LayerNorm 激活,通信量不变;
- **流水线并行**:气泡 $(p-1)(t_f+t_b)$、气泡率 $(p-1)/(m+p-1)$;1F1B 气泡不变但激活 $m \to p$;ZB-H1/H2 把 $dW$ 延后填隙,气泡压到 $1/3$ 至 0,代价是激活驻留;PP 可跨机;
- **3D 布局**:TP 卡内、PP 跨机、DP 最外,ZeRO 兜状态——三者通信特征互补,组合才是千卡训练的完整答案;
- **激活重计算**:33% 额外计算换激活显存降一个量级,选择性重计算是甜点。

下一篇:[LLM分布式训练4——集群架构:从 NVLink 到 NCCL](/posts/74a233f5.html)
