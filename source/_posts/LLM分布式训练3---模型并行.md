---
title: LLM分布式训练3---模型并行
date: 2025-04-24 00:00:00
updated: 2026-10-25 00:00:00
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

[第一篇](/posts/9aac52d9.html)的 16 字节账单说明:数据并行无论加多少卡,每卡都要完整状态;[第二篇](/posts/3a50363.html)的 ZeRO 解决了状态分片,但通信量随数据并行度上升,且**激活显存**始终没人管。当单层矩阵大到一张卡算不动、或一个模型长到一张卡放不下时,就要切模型本身——这就是模型并行。本篇讲两条切法:横着切矩阵(张量并行)、竖着切层(流水线并行),最后给出它们与数据并行组合的 3D 布局。

## 一、张量并行:切一个矩阵乘

### 列切(Column Parallel)

考虑 $Y = XW$,其中 $W \in \mathbb{R}^{k \times n}$。把 $W$ 按列切成两半 $W = [W_1, W_2]$,两张卡各持一半:

$$
Y = X[W_1, W_2] = [XW_1,\; XW_2]
$$

- 卡 $i$ 只需完整输入 $X$ 和自己的 $W_i$,产出 $Y$ 的第 $i$ 半列;
- 输出在**列维上是切的**($Y_1, Y_2$ 分属两卡),下游若直接接逐元素运算(如 GELU),无需通信;
- 通信点在前一个算子:每卡都要完整的 $X$,即前一层输出若在行维上分布,需要一次 AllGather(或由上游的行切直接配合,见下)。

### 行切(Row Parallel)

把 $W$ 按行切、$X$ 按列切:

$$
Y = XW = [X_1, X_2]\begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = X_1W_1 + X_2W_2
$$

- 卡 $i$ 持 $X_i$(输入的列片)与 $W_i$(权重的行片),各算出一个部分和;
- 部分和需要相加:一次 **AllReduce**(或 ReduceScatter)得到完整 $Y$;
- 输出在行维上是**完整的**(每卡得到全量 $Y$ 的同一份)。

### Megatron 的 Transformer 块:列切与行切共轭

Megatron-LM(Shoeybi et al. 2019)把注意力头的输出投影做成"列切"、FFN 的第二个矩阵做"行切",使一个 Transformer 块内**只需要两次前向 AllReduce**:

```
       ┌─ 注意力:QKVO 四个投影按头列切(每卡持有部分头)
AllReduce  ←①
       └─ FFN:第一个矩阵列切(每卡算自己的升维),第二个矩阵行切(部分和)
AllReduce  ←②
```

反向对称地再各一次 AllReduce。结论:**张量并行度 $t$ 时,每层前向+反向共 4 次 AllReduce,每次通信量 $2\Psi_{\text{layer}}(t-1)/t$**(BF16)。

这决定了 TP 的适用半径:

1. **通信频繁且细粒度**(每层 4 次、逐层发生),必须放在**卡内 NVLink/NVSwitch 域**——NVLink 单向 450–900 GB/s 级,PCIe 4.0 x16 只有 32 GB/s,差一个数量级;
2. TP 度一般取 2/4/8(单机卡数),**不跨机**;跨机做 TP 的方案(DeepSpeed-Ulysses 类序列切分除外)通信必被网络时延吞掉;
3. TP 同时切了激活:每卡只持自己那份头的 KV 与中间激活,对大 batch 长序列是显著的显存红利。

### 序列并行(SP):把 TP 的"其余激活"也切了

TP 块内的 LayerNorm/Dropout 作用于完整序列,激活不切,成为 TP 下的显存残留。Megatron-SP(Korthikanti et al. 2022)把 LayerNorm 与 Dropout 的激活沿**序列维**再切:

- 进 TP 块前 ReduceScatter(把完整序列散成序列片),出 TP 块后 AllGather;
- 两次操作恰好**替换**掉原来的两次 AllReduce(通信量不变),LayerNorm 激存降到 $1/t$;
- 这篇论文同时给出激活的完整显存公式与重计算组合:TP+SP+选择性重计算使 530B 训练的激活显存进入可行区间。

## 二、流水线并行:切层

### 基本形态与气泡

把 $L$ 层模型按顺序切成 $p$ 段,第 $i$ 段放第 $i$ 组卡;数据以 **micro-batch**(共 $m$ 个)粒度流入。各段像流水线一样接力:段间传输的是激活(前向)与梯度(反向),通信量 $O(\text{batch} \times \text{hidden})$,**粒度粗、频率低**,可以跨机走 IB/RoCE——这与 TP 恰好互补。

流水线的固有空闲称为**气泡**。朴素调度(填满-排空,GPipe):所有 micro-batch 先全部前向、再全部反向,每段耗时 $t_f + t_b$(单 micro-batch),总时间:

$$
T_{\text{GPipe}} = (m + p - 1)(t_f + t_b)
$$

完美流水的时间是 $m(t_f+t_b)$,所以**气泡时间**为:

$$
T_{\text{bubble}} = (p-1)(t_f + t_b), \qquad
\text{气泡率} = \frac{p-1}{m + p - 1}
$$

工程含义:提高 $m$(micro-batch 数)能把气泡摊薄——$m \gg p$ 时气泡率 $\to (p-1)/m$;$p$ 越深气泡越重,单机 8 卡 PP 通常不超过 4–8 段。

### 1F1B:同样的气泡,省一半激活驻留

GPipe 要保留全部 $m$ 份激活等反向,显存峰值 $O(m)$。PipeDream-Flush(1F1B)在稳态交替执行一前向一反向:前向填充到第 $p$ 个 micro-batch 后,每来一个新前向就配一个反向,激活驻留从 $m$ 份降到约 $p$ 份:

- **气泡不变**:仍是 $(p-1)(t_f+t_b)$;
- **显存换稳定**:激活峰值从 $mM_B$ 降到 $pM_B$($M_B$ 为单 micro-batch 激活);代价是调度依赖更紧,完成时间通常比填满排空慢约 3%。

### 零气泡:把反向拆成 B 和 W

反向的输入梯度 $dX$(记 B)必须尽早传给上一段(在关键路径上);权重梯度 $dW$(记 W)只用于本段的优化器累积,**可以延后**。设每段前向、B、W 耗时 $T_F, T_B, T_W$:

$$
\text{1F1B 气泡} = (p-1)(T_F + T_B + T_W)
$$

$$
\text{ZB-H1: 把 } W \text{ 填进空隙} \Rightarrow (p-1)(T_F + T_B - T_W)
$$

$$
\text{ZB-H2: 再提前更多前向} \Rightarrow (p-1)(T_F + T_B - 2T_W)
$$

$T_B = T_W$ 时 ZB-H1 的气泡恰好是 1F1B 的三分之一、ZB-H2 下界为零。代价:延后的 $dW$ 迫使靠后阶段把激活多留几个 micro-batch,显存回升——**时间与容量两根轴没有同时占优的调度**,选择取决于约束在哪边。

两个工程化变体:**交错式 1F1B**(每卡持 $v$ 个不连续层块,气泡率再除以 $v$,代价是跨界通信次数升到 $2p-1$ 次)与 **DualPipe**(DeepSeek-V3,从流水线两端相向注入,bubble 为 $(PP/2-1)(F\&B + B - 3W)$,代价是每卡参数翻倍)。

## 三、3D 并行:布局的三条原则

单一大模型训练同时用三种并行,总卡数 $= d_{\text{dp}} \times t_{\text{tp}} \times p_{\text{pp}}$。布局由各并行的通信特征决定:

| 并行 | 通信特征 | 量级 | 放哪里 |
| --- | --- | --- | --- |
| TP | 每层 4 次 AllReduce,细粒度高频 | 最高带宽需求 | 卡内 NVLink 域(单机 8 卡) |
| PP | 每层边界 1 次点对点,粗粒度低频 | 中 | 可跨机(IB/RoCE) |
| DP/ZeRO | 每步 $1{\sim}1.5\times$ 梯度/参数集合通信 | 随卡数增长 | 最外层,规模可巨大 |

三条经验原则:

1. **TP 贴 NVLink**:$t$ = 单机卡数(8),绝不跨机;
2. **PP 跨机**:段间通信稀疏,吃得住网络时延;同一"流水线复制组"内各段尽量在不同机;
3. **DP 最外层兜规模**:剩下的卡全给数据并行,配 ZeRO-1/2 把状态分片摊掉。

以 512 卡 H100 训练类 GPT 模型的典型布局:$t=8$(机内), $p=8$(跨 8 机), $d=8$(64 机);GPT-3 的 10K V100 集群即 DP×MP 的朴素版本,现代栈(Megatron + DeepSpeed)则是 TP+PP+ZeRO-DP 的组合。

## 小结

- 张量并行:列切免通信进逐元素算子、行切部分和 AllReduce;Megatron 块每层前向 2 次 + 反向 2 次 AllReduce,必须住 NVLink 域;SP 用序列切分消掉 LayerNorm 激活,通信量不变;
- 流水线并行:气泡 $(p-1)(t_f+t_b)$、气泡率 $(p-1)/(m+p-1)$;1F1B 气泡不变但激活 $m \to p$;ZB-H1/H2 把 $dW$ 延后填隙,气泡压到 $1/3$ 至 0,代价是激活驻留;
- 3D 布局:TP 卡内、PP 跨机、DP 最外,ZeRO 兜状态——三者通信特征互补,组合才是千卡训练的完整答案。

下一篇:[LLM分布式训练4——集群架构:从 NVLink 到 NCCL](/posts/74a233f5.html)
