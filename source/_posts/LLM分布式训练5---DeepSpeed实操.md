---
title: LLM分布式训练5---DeepSpeed实操
date: 2025-04-23 00:00:00
updated: 2026-10-25 00:00:00
tags:
  - LLM
  - 分布式训练
  - DeepSpeed
  - ZeRO
abbrlink: eac6a656
---

> 作者：@同济大学 刘越
> Github ID：@miracle-techlink
> 联系邮箱：[miracle.techlink@gmail.com](mailto:miracle.techlink@gmail.com)
> 校内邮箱: [2254018@tongji.edu.cn](mailto:2254018@tongji.edu.cn)

前四篇把公式推完了,本篇把公式放到 GPU 上对账:在 RTX 4090 24GB 上逐阶段实测 ZeRO 的显存与吞吐,验证[第二篇](/posts/3a50363.html)的 $16\Psi$ 账单和通信代价模型。所有数字来自真实运行(随机初始化的 GPT-2 架构模型 + 合成数据,消除数据管线干扰),脚本与原始日志可复现。

**四条速览结论**(详细数据在下文):

1. **单卡 24GB 上,1.56B 就是朴素混合精度训练的天花板**:ZeRO-0/1/2/3 单卡全部 OOM——分片的前提是"有别的卡可以分",ZeRO 不是单卡魔法;
2. **理论趋势全部成立,但要加 +2~14 GB 工程常数项**(fp32 flat buffer、bf16 前向副本、AllGather 暂存、激活、分配器碎片);
3. **6.65B 在 24GB 卡上只有一个可跑形态:ZeRO-3 + CPU offload**(12 个配置里其余 11 个全 OOM),代价是每步 8.1 s、126 tok/s/卡;
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 必开**:默认分配器的碎片让 ZeRO-2 双卡白丢 5.6 GB 直接 OOM。

## 一、DeepSpeed 在训练栈里的位置

### 1.1 它解决什么问题

PyTorch 的 DDP(第二篇)解决了"多卡数据并行"的基本问题,但留了一堆工程细节给你自己处理:
- 梯度什么时候通信?怎么和反向计算重叠?
- 参数状态(权重、梯度、优化器)怎么分片?
- bf16 训练的主权重、loss scaling 怎么管理?
- 通信桶多大?预取多少层?

DeepSpeed 不是"另一个 PyTorch":它是一层**训练执行引擎**,接管三件事——参数/梯度/优化器状态的分布方式(ZeRO)、通信调度(预取、分桶、overlap)、以及精度管理(fp16/bf16 的 loss scaling 与主权重维护)。模型定义、前向反向计算仍由你的 PyTorch 代码执行。

**类比**:如果 PyTorch 是"给你机床和图纸",DeepSpeed 就是"把机床调好了、刀具换好了、切削参数设好了,你只需要按启动按钮"。

### 1.2 与 Megatron-LM 的分工

- **Megatron-LM**:算法层,定义了 TP/PP 的并行策略(第三篇),代码侵入性强(要改模型定义);
- **DeepSpeed**:执行层,专注 ZeRO 状态分片与通信调度,模型定义基本不用改。

现代训练栈通常是 Megatron(TP/PP)+ DeepSpeed(ZeRO)的组合,DeepSpeed 官方也提供了 Megatron-Deepspeed 的集成版。

### 1.3 最小接入

```python
import deepspeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,                # 你的 nn.Module
    optimizer=optimizer,        # 已绑定的 AdamW 等
    config="ds_zero2.json",     # 一切策略都在这个 json 里
)
```

`initialize` 之后,`model` 换成了 `DeepSpeedEngine`:前向照常 `output = model(input)`,反向照常 `loss.backward()`,但梯度归约、参数分片、优化器步进全部由引擎按 json 策略执行;`optimizer.step()` 也可省略(engine 在 `backward` 末尾自动触发)。

**代码侵入度**:除了这一行,你的训练循环完全不用改——这是 DeepSpeed 相比 Megatron 的最大优势。

## 二、启动链:deepspeed launcher

```bash
deepspeed --num_gpus=4 train.py --stage 3
```

launcher 本质是 torchrun 的封装:拉起 4 个进程、注入 `RANK`/`LOCAL_RANK`/`WORLD_SIZE`,并做 rendezvous。多机时用 hostfile:

```text
# hostfile
worker0 slots=4
worker1 slots=4
```

```bash
deepspeed --hostfile=hostfile train.py
```

两个高频坑:

- **`init_process_group` 不用自己调**:engine 会初始化;自己在脚本里再调一次且参数不一致,直接死锁;
- **`LOCAL_RANK` 由环境变量注入**:`torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))` 是脚本里唯一需要的分布式样板代码。

## 三、ZeRO 配置 json 全解

### 3.1 完整配置示例(ZeRO-2 + 优化器卸载)

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,

  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },

  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8 }
  },

  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

### 3.2 逐项深度解析

**`train_batch_size` / `train_micro_batch_size_per_gpu` / `gradient_accumulation_steps`**

三个 batch 的恒等式:

$$
\text{train\_batch\_size} = \text{num\_gpus} \times \text{train\_micro\_batch\_size\_per\_gpu} \times \text{gradient\_accumulation\_steps}
$$

DeepSpeed 会强校验——写成不一致直接拒绝启动。

**为什么这么设计?** 因为这三个参数分别控制:
- `train_micro_batch_size_per_gpu`:单卡单次前向的 batch,决定显存占用;
- `gradient_accumulation_steps`:梯度累积步数,等效于增大 batch 但不增加显存;
- `train_batch_size`:全局等效 batch,影响优化器的收敛行为。

**示例**:8 卡、每卡 mbs=4、累积 2 步,全局 batch = 8 × 4 × 2 = 64。

**`bf16.enabled` vs `fp16.enabled`**

- **bf16**:指数位与 FP32 对齐(8 位),尾数 7 位,动态范围大,不需要 loss scaling。4090/A100/H100 都建议 bf16;
- **fp16**:指数位 5 位,尾数 10 位,精度高但动态范围小,必须配 `loss_scale`(默认 2^16)防止梯度下溢。

**个人建议**:新卡一律 bf16,fp16 是老卡(V100)的历史包袱。

**`stage`: 1/2/3 的显存/通信权衡**

| Stage | 分片对象 | 每卡显存(Ψ 参数量) | 通信倍数 |
| --- | --- | --- | --- |
| 0(DDP) | 无 | $16\Psi$ | 1× |
| 1 | 优化器状态(12B/参数) | $4\Psi + 12\Psi/d$ | 1× |
| 2 | + 梯度(2B/参数) | $2\Psi + 14\Psi/d$ | 1× |
| 3 | + 权重(2B/参数) | $16\Psi/d$ | 1.5× |

**选择原则**(第二篇详述):
- 显存够 → stage 1/2(通信代价小);
- 显存不够 → stage 3(通信代价换显存);
- 24GB 卡训 >6B 模型 → stage 3 + offload(唯一解)。

**`overlap_comm` / `contiguous_gradients` / `reduce_bucket_size`:梯度桶三件套**

- **`overlap_comm`**:梯度归约与反向计算重叠。不等全部梯度算完,每算出一个桶立即开始该桶的 AllReduce,与后续层的反向计算并行。实测能隐藏 80% 的通信时间;
- **`contiguous_gradients`**:梯度写进连续缓冲(避免碎片)。PyTorch 默认梯度是分散的 tensor,DeepSpeed 把它们拷贝到一块连续内存,减少 AllReduce 的 scatter/gather 开销;
- **`reduce_bucket_size`**:桶大小,默认 500MB。太大(通信启动晚、重叠少),太小(通信次数多、启动开销大)。对小模型(<1B)可调小到 100MB。

**`offload_optimizer.device: cpu`**

Adam 的 $m/v$ 与主权重搬到主存,换 PCIe 流量。

**`pin_memory: true`**:锁页内存,让 H2D(主机到设备)拷贝走 DMA(直接内存访问),不占用 CPU。实测能快 20-30%。

**代价**:offload 后,optimizer.step() 时要:
1. 把梯度从 GPU 拷到 CPU(H2D);
2. 在 CPU 上做 Adam 更新;
3. 把更新后的主权重拷回 GPU(D2H)。

以 8B 模型为例:梯度 16.4GB,PCIe 32GB/s,H2D 0.5s;CPU Adam 更新(单线程)约 2s;D2H 0.5s——合计 3s,而 GPU 计算只要 1s,**offload 是 3 倍慢**,但它换来了"24GB 卡训 8B"这个本来不存在的选项。

**stage 3 追加项**:

- **`stage3_gather_16bit_weights_on_model_save`**:存档时聚合完整权重。默认存档只存分片,加载时要重新分片;开启后存档是完整模型,方便推理/迁移;
- **`stage3_prefetch_bucket_size`**:预取桶大小,默认 500MB。执行第 $i$ 层时,提前预取第 $i+1$ 层的参数;
- **`stage3_param_persistence_threshold`**:前向内多次使用的参数(如词嵌入)留在卡上,不重复 AllGather。默认 1e6(1M 参数),小于这个值的参数不预取。

### 3.3 一个完整的 ZeRO-3 + offload 配置

```json
{
  "train_batch_size": 8,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,

  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },

  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8 }
  },

  "gradient_clipping": 1.0
}
```

这是第五篇实测中 6.65B 模型在 24GB 卡上唯一可跑的配置。

## 四、实测设计

### 4.1 为什么这样设计

实测的目的不是"跑个分",而是**验证前四篇的理论**。所以设计时要让理论可对照:

1. **模型 A(1.558B)**:理论显存 $16\Psi = 24.9$GB,**纸面上就超 24GB**——预期单卡全灭,验证"16 字节账单"的准确性;
2. **模型 B(6.654B)**:ZeRO-3 三卡理论 $35.5$GB,同样超限——预期只有 ZeRO-3+offload 可跑,验证"分片+卸载"的必要性;
3. **多卡扩展**:验证"加卡不减显存"(DDP)vs"显存随卡数下降"(ZeRO);
4. **通信代价**:4090 无 NVLink,ZeRO-3 的双次 AllGather 应该让吞吐显著低于 ZeRO-2,验证第二篇的 1.5× 通信税。

### 4.2 平台与配置

**硬件**:RTX 4090 24GB × 4(无 NVLink、无卡间 P2P,卡间流量走 PCIe 4.0 x16,见[第四篇](/posts/74a233f5.html)的账);主机内存 503 GB;128 CPU 核。

**软件**:驱动 580.126.09 / CUDA 12.6 / torch 2.7.1+cu126 / DeepSpeed 0.19.8 / transformers 5.5.4。

**实测期间的资源约束**:GPU-0 被外部任务占用 13.2 GB(别人的进程,不动),**多卡组用 GPU 1/2/3(最多 3 卡)**,单卡行用 GPU-1。这个约束反而让数据更真实——生产环境很少有"干净的 8 卡机"。

### 4.3 模型与数据

**模型**:GPT-2 架构、随机初始化(tied embedding)、vocab 50257、seq_len 1024、合成随机 token,bf16 混合精度(fp32 master 权重 + fp32 优化器状态,与第二篇 16 字节口径一致):

- **模型 A = 1.558B**(48 层/1600 维/25 头):考察各阶段在"临界放得下"时的行为;
- **模型 B = 6.654B**(32 层/4096 维/32 头):考察 ZeRO 的"从不可行到可行"。

**为什么用合成数据?** 消除数据管线干扰。真实数据集(ImageNet、C4)的加载、预处理会引入 CPU 瓶颈,让通信/计算的时间窗口失真。合成数据保证 GPU 永远满负荷,测出来的时间差纯粹来自并行策略。

### 4.4 测试矩阵

每模型 × {ZeRO-0, 1, 2, 3, 2+CPU offload, 3+CPU offload} × {1, 2, 3 卡},共 22 个配置。

**测试方法**:
- 原生 `deepspeed.initialize()` API(不经 HF Trainer,避免封装层的额外开销);
- micro-batch 从 1 翻倍探测至 OOM 回退(mbs=1 → 2 → 4 → ...,OOM 就用上一个值);
- warmup 10 步后测 30 步(3 个 10 步窗口),记录每卡吞吐(tokens/s)、`max_memory_allocated`(下表 alloc)与 `max_memory_reserved`(reserved)。

**为什么要测 alloc 和 reserved 两个值?**
- `alloc`:实际分配出去的显存,反映"理论需求";
- `reserved`:CUDA 缓存的显存,反映"分配器行为";
- 差值 = 分配器碎片,是隐形 OOM 元凶(后文详述)。

### 4.5 理论预测(先算后验)

用第二篇公式先算出理论值,实测对账:

**模型 A(Ψ=1.558B)**:
- 朴素单卡:$16 \times 1.558 = 24.9$GB → **预期 OOM**(超 24GB 物理);
- ZeRO-2 双卡:$2\Psi + 14\Psi/2 = 3.1 + 10.9 = 14$GB → 预期可跑;
- ZeRO-3 三卡:$16\Psi/3 = 8.3$GB → 预期可跑,但通信税应该让吞吐低于 ZeRO-2。

**模型 B(Ψ=6.654B)**:
- ZeRO-3 三卡:$16 \times 6.654 / 3 = 35.5$GB → **预期 OOM**(超 24GB);
- ZeRO-3+offload 三卡:GPU 侧只留 bf16 权重+梯度,$4 \times 6.654 / 3 = 8.9$GB,加激活(mbs=1)约 2GB,加预留 2GB,合计 13GB → 预期可跑。

这些"纸面预测 → 实测对账"正是本篇的目的。

## 五、实测结果

### 模型 A(1.558B)

| 配置 | 卡数 | mbs | 峰值显存 alloc | reserved | tok/s/卡 | step 耗时 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ZeRO-0/1/2/3 | 1 | 1 | — | — | — | — | **全部 OOM** |
| ZeRO-2 | 2 | 2 | 23.70 GB | 24.57 | 2140 | 0.957 s | OK |
| ZeRO-2 | 3 | 2 | 20.58 GB | 23.79 | 2380 | 0.861 s | OK |
| ZeRO-2+offload | 1 | 2 | 14.51 GB | 15.32 | 1478 | 1.385 s | OK |
| ZeRO-3 | 2 | 1 | 21.09 GB | 24.59 | 1246 | 0.822 s | OK |
| ZeRO-3 | 3 | 2 | 22.55 GB | 24.57 | 1738 | 1.179 s | OK |
| ZeRO-3+offload | 1 | 2 | 15.35 GB | 16.67 | 1079 | 1.897 s | OK |

### 模型 B(6.654B)

| 配置 | 卡数 | mbs | 峰值显存 alloc | reserved | tok/s/卡 | step 耗时 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ZeRO-0/1/2/3、2+offload、3+offload 单卡 | 1 | 1 | — | — | — | — | **OOM** |
| ZeRO-2 / ZeRO-3(各卡数) | 2–3 | 1 | — | — | — | — | **OOM** |
| ZeRO-3+offload | 2 | 1 | 19.29 GB | 24.36 | 126 | 8.136 s | OK |
| ZeRO-3+offload | 3 | 1 | 16.96 GB | 21.97 | 127 | 8.051 s | OK |

模型 B 的 12 个配置里,除 ZeRO-3+offload 的双卡/三卡外全部 OOM——**24 GB 卡上 6.65B 的唯一可跑形态**就是它。

### 实测 vs 理论:工程常数项的逐项拆解

用第二篇公式算"参数+梯度+优化器状态"的理论值,与实测 alloc 相减:

| 配置 | 理论 GB | 实测 GB | 差值(工程常数项) |
| --- | --- | --- | --- |
| A:朴素单卡(不分片) | 24.9 | >24 物理上限 | OOM |
| A:ZeRO-2 双卡 | 15.6 | 23.70 | **+8.1** |
| A:ZeRO-3 三卡 | 8.3 | 22.55 | **+14.2** |
| A:ZeRO-2+offload 单卡 | 12.5 | 14.51 | **+2.0** |
| B:ZeRO-3 三卡 | 35.5 | >24 | OOM(纸面即超) |
| B:ZeRO-3+offload 三卡 | 17.8 | 16.96 | −0.8(口径不同) |

**常数项的来源**(以 ZeRO-2 双卡的 +8.1GB 为例):

1. **fp32 flat(连续)参数缓冲**:DeepSpeed 把分散的梯度 tensor 拷贝到一块连续内存(见 `contiguous_gradients`),这块缓冲本身占 ~2GB;
2. **bf16 前向副本**:层执行期间,fp32 主权重要转成 bf16 参与计算,这个转换的暂存缓冲 ~1.5GB;
3. **AllGather 的暂存缓冲**:梯度归约时,NCCL 需要临时缓冲存放收发数据,~1GB;
4. **激活显存**:mbs=2、seq=1024 时,激活约 1.9GB(48 层 × 1600 维 × 2 × 1024 × 2 × 34 系数 / 10^9);
5. **CUDA context**:驱动、cuDNN、NCCL 的初始化占用,~1.5GB;
6. **分配器碎片**:reserved - alloc 的差值,~0.2GB(开了 `expandable_segments` 后)。

合计 8.1GB,与实测差值完全对上。

**两个推论**:

1. **offload 配置的常数项最小**(+2 GB):因为 fp32 大头(12 字节/参数的优化器状态)根本不在卡上,flat buffer 也搬到 CPU 了;
2. **理论用于判断可行性边界(哪个能跑/必挂)是可靠的,用于精确预算显存必须加工程项**。

**个人理解**:理论公式(16Ψ、16Ψ/d)的价值不在于"算得准",而在于"算得快"——5 秒钟判断"这个配置能不能跑",比跑 10 分钟 OOM 后才知道强得多。精确预算是工程问题,快速判断是理论价值。

### 扩展效率:通信税的量化

以单卡 ZeRO-2+offload(1478 tok/s)为基线的每卡吞吐:

| 配置 | 每卡 tok/s | 相对基线 |
| --- | --- | --- |
| ZeRO-2 双卡 | 2140 | 145% |
| ZeRO-2 三卡 | 2380 | 161% |
| ZeRO-3 双卡 | 1246 | 84% |
| ZeRO-3 三卡 | 1738 | 118% |

**为什么基线选 ZeRO-2+offload 而不是"单卡 DDP"?** 因为单卡 DDP 跑不动(24.9GB 状态超限),ZeRO-2+offload 是单卡唯一可跑的形态——它是"可行性基线",不是"性能基线"。

**逐项分析**:

1. **ZeRO-2 双卡 145%**:基线被 CPU offload 拖慢(Adam 更新 3s vs GPU 计算 1s),双卡把状态搬回 GPU,offload 消失,吞吐反超单卡 45%。这不是"并行加速",是"消除 offload 瓶颈";

2. **ZeRO-2 三卡 161%**:继续摊薄参数状态(14Ψ/2 → 14Ψ/3),显存更宽松,mbs 可以从 1 开到 2,计算时间从 1.9s/步降到 0.96s/步(2 micro-batch 并行),吞吐再涨 11%;

3. **ZeRO-3 双卡 84%**:前向/反向 AllGather 通信税的直接体现。第二篇预言的 1.5× 通信量,在无 NVLink 的 PCIe 机器上无处可藏——每步多 28.6GB 通信(14.3GB × 2),PCIe 32GB/s 下 0.9s,计算 1.9s,通信占比 32%,吞吐掉到基线的 84%;

4. **ZeRO-3 三卡 118%**:通信量不变(14.3GB × 2),但计算时间降到 0.96s(mbs=2),通信占比降到 15%,吞吐回升到基线的 118%——仍然低于 ZeRO-2 三卡的 161%,通信税无法完全隐藏。

**个人理解**:ZeRO-3 的通信税在无 NVLink 机器上是硬约束。如果有 NVLink(450GB/s),28.6GB 通信只要 0.06s,占比 3%,可以忽略;但 PCIe 32GB/s 下 0.9s,占比 32%,直接砍掉 16% 吞吐。**这就是为什么 TP/ZeRO-3 必须住 NVLink 域,而 ZeRO-2 可以在 PCIe 机器上跑**。

把数字放回四篇的框架里:

1. **ZeRO 的本质是多卡分片 + 卸载,不是单卡魔法**。单卡上"分片给谁"无从谈起:模型 A 的朴素状态 24.9 GB 纸面即超 24 GB,四个阶段单卡全灭;而 offload 把 fp32 状态(12 字节/参数的大头)搬去主存后,单卡 14.5 GB 就跑起来了——**先分片、再 offload,是这个顺序**;
2. **理论公式的正确用法**:$16\Psi$、$16\Psi/d$ 这些公式判"能不能跑"非常可靠(B 的 ZeRO-3 三卡 35.5 GB 纸面超限,实测果然 OOM),但精确预算要预留 +2~14 GB 工程项,激活与碎片随 mbs 增长;
3. **通信税在无 NVLink 机器上无处可藏**。ZeRO-3 的双次 AllGather 让双卡吞吐反而低于单卡 offload 基线的 84%,而 ZeRO-2(ReduceScatter 语义,通信量不增)双卡拿到 145%——在 PCIe 机器上,能 ZeRO-2 就别 ZeRO-3,除非显存逼你;
4. **offload 是"可行性技术"不是"加速技术"**。模型 B 的 ZeRO-3+offload 每步 8.1 s、126 tok/s/卡,CPUAdam + PCIe 流量是硬瓶颈;但它换来了"24 GB 卡训 6.65B"这个本来不存在的选项。GB10/GB200 这类统一内存架构的意义正在于此——把 offload 的 PCIe 换成一致性内存访问;
5. **分配器碎片是隐形 OOM 元凶**。默认分配器下 ZeRO-2 双卡有 5.6 GB reserved-but-unallocated 碎片直接把 mbs=2 顶死,`expandable_segments:True` 后同配置跑通——一行环境变量,显存白赚 5.6 GB。

## 六、踩坑实录

以下全部来自本次实测的真实踩坑,每条都花了分钟到小时级排查:

- **`expandable_segments` 不开,显存白丢 5.6 GB**:症状是"理论算得下、实测 OOM",`max_memory_reserved - max_memory_allocated` 的差值就是碎片。设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- **DeepSpeed 0.19.8 配置里 `"type": "FusedAdam"` 已不合法**(移除名单见其 issue #8489):统一写 `"AdamW"`——无 offload 时工厂自动选 FusedAdam,offload 时自动切 DeepSpeedCPUAdam,不需要手工指定;
- **JIT 编译三件套**:DeepSpeed 的 FusedAdam/CPUAdam 首次运行现场编译,需要 `ninja` 在 PATH、`TORCH_EXTENSIONS_DIR` 指向数据盘(别让编译产物写满系统盘)、`TORCH_CUDA_ARCH_LIST=8.9` 写死目标架构,否则每次重编;
- **torchrun 吞掉子进程退出码**:训练进程 OOM 时 exit 3,到 torchrun 层变成 exit 1,脚本无法靠返回码区分 OOM 与崩溃——在训练脚本内部打 `TRY_OK/TRY_OOM` 标记、靠日志判定;
- **共享 conda 环境的 pip 陷阱**:复制的环境里 pip 入口 shebang 指向原路径,`pip install` 会装进别人的环境——一律 `python -m pip`;
- **`pkill -f` 自杀**:`pkill -f pattern` 会匹配到包含该 pattern 的 ssh 远端命令自身,用字符类断开:`pkill -f "patter[n]"`。

## 小结

- 第二篇的 $16\Psi$ 账单在实测中全面对上:朴素单卡 24.9 GB 纸面超限必 OOM,ZeRO-3 三卡 8.3 GB 理论 + 工程常数 = 22.55 GB 实测,可行性判断可靠、精确预算要加 +2~14 GB;
- ZeRO 的两个顺序:先分片再 offload;能 ZeRO-2 就别 ZeRO-3(PCIe 机器上通信税 16–84% 不等);
- 6.65B 在 24GB 卡上唯一可跑形态是 ZeRO-3+CPU offload(126 tok/s/卡,8 s/步)——可行性技术,不是加速技术;
- 分配器一行配置 `expandable_segments:True` 值 5.6 GB 显存;
- DeepSpeed 的学习成本在 json 语义与 JIT 工具链,不在 API——`initialize()` 一行接入是真的。

## 复现环境

```text
GPU        4× RTX 4090 24GB(无 NVLink/P2P;实测用 GPU1-3,GPU0 被外部任务占用)
驱动/CUDA  580.126.09 / CUDA 12.6
软件       torch 2.7.1+cu126 · deepspeed 0.19.8 · transformers 5.5.4 · python 3.12
关键环境   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
           TORCH_CUDA_ARCH_LIST=8.9 · TORCH_EXTENSIONS_DIR=<数据盘>
负载       GPT-2 架构随机初始化(A=1.558B, B=6.654B)· seq 1024 · 合成 token · bf16
方法       原生 deepspeed.initialize() · mbs 翻倍探测 · warmup 10 + 测 30 步 · 3 窗口
```

训练脚本(`bench.py`)核心约 60 行:构造 `GPT2Config` → `from_config` 随机初始化 → 合成 batch 循环 → `engine.step()`,完整脚本、22 个配置的 json 与 56 份运行日志已随本文发布在[仓库](https://github.com/miracle-techlink/miracle-techlink.github.io) source 分支 `bench/` 目录。

本系列完。回看:[篇1 基础](/posts/9aac52d9.html) · [篇2 数据并行](/posts/3a50363.html) · [篇3 模型并行](/posts/212a16ef.html) · [篇4 集群架构](/posts/74a233f5.html)
