---
title: LLM分布式训练2---并行策略之数据并行
date: 2025-04-23 00:00:00
updated: 2026-10-26 00:00:00
tags:
  - LLM
  - 分布式训练
  - DDP
  - ZeRO
abbrlink: 3a50363
---

> 作者：@同济大学 刘越
> Github ID：@miracle-techlink
> 联系邮箱：[miracle.techlink@gmail.com](mailto:miracle.techlink@gmail.com)
> 校内邮箱: [2254018@tongji.edu.cn](mailto:2254018@tongji.edu.cn)

**本篇在系列中的位置**。[第一篇](/posts/9aac52d9.html)算清了训练的账——算力、显存、通信三堵墙。本篇给出第一个完整的解决方案:数据并行。我们会从最严格的数学定义出发,证明数据并行为什么是对的(数学等价性),推导 Ring-AllReduce 的通信量,然后落到一个完整的、能跑的 PyTorch 实现。最后回答一个关键问题:数据并行为什么救不了大模型,以及 ZeRO 如何在数据并行的框架内把第一篇的 16 字节账单重新分配。

## 一、数据并行的数学表述:严格版

### 1.1 问题定义

设:
- 模型参数 $\theta \in \mathbb{R}^{\Psi}$($\Psi$ 是参数量)
- 训练数据集 $\mathcal{D} = \{(x_j, y_j)\}_{j=1}^{N}$,共 $N$ 个样本
- 单卡 batch 大小 $b$(每卡每次处理 $b$ 个样本)
- 卡数 $d$

目标:用 $d$ 张卡训练模型,使得结果与"单卡 batch=$b \times d$"完全一致。

### 1.2 一步训练的完整数学过程

**第 1 步:数据划分**

把 $\mathcal{D}$ 划分成 $d$ 份,第 $i$ 张卡持有子集 $\mathcal{D}_i = \{(x_j, y_j)\}_{j \in S_i}$,$|S_i| = N/d$。划分必须满足:$\bigcup_i \mathcal{D}_i = \mathcal{D}$ 且 $\mathcal{D}_i \cap \mathcal{D}_k = \emptyset$(各卡数据不重叠)。

**第 2 步:前向传播(各卡独立)**

第 $i$ 张卡对本地 batch $X^{(i)} = \{x_j\}_{j \in S_i}$ 计算:

$$
z^{(i)} = f(X^{(i)}; \theta)
$$

其中 $f$ 是模型的前向函数,输出 $z^{(i)} \in \mathbb{R}^{b \times \text{vocab}}$(对 LLM 是下一个 token 的概率分布)。

**第 3 步:损失计算(注意归一化)**

每张卡计算本地损失,但**分母要用全局 batch 大小 $b \times d$**,不是本地 $b$:

$$
\ell^{(i)}(\theta) = \frac{1}{b \cdot d} \sum_{j \in S_i} \mathcal{L}(z_j^{(i)}, y_j)
$$

为什么要这样?因为这样:

$$
\sum_{i=0}^{d-1} \ell^{(i)}(\theta) = \frac{1}{b \cdot d} \sum_{j=1}^{N} \mathcal{L}(z_j, y_j) = \ell_{\text{global}}(\theta)
$$

各卡损失相加 = 全局 batch 损失。这是数学等价性的关键。

**第 4 步:反向传播(各卡独立)**

每张卡对本地损失求梯度:

$$
g^{(i)} = \nabla_\theta \ell^{(i)}(\theta)
$$

**第 5 步:梯度聚合(AllReduce)**

把所有卡的梯度求和(不是平均,因为损失已经归一化了):

$$
g_{\text{global}} = \sum_{i=0}^{d-1} g^{(i)}
$$

这个操作叫 **AllReduce**:每张卡输入 $g^{(i)}$,输出 $g_{\text{global}}$,所有卡得到相同结果。

**第 6 步:参数更新**

每张卡用全局梯度更新本地参数副本:

$$
\theta \leftarrow \theta - \eta \cdot g_{\text{global}}
$$

其中 $\eta$ 是学习率。因为所有卡用的是相同的 $g_{\text{global}}$,所以更新后的 $\theta$ 也完全一致。

### 1.3 数学等价性证明

**定理**:同步数据并行与单卡 batch=$b \times d$ 的训练在数学上等价。

**证明**:

单卡 batch=$b \times d$ 的损失:

$$
\ell_{\text{single}}(\theta) = \frac{1}{b \cdot d} \sum_{j=1}^{b \cdot d} \mathcal{L}(f(x_j; \theta), y_j)
$$

梯度:

$$
\nabla_\theta \ell_{\text{single}}(\theta) = \frac{1}{b \cdot d} \sum_{j=1}^{b \cdot d} \nabla_\theta \mathcal{L}(f(x_j; \theta), y_j)
$$

数据并行的全局梯度(由第 5 步):

$$
g_{\text{global}} = \sum_{i=0}^{d-1} \nabla_\theta \ell^{(i)}(\theta) = \sum_{i=0}^{d-1} \nabla_\theta \left(\frac{1}{b \cdot d} \sum_{j \in S_i} \mathcal{L}(z_j^{(i)}, y_j)\right)
$$

交换求和顺序:

$$
g_{\text{global}} = \frac{1}{b \cdot d} \sum_{i=0}^{d-1} \sum_{j \in S_i} \nabla_\theta \mathcal{L}(z_j^{(i)}, y_j) = \frac{1}{b \cdot d} \sum_{j=1}^{b \cdot d} \nabla_\theta \mathcal{L}(z_j, y_j)
$$

与单卡梯度完全一致。$\blacksquare$

**个人理解**:数学等价性是数据并行最大的优点——它不改变优化轨迹,只改变速度。这意味着你不需要担心"并行会不会影响收敛",只需要担心"并行能多快"。但代价是每步都要做 AllReduce,通信量 $2\Psi$ 字节(BF16)不随 batch 变化,这就是第一篇说的通信墙的来源。

## 二、从 DataParallel 到 DistributedDataParallel

### 2.1 nn.DataParallel:单进程多线程的失败设计

PyTorch 最早的多卡接口是 `nn.DataParallel`,用法极简:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

它在一个进程里开多线程,每步 forward 时:
1. 把模型广播到各卡;
2. 把输入按第 0 维切分;
3. 各卡并行前向;
4. 输出在主卡(GPU-0)拼回。

**为什么它被淘汰了?** 三个结构性缺陷:

**缺陷 1:GIL 锁**

Python 的 GIL(Global Interpreter Lock)保证同一时刻只有一个线程执行 Python 字节码。虽然矩阵乘是 CUDA 操作(不持 GIL),但调度的 Python 逻辑(切数据、启动 kernel)被 GIL 串行化了。8 张卡的 forward 调度时间 = 单卡的 8 倍,多卡的 CPU 开销并没有摊薄。

**缺陷 2:主卡瓶颈**

梯度默认在 GPU-0 聚合,再广播给其他卡。$d$ 卡时,GPU-0 要收发 $(d-1) \times 2\Psi$ 字节,其他卡只需 $2\Psi$——卡数越多,主卡通信量线性增长,其他卡在等它。

**缺陷 3:显存倾斜**

主卡要额外分配聚合缓冲(完整梯度),8B 模型 BF16 梯度 16.4GB,主卡要多 16.4GB 显存。其他卡 122GB 装得下,主卡 138GB 就 OOM 了。

实测数据(我自己的实验,ResNet50 + ImageNet,8 卡):DataParallel 比单卡快 3.2 倍,DistributedDataParallel 快 6.8 倍——差了 2 倍多,全在这三个缺陷上。

### 2.2 nn.DistributedDataParallel:多进程的正确姿势

DDP 换成多进程架构:

- **每个 GPU 一个进程**:进程 $i$ 绑定 GPU $i$,各进程执行相同脚本、不同 rank;
- **梯度桶(bucket)**:不等全部梯度算完,每算出一个桶(约 25MB)立即开始该桶的 AllReduce;
- **Ring-AllReduce**:没有主卡,每张卡的收发量完全对称。

DDP 的关键代码:

```python
import torch.distributed as dist
import torch.nn as nn

# 初始化进程组(NCCL 后端,GPU 集合通信的事实标准)
dist.init_process_group(backend="nccl")

# 绑定 GPU(由 torchrun 注入环境变量)
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 模型复制到 GPU
model = MyModel().cuda()

# 包装成 DDP
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False  # 全参训练关掉,省每步图遍历
)
```

**两个关键优化**:

1. **梯度分桶 + 通信重叠**:DDP 把参数按逆序(反向传播先算出的先通信)切成桶,某桶内所有梯度就绪即启动该桶的 AllReduce。这样前 $i$ 层的梯度通信与后 $i$ 层的反向计算并行,实测能隐藏 80% 的通信时间。

2. **通信钩子**:`model.register_comm_hook()` 允许替换默认 AllReduce 为其他原语(ReduceScatter、FP16 压缩通信等)。ZeRO/FSDP 就是用这个钩子插入自己的逻辑。

## 三、Ring-AllReduce:通信量的完整推导

### 3.1 算法描述

$d$ 张卡排成环(物理上可以是任何拓扑,逻辑上按编号 0→1→...→$d$-1→0)。把每卡的梯度缓冲切成 $d$ 块,每块大小 $V/d$($V = 2\Psi$ 字节,BF16 梯度)。

**阶段 1:ReduceScatter($d-1$ 步)**

第 $t$ 步($t = 0, 1, ..., d-2$):
- 卡 $i$ 把自己的第 $(i-t) \mod d$ 块发给右邻居(卡 $(i+1) \mod d$);
- 卡 $i$ 接收左邻居的第 $(i-t-1) \mod d$ 块,累加到自己的对应块。

$d-1$ 步后,每卡上恰好有一块是**全局归约结果**,且各卡持有的块互不相同。

**阶段 2:AllGather($d-1$ 步)**

第 $t$ 步($t = 0, 1, ..., d-2$):
- 卡 $i$ 把已归约的第 $(i-t+1) \mod d$ 块发给右邻居;
- 卡 $i$ 接收左邻居的第 $(i-t) \mod d$ 块,覆盖自己的对应块。

$d-1$ 步后,每卡都拥有完整的 $g_{\text{global}}$。

### 3.2 通信量计算

**阶段 1**:每步每卡发送 $V/d$ 字节、接收 $V/d$ 字节,共 $d-1$ 步:

$$
V_{\text{阶段1}} = (d-1) \cdot \frac{V}{d} = V \cdot \frac{d-1}{d}
$$

**阶段 2**:同理,也是 $V(d-1)/d$。

**合计**(收+发):

$$
V_{\text{ring}} = 2V \cdot \frac{d-1}{d} = 2 \cdot 2\Psi \cdot \frac{d-1}{d} = 4\Psi \cdot \frac{d-1}{d}
$$

### 3.3 三个关键性质

**性质 1:与卡数几乎无关**

$d \to \infty$ 时,$V_{\text{ring}} \to 4\Psi$。总链路流量 $d \times 4\Psi = 4\Psi d$ 被均摊到 $d$ 张卡,每卡恒定 $4\Psi$。这是 Ring 最优美的性质:**卡数增长不增加单卡负担**。

对比参数服务器(PS):worker 与 server 间流量集中,server 侧带宽 $O(d \cdot V)$,千亿模型下没有任何单点扛得住。

**性质 2:同步步数 $2(d-1)$**

Ring 每步只传 $V/d$,大消息下带宽占优,但小消息被启动时延支配。8B 模型梯度 16.4GB,8 卡时每步传 2GB,PCIe 32GB/s 下 62ms 一步,完全可接受;但如果是 1MB 的小梯度,启动时延(10-100μs)× 14 步 = 140-1400μs,占比就很高了。

所以 NCCL 对小消息用 Tree 算法($2\log_2 d$ 步),大消息用 Ring。

**性质 3:链路均摊**

环形通信把跨机流量分到每张卡的网卡上。8 卡机配 8×200Gbit/s 网卡,每张卡的 Ring 流量都走自己的网卡,单网卡 25GB/s 不构成瓶颈——前提是流量对称(Ring 天然满足)。

## 四、实战:手写 DistributedSampler + DDP 全流程

### 4.1 为什么要理解 DistributedSampler

DDP 的正确性依赖一个隐含契约:**各卡数据互不重叠且总量覆盖全集**。`DistributedSampler` 的实现浓缩了这个契约,值得手写一遍,因为它直接影响训练的数学正确性。

### 4.2 完整实现(带详细注释)

```python
import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedSampler(Sampler):
    """
    分布式采样器:确保各卡数据不重叠且覆盖全集。
    
    核心思想:
    1. 生成一个全局排列(所有卡用同一个种子,保证排列一致)
    2. 补齐到 total_size(样本数 × 卡数),保证每卡样本数相同
    3. 跨步切片:rank i 拿第 i, i+d, i+2d, ... 个样本
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        # num_replicas: 总卡数(进程数)
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        # rank: 当前进程编号(0 到 num_replicas-1)
        if rank is None:
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0  # 当前训练轮次,用于变化 shuffle
        
        # 关键:向上取整,保证每卡样本数相同
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas  # 补齐后的总大小
        
        self.shuffle = shuffle
        self.seed = seed
    
    def __iter__(self):
        # 第 1 步:生成全局排列
        if self.shuffle:
            g = torch.Generator()
            # 关键:所有卡用同一个种子,生成相同的排列
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # 第 2 步:补齐到 total_size(重复采样尾部样本)
        # 例:1000 样本、3 卡 → num_samples=334,total_size=1002,补 2 个
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # 第 3 步:跨步切片(核心!)
        # rank 0: indices[0], indices[3], indices[6], ...
        # rank 1: indices[1], indices[4], indices[7], ...
        # rank 2: indices[2], indices[5], indices[8], ...
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """
        每个 epoch 必须调用!
        否则所有 epoch 用同一个排列,模型会过拟合到固定顺序。
        """
        self.epoch = epoch
```

### 4.3 两个容易踩的坑

**坑 1:`shuffle=(seed+epoch)` 的同步性**

所有 rank 必须生成**同一个排列**再跨步切片。如果各 rank 各自随机,样本会重叠或丢失。

错误示范:

```python
# ❌ 错误:各 rank 用自己的种子
g.manual_seed(self.seed + self.epoch + self.rank)  # 每卡排列不同!
indices = torch.randperm(len(self.dataset), generator=g).tolist()
```

这会导致:rank 0 和 rank 1 可能都拿到样本 5,而样本 10 没有任何卡拿到。训练在数学上不再等价于单卡。

**坑 2:`DataLoader(shuffle=True)`**

指定 sampler 时,必须关掉 DataLoader 自己的 shuffle:

```python
# ✅ 正确
train_loader = DataLoader(
    dataset,
    sampler=DistributedSampler(dataset),
    shuffle=False  # 必须关!
)

# ❌ 错误
train_loader = DataLoader(
    dataset,
    sampler=DistributedSampler(dataset),
    shuffle=True  # sampler 被忽略,各卡数据重叠!
)
```

### 4.4 完整训练脚本(ResNet18 + CIFAR-10)

```python
import argparse, os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision import datasets, models

def main():
    # torchrun 注入的环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='每卡 batch(全局 = b × 卡数)')
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args()
    
    # 初始化进程组
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    # 模型
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 十分类
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False
    )
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=1e-4)
    
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # sampler 在,必须关
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # 每 epoch 重洗
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            outputs = model(inputs)  # [b, 10]: b 是每卡 batch
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # DDP 在此对梯度桶做 Ring-AllReduce
            optimizer.step()
            
            if i % 50 == 0 and dist.get_rank() == 0:
                print(f'epoch {epoch} step {i} loss {loss.item():.4f}')
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### 4.5 torchrun:启动语义

```bash
torchrun --nproc_per_node=4 main.py
```

`torchrun` 做三件事:

1. **拉起 4 个进程**:每个进程绑定一张卡;
2. **注入环境变量**:`RANK`(全局进程号)、`LOCAL_RANK`(本机进程号)、`WORLD_SIZE`(总进程数);
3. **rendezvous**:选一个 rank 0 做初始化同步点,其他进程等它就绪。

多机时:

```bash
torchrun --nnodes=2 --nproc_per_node=4 \
         --rdzv_id=123 --rdzv_backend=c10d \
         --rdzv_endpoint=worker0:29500 \
         main.py
```

**不要用 `mp.spawn` 手搓启动器**:容错(进程死亡自动重启整组)和 c10d rendezvous 的坑远比想象中多,torchrun 都处理好了。

## 五、数据并行的边界:ZeRO 的诞生

### 5.1 数据并行的显存死局

DDP 每卡复制**全部**训练状态。由第一篇,常驻状态 $16\Psi$ 字节/卡,与卡数无关——**加卡不减显存**。

8B 模型:$16 \times 8.2\text{B} = 131$GB/卡。任何 24-80GB 的卡都装不下。这就是数据并行的显存边界,与算力无关。

### 5.2 ZeRO 的核心洞察

ZeRO(ZeRO-DP, Rajbhandari et al. 2020)的洞察:DDP 里那 16 字节/参数的状态,大部分**不必每卡全量持有**。

**为什么?** 因为:
- 第 $i$ 张卡在前向/反向时,只用得到模型的一部分参数(当前层);
- 优化器更新时,只需要自己负责的那部分参数的梯度;
- 主权重和 Adam 状态只在 optimizer.step() 时用一次。

所以可以把这 16 字节按"谁需要谁持有"的原则分片。

### 5.3 ZeRO 三阶段:逐级分片

**ZeRO-1:分片优化器状态(12 字节/参数)**

- 每卡持有全部 bf16 权重 + bf16 梯度(4 字节/参数);
- 优化器状态(FP32 主权重 + $m$ + $v$, 12 字节/参数)按卡分片,每卡只持 $12\Psi/d$;
- optimizer.step() 时,每卡只更新自己负责的那部分参数,然后 AllGather 完整权重。

每卡显存:$4\Psi + 12\Psi/d$。

**通信量**:与 DDP 相同(梯度的 AllReduce 天然由 ReduceScatter + AllGather 组成,ZeRO-1 只保留 ReduceScatter)。

**ZeRO-2:分片梯度(2 字节/参数)**

- 在 ZeRO-1 基础上,梯度也分片:每卡只保留自己负责的那部分参数的梯度;
- 反向传播时,每卡算出完整梯度,ReduceScatter 后只留自己那份。

每卡显存:$2\Psi + 14\Psi/d$。

**通信量**:与 ZeRO-1 相同。

**ZeRO-3:分片权重(2 字节/参数)**

- 权重也分片:每卡只持 $2\Psi/d$;
- 前向/反向执行第 $i$ 层时,AllGather 该层的完整参数,用完即弃。

每卡显存:$16\Psi/d$。

**通信量**:多了前向/反向各一次权重 AllGather(每卡 $2 \times 2\Psi(d-1)/d$/次),约 1.5 倍于 ZeRO-2。

### 5.4 显存节省的实测对照

8B 模型、8 卡:

| 阶段 | 每卡常驻状态(GB) | 通信倍数 |
| --- | --- | --- |
| DDP | 122.1 | 1× |
| ZeRO-1 | 42.0 | 1× |
| ZeRO-2 | 28.6 | 1× |
| ZeRO-3 | 15.3 | 1.5× |

(数字来自第一篇的 $16\Psi$ 公式,实测见第五篇)

### 5.5 ZeRO 的代价结构

**ZeRO-1/2 几乎不增加通信量**:

梯度的 AllReduce 本来就由 ReduceScatter + AllGather 组成。ZeRO-1/2 只保留 ReduceScatter(每卡收 $2\Psi(d-1)/d$),砍掉 AllGather,省的是"每卡都要完整梯度"这一步。

**ZeRO-3 通信量约 1.5 倍**:

多了前向/反向各一次权重 AllGather。以 8B 模型、8 卡为例:

- 前向 AllGather:$2 \times 8.2\text{B} \times 7/8 = 14.3$GB/卡
- 反向 AllGather:同上
- 梯度 ReduceScatter:$14.3$GB/卡(与 ZeRO-2 相同)

合计 $42.9$GB/卡,是 ZeRO-2($14.3$GB)的 3 倍——但 ZeRO-2 还有 AllGather(我没算,因为可以藏在反向里),实际约 1.5 倍。

**个人理解**:ZeRO-3 的通信税在无 NVLink 的机器(4090)上无处可藏。第五篇实测会显示,ZeRO-3 双卡吞吐反而比单卡 offload 基线低 16%,而 ZeRO-2 双卡有 145% 的相对收益——在 PCIe 机器上,能 ZeRO-2 就别 ZeRO-3。

## 六、激活显存:ZeRO 管不了的部分

### 6.1 激活显存的公式

$$
M_{\text{act}} \approx B \times L \times d \times 34 \times \text{层数}\ \text{字节}
$$

8B 模型、batch=4、seq=4096:$4 \times 4096 \times 4096 \times 34 \times 32 = 7.4$GB。

### 6.2 为什么 ZeRO 不管激活

ZeRO 分片的是"参数状态"(权重、梯度、优化器),这些是与参数绑定的、可以按参数索引切分的数据。

激活不同:它是"输入数据的函数",与参数无直接对应关系。第 $i$ 张卡的激活只对应它处理的那部分数据,不能按参数分片。

### 6.3 激活的解决方案

- **激活重计算**:不保存中间激活,反向时重新前向算一遍。约 33% 额外计算换激活显存降一个量级;
- **序列并行**(第三篇):把激活沿序列维切分;
- **流水线并行**(第三篇):每卡只保存自己那几层的激活。

这些是模型并行的职责,不在 ZeRO 的射程内。

## 小结

- **数学等价性**:数据并行与单卡 batch=$b \times d$ 完全一致,关键是损失归一化用全局 batch;
- **DP vs DDP**:DP 因 GIL/主卡瓶颈/显存倾斜被淘汰;DDP 多进程 + 梯度桶 + Ring-AllReduce 是标准形态;
- **Ring-AllReduce**:每卡收发 $2\Psi(d-1)/d$/方向,总流量均摊,无中心瓶颈;
- **DistributedSampler**:跨步切片契约是正确性的根基,`shuffle=(seed+epoch)` 必须全卡同步;
- **ZeRO**:分片 16 字节账单,ZeRO-1/2/3 分别降到 $4\Psi+12\Psi/d$、$2\Psi+14\Psi/d$、$16\Psi/d$,通信代价 1×/1×/1.5×;
- **激活显存**:ZeRO 管不了,靠重计算/序列并行/流水线。

下一篇:[LLM分布式训练3——模型并行:张量、流水线与 3D 并行](/posts/212a16ef.html)
