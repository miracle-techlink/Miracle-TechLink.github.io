---
title: LLM分布式训练2---并行策略之数据并行
date: 2025-04-23 00:00:00
updated: 2026-10-25 00:00:00
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

本篇是[系列第二篇](/posts/9aac52d9.html)。数据并行是分布式训练的最小完整闭环:数据切分、模型复制、梯度同步、参数更新。本篇先把它当数学对象严格写出来,再落到 PyTorch 的两代实现(`DataParallel` → `DistributedDataParallel`),推导 Ring-AllReduce 的通信量,最后回答一个关键问题:数据并行为什么救不了大模型——以及 ZeRO 如何在数据并行的框架内重新分配第一篇算出的 16 字节/参数账单。

## 一、数据并行的数学表述

设模型参数 $\theta \in \mathbb{R}^{\Psi}$、全局 batch 为 $B$、共 $d$ 张卡。把 batch 划分为 $d$ 份,第 $i$ 张卡持有 $\{(x_j, y_j)\}_{j \in S_i}$,$|S_i| = B/d$,模型复制到每张卡。一步同步数据并行训练为:

**前向**:各卡独立计算本地 logits

$$
z^{(i)} = f(x^{(i)}; \theta)
$$

**损失**:各卡计算本地损失的**和**的形式(注意分母是 $B$ 而不是 $B/d$,这样各卡损失相加才等于全局 batch 损失)

$$
\ell^{(i)}(\theta) = \frac{d}{B}\sum_{j \in S_i} \mathcal{L}\big(f(x_j;\theta),\, y_j\big)
$$

**反向**:各卡对本地损失求梯度

$$
g^{(i)} = \nabla_\theta \ell^{(i)}(\theta)
$$

**聚合**:全体卡做梯度均值(AllReduce)

$$
\bar{g} = \frac{1}{d}\sum_{i=0}^{d-1} g^{(i)}
$$

**更新**:各卡用相同的平均梯度更新各自的参数副本

$$
\theta \leftarrow \theta - \eta\, \bar{g}
$$

由于 $\frac{1}{d}\sum_i \nabla_\theta \ell^{(i)} = \nabla_\theta \frac{1}{B}\sum_{j=1}^{B}\mathcal{L}_j$,同步数据并行与 batch 为 $B$ 的单卡训练**数学等价**——这是它最重要的性质:并行不改变优化轨迹,只改变速度。代价是每步一次全梯度的集合通信,通信量 $2\Psi$ 字节(BF16)不随 $B$ 变化,于是 batch 越大、模型计算越重,通信越容易被计算隐藏。

![框架图](/images/insert_images/DDP/DDP2/1.png)

## 二、从 DataParallel 到 DistributedDataParallel

### nn.DataParallel:单进程多线程(已被弃用)

`DataParallel` 在一个进程里开多线程,每步 forward 时把模型广播到各卡、输入按第 0 维切分、输出在主卡拼回:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

它的三个结构性缺陷决定了它只适合原型:

1. **GIL 串行化**:单进程多线程,CPU 侧的 python 逻辑被 GIL 限制,多卡前向的调度本身不能并行;
2. **主卡带宽瓶颈**:梯度默认在 GPU-0 聚合再广播,卡数越多,GPU-0 的通信越接近 $2\times(d-1)$ 倍单卡梯度量,其他卡空等;
3. **显存倾斜**:主卡额外承担聚合缓冲,各卡实际可用显存不一致。

### nn.DistributedDataParallel:多进程,一人一卡

DDP 换成多进程架构:每个 GPU 一个进程,各进程执行相同脚本、不同 rank;梯度同步不再走"主卡中转",而是对梯度桶(bucket)做 **Ring-AllReduce**——没有任何一张卡是中心,每张卡的收发量完全对称。

DDP 的两个关键工程优化:

- **分桶重叠**:参数按逆序(反向传播先算出的先通信)切成约 25 MB 的桶,某桶内所有梯度就绪即启动该桶的 AllReduce,与后续层的反向计算重叠;
- **通信钩子**:`register_comm_hook` 允许替换默认 AllReduce 为 ReduceScatter 等原语(配合 ZeRO/FP16 通信压缩)。

## 三、Ring-AllReduce:通信量的推导

$d$ 张卡排成环,每卡初始持有完整梯度 $G$(大小 $V = 2\Psi$ 字节)。把每卡的缓冲切成 $d$ 块,算法分两阶段:

**阶段一 ReduceScatter**:共 $d-1$ 步。第 $t$ 步,每卡把自己的一块发给右邻居并累加。$d-1$ 步后,每卡上恰好有一块是**全局归约结果**,且各卡持有的块互不相同——每卡收发 $(d-1) \cdot \frac{V}{d}$。

**阶段二 AllGather**:再 $d-1$ 步,把各卡的归约块沿环扩散。结束后每卡都拥有完整的 $\bar{G}$——同样每卡收发 $(d-1)\cdot\frac{V}{d}$。

合计每卡通信量:

$$
V_{\text{ring}} \;=\; 2 \cdot \frac{d-1}{d} \cdot 2\Psi \;\xrightarrow{d\to\infty}\; 4\Psi \ \text{字节}
$$

三个要点:

1. **与卡数几乎无关**:总链路流量被环均摊,每卡收发量上界 $4\Psi$,这正是第一篇引用的 $2\Psi(d-1)/d$ 每方向公式(收发合计 $4\Psi(d-1)/d$……按"收+发"计各 $2\Psi(d-1)/d$);
2. **同步步数 $2(d-1)$**:环算法每步只传 $V/d$,大梯度下带宽占优、时延劣势,所以 NCCL 在小消息时切 Tree 算法($\log d$ 步),见第四篇;
3. **对比参数服务器**:PS 架构中 worker 与 server 间流量集中,server 侧带宽 $O(d \cdot V)$;Ring 把这个问题消掉,是"去中心化"换来的可扩展性。

## 四、实战:手写 DistributedSampler + DDP 全流程

### 为什么要理解 DistributedSampler

DDP 的正确性依赖"各卡数据互不重叠且总量覆盖全集"。`DistributedSampler` 的实现浓缩了这个契约,值得手写一遍:

```python
import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()   # 进程数 = 卡数
        if rank is None:
            rank = dist.get_rank()                 # 当前进程编号
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # 向上取整,保证每卡样本数相同
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas  # 补齐后的总大小
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)  # 各进程同种子 → 同一排列
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # 补齐到 total_size(重复采样尾部样本)
        indices += indices[:(self.total_size - len(indices))]
        # 关键一步:跨步切片,rank i 拿第 i, i+d, i+2d … 个样本
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # 每 epoch 换洗牌,且所有进程同步换,保证切分始终对齐
        self.epoch = epoch
```

两个容易踩的坑:

- **`shuffle=(seed+epoch)` 的同步性**:所有 rank 必须生成同一个排列再跨步切片。若各 rank 各自随机,样本会重叠或丢失,训练在数学上不再等价;
- **`DataLoader(shuffle=False)`**:指定 sampler 时必须关掉 DataLoader 自己的 shuffle,否则 sampler 被忽略。

### 完整训练脚本 main.py(ResNet18 + CIFAR-10)

```python
import argparse, os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision import datasets, models

# torchrun 注入的环境变量,不要手工猜
local_rank = int(os.environ["LOCAL_RANK"])

parser = argparse.ArgumentParser(description='ResNet18 on CIFAR-10 (DDP)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='每卡 batch size(全局 batch = b × 卡数)')
parser.add_argument('--epochs', default=100, type=int)
args = parser.parse_args()

# NCCL 后端:GPU 集合通信的事实标准
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)   # CIFAR-10 十分类
model = model.cuda()
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False,  # ResNet 全参训练,关掉可省每步图遍历
)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=1e-4)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
train_sampler = DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,  # sampler 在,shuffle 必须关
    num_workers=4, pin_memory=True, sampler=train_sampler,
)

for epoch in range(args.epochs):
    model.train()
    train_sampler.set_epoch(epoch)      # 每 epoch 重洗且各 rank 同步
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        outputs = model(inputs)          # [B, 10]:B 是每卡 batch,10 是类别数
        loss = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()                  # DDP 在此对梯度桶做 Ring-AllReduce
        optimizer.step()

        if i % 50 == 0 and dist.get_rank() == 0:
            print(f'epoch {epoch} step {i} loss {loss.item():.4f}')

dist.destroy_process_group()
```

### torchrun:启动语义

```bash
torchrun --nproc_per_node=2 main.py
```

`torchrun` 做三件事:拉起 2 个进程、给每个进程注入 `RANK`/`LOCAL_RANK`/`WORLD_SIZE` 环境变量、选一个 rank 0 做 rendezvous。多机时再指定 `--nnodes`/`--node_rank`/`--rdzv_endpoint`。**不要用 `mp.spawn` 手搓启动器**——容错(进程死亡自动重启整组)和 c10d rendezvous 的坑远比想象中多。

正确运行的输出(每卡各打印自己的 rank,batch 维是每卡 batch=64,输出形状 `[64, 10]`):

```plaintext
Use GPU:0 for training
epoch 0 step 0 loss 2.3721
epoch 0 step 50 loss 1.8456
Use GPU:1 for training
epoch 0 step 0 loss 2.3987
epoch 0 step 50 loss 1.9102
```

数据集下载慢可手动处理:`wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` 后解压到 `./data/cifar-10-batches-py` 即可。

![框架图](/images/insert_images/DDP/DDP2/2.png)

## 五、数据并行的边界与 ZeRO

DDP 每卡复制**全部**训练状态。由第一篇,常驻状态为 $16\Psi$ 字节/卡,与卡数无关——**加卡不减显存**。8B 模型需要每卡 122 GiB 常驻状态,任何 24–80 GB 的卡都放不下:这就是数据并行的显存边界,与算力无关。

ZeRO(ZeRO-DP, Rajoarski et al. 2020)的洞察:DDP 里那 16 字节/参数的状态,大部分**不必每卡全量持有**。按状态逐级分片:

| 阶段 | 分片对象 | 每卡常驻状态 | 额外通信(vs DDP) |
| --- | --- | --- | --- |
| DDP | 无 | $16\Psi$ | 基准:每步 $2\Psi(d-1)/d \cdot 2$ |
| ZeRO-1 | 优化器状态(12B/参数) | $4\Psi + 12\Psi/d$ | 无(归约后即分片,ReduceScatter 代替 AllReduce) |
| ZeRO-2 | + 梯度(2B/参数) | $2\Psi + 14\Psi/d$ | 无(梯度分片天然 ReduceScatter) |
| ZeRO-3 | + 权重(2B/参数) | $16\Psi/d$ | 前向/反向各一次 AllGather:$+4\Psi(d-1)/d$/步 |

以 8B 模型、8 卡为例,每卡常驻状态从 122.1 GiB 降到 42.0 → 28.6 → 15.3 GiB(ZeRO-1/2/3)。代价结构值得记:

- **ZeRO-1/2 几乎不增加通信量**:梯度的 AllReduce 本来就由 ReduceScatter + AllGather 组成,只保留 ReduceScatter、砍掉 AllGather,省的是"每卡都要完整梯度"这一步;
- **ZeRO-3 通信量约 1.5 倍**:多了前向/反向各一次权重 AllGather(每卡 $2\times 2\Psi(d-1)/d$/次),需要用**预取 + 通信计算重叠**把这部分藏进计算里,否则吞吐损失明显;
- ZeRO-3 的执行方式(参数用时收集、用完即弃、梯度按片归约)与 PyTorch 官方 FSDP 完全同构,FSDP 可视为 ZeRO-3 的框架化实现。

激活显存不在 $16\Psi$ 账单内,ZeRO 也不处理它——那是**激活重计算**(用约 33% 额外计算换激活显存降一个量级)与第三篇**序列并行/流水线**的职责。

## 小结

- 同步数据并行 $=$ 大 batch 单卡训练,数学严格等价,代价是每步 $2\Psi$ 字节级梯度集合通信;
- `DataParallel` 因 GIL/主卡瓶颈被弃用;DDP 多进程 + 梯度桶 + Ring-AllReduce 是标准形态,`DistributedSampler` 的跨步切片契约是正确性的根基;
- Ring-AllReduce 每卡收发 $2\Psi(d-1)/d$/方向,总流量在环上均摊,无中心瓶颈;
- DDP 每卡 $16\Psi$ 状态不随卡数下降;ZeRO-1/2/3 把账单降到 $4\Psi{+}12\Psi/d$、$2\Psi{+}14\Psi/d$、$16\Psi/d$,通信代价分别为 $\times 1$、$\times 1$、$\times 1.5$——各阶段的真实显存与吞吐数字,见[第五篇](/posts/eac6a656.html)在 4×RTX 4090 上的实测。

下一篇:[LLM分布式训练3——模型并行:张量、流水线与 3D 并行](/posts/212a16ef.html)
