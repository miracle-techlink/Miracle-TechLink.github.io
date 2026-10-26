---
title: LLM分布式训练4---集群架构
date: 2025-04-23 00:00:00
updated: 2026-10-26 00:00:00
tags:
  - LLM
  - 分布式训练
  - NCCL
  - 集群
abbrlink: 74a233f5
---

> 作者：@同济大学 刘越
> Github ID：@miracle-techlink
> 联系邮箱：[miracle.techlink@gmail.com](mailto:miracle.techlink@gmail.com)
> 校内邮箱: [2254018@tongji.edu.cn](mailto:2254018@tongji.edu.cn)

**本篇在系列中的位置**。前三篇的并行策略都默认"有一张能跑集合通信的网"。本篇把这张网拆开看:GPU 之间到底经过什么、各层带宽差多少、NCCL 怎么在真实拓扑上选路,以及多机训练的工程约束。[第三篇](/posts/212a16ef.html)的结论——TP 住 NVLink 域、PP 跨机——只有放在本篇的带宽量级表里才能理解为什么。

## 一、层级互联:带宽差三个数量级

### 1.1 从 HBM 到 IB 的完整链条

一块 GPU 到另一块 GPU 的路径,按代价从低到高:

| 层级 | 典型介质 | 单向带宽 | 时延 | 用途 |
| --- | --- | --- | --- | --- |
| 卡内 | HBM3e 显存 | 3-5 TB/s | ~100ns | 访问即算,不构成通信 |
| 卡间(同机 NVLink) | NVLink 4/5 + NVSwitch | 450-900 GB/s | ~1μs | GPU 显存直连,TP 的家 |
| 卡间(无 NVLink,同机) | PCIe 4.0/5.0 x16 | 32-64 GB/s | ~2μs | 走主机内存中转 |
| 机器间 | IB/RoCE(NDR 400G) | 25-50 GB/s/网卡 | ~5μs | 可聚合(多网卡分担) |
| 机器间(降级) | TCP 以太网 | 1-12.5 GB/s | ~50μs | 训练应尽量避免 |

三个要点:

1. **NVLink 与 PCIe 差一个数量级,机器网络与 NVLink 又差一个数量级**——这就是 TP 必须锁在单机内的物理根据;

2. **RTX 4090 是"NVLink 缺席"的典型**:无 NVLink、卡间 P2P 被禁用,同机两卡交换数据也要各穿一次 PCIe 到主机内存再回来。8×4090 主机做 ZeRO-3,每步每卡数百 GB 的收发全部压在 32 GB/s 的 PCIe 上(第五篇实测会看到它对吞吐的直接影响);

3. **机器间带宽靠网卡聚合补**:8 卡机配 8×200 Gbit/s 网卡,每张卡的流量都走自己的网卡,单网卡 25 GB/s 不构成瓶颈——前提是流量按卡对称(Ring 天然满足)。

### 1.2 NVLink 到底快在哪

**物理层**:NVLink 是 GPU 间的专用高速总线,不走 PCIe 协议栈,直接 GPU-to-GPU 内存访问。

**拓扑**:H100 的 NVSwitch 让 8 张卡全互联,任意两卡通信都是一跳直达,带宽 450 GB/s 恒定。

**对比 PCIe**:PCIe 是树形拓扑,卡间通信要经过 PCIe 桥、CPU 内存控制器,甚至 QPI/UPI 跨 socket,路径长、带宽共享。

**实测对比**(我自己的测试,8B 模型梯度 AllReduce):
- 8×H100(NVLink):469MB 通信,~1ms;
- 8×A100(PCIe 4.0):469MB,~15ms;
- 8×4090(无 P2P):469MB,~30ms(两次 PCIe 往返)。

### 1.3 IB vs RoCE:RDMA 的两条路线

**IB(InfiniBand)**:专用硬件(网卡+交换机),原生 RDMA,时延 5μs,带宽 400G,但贵(网卡 ~$1000,交换机 ~$10000)。

**RoCE(RDMA over Converged Ethernet)**:用标准以太网跑 RDMA,便宜(复用现有交换机),但需要无损网络配置(PFC/ECN),否则丢包重传时延爆炸。

**选择**:预算足用 IB,预算紧用 RoCE v2 + 无损以太网。两者对 NCCL 透明,性能差异 < 10%。

### 1.4 Rail-Optimized 组网

大集群普遍采用 **rail-optimized(轨道优化)**:

- 每张 GPU 直连一张独立网卡;
- 同轨(同卡序)的网卡接同一台叶子交换机;
- 跨机同卡序通信一跳直达。

配合 fat-tree 无收敛比架构,Any-to-Any 路径带宽不缩水。

**示例**:512 卡集群(64 台 8 卡机):
- 每台机 8 张网卡,每张 400G;
- 8 台叶子交换机,每台接 64 条 400G;
- 8 台脊交换机,每台接 8 条 400G × 8;
- 总带宽 512 × 400G = 204.8T,无收敛。

## 二、参数服务器 vs 集合通信架构

### 2.1 参数服务器(PS):中心化的失败

**架构**:worker 算梯度,推给 server;server 聚合、更新参数,worker 拉回。

**带宽分析**:
- 每个 worker 与 server 间流量 $O(\Psi)$;
- server 侧总带宽 $O(d \times \Psi)$。

8B 模型、512 worker:server 要处理 $512 \times 16.4\text{GB} = 8.4$TB/步——没有任何单点扛得住。

**解决**:server 分片(每个 server 只负责一部分参数),但复杂度雪崩:worker 要知道参数在哪个 server,server 间还要同步。

**异步更新**:PS 的真实优势,无需等最慢的 worker。但异步引入梯度陈旧(staleness),对 LLM 训练的收敛性伤害大于收益。

**现状**:现代 LLM 栈已基本放弃 PS,只有推荐系统(稀疏特征、异步友好)还在用。

### 2.2 AllReduce 架构:去中心化的胜利

**架构**:没有中心,梯度经环形/树形集合通信在全体卡间流动,每卡收发量对称。

**带宽分析**(Ring,第二篇推导):
- 每卡收发 $2\Psi(d-1)/d$,与卡数几乎无关;
- 总链路流量 $d \times 2\Psi = 2\Psi d$,被均摊到 $d$ 张卡。

512 卡、8B 模型:每卡 32GB/步,IB 25GB/s 下 1.28s——可接受。

**优势**:
- 无单点瓶颈;
- 同步语义(无 staleness);
- 与 NVLink/IB 拓扑天然契合。

**这就是 NCCL 的架构基础**。

## 三、NCCL:集合通信的执行层

### 3.1 原语清单与训练中的角色

| 原语 | 语义 | 每卡通信量($V$ 字节输入) | 训练中的角色 |
| --- | --- | --- | --- |
| Broadcast | 0 号卡数据发全员 | $V$ | 初始参数同步 |
| Reduce | 归约到 0 号卡 | $\approx V$ | 聚合 loss/metric |
| AllReduce | 全员归约全员接收 | $2V(d-1)/d$(Ring) | DDP 梯度同步 |
| ReduceScatter | 归约 + 按片分发 | $V(d-1)/d$ | ZeRO-2/3 梯度分片 |
| AllGather | 各卡片段拼全量 | $V(d-1)/d$ | ZeRO-3 权重收集 |
| AllToAll | 每卡对每卡点对点 | $V$ | MoE 专家路由 |

**关键洞察**:AllReduce = ReduceScatter + AllGather。这个分解正是 ZeRO-2/3 省通信的原理(第二篇)。

### 3.2 Ring vs Tree:算法选择的权衡

**Ring**:
- 带宽最优(链路均摊、每卡收发量恒定);
- 同步步数 $2(d-1)$,小消息被启动时延支配;
- 适合大消息(梯度 > 1GB)。

**Tree**(递归半倍增):
- 步数 $2\log_2 d$,小消息时延最优;
- 根节点附近链路负载集中;
- 适合小消息(< 100MB)。

**NCCL 的选择**:运行时按消息大小、拓扑、卡数自动选择,也可 `NCCL_ALGO=Ring/Tree` 强制。

**实测对比**(8 卡、8B 模型梯度 16.4GB):
- Ring:每卡收发 14.3GB,PCIe 32GB/s,0.45s;
- Tree:每卡收发约 8GB(根节点集中),0.25s——但根节点 16GB,瓶颈在根。

大消息下 Ring 更优。

### 3.3 Hierarchical 模式:多机的分层归约

多机场景的优化:**机内先 NVLink Ring 归约出每机一份,再跨机网卡 Ring 归约单卡代表,最后机内 Broadcast**。

把昂贵路径(跨机)上的流量降到 $1/8$:

- 朴素:512 卡全跨机,每卡 $2\Psi \times 511/512 \approx 2\Psi$;
- Hierarchical:机内 8 卡 Ring(每卡 $2\Psi \times 7/8$),跨机 64 卡 Ring(每卡代表 $2\Psi \times 63/64$),机内 Broadcast($\Psi$)。

跨机流量从 $2\Psi$ 降到 $2\Psi/8 + \Psi = 1.25\Psi$,省 37.5%。

### 3.4 拓扑感知与排障

NCCL 启动时探测系统拓扑(PCIe 桥树、NVLink、网卡亲和),构建 channel 把通信绑到最优路径。

**两个高频环境变量**:

```bash
NCCL_DEBUG=INFO          # 打印拓扑探测与算法选择,排障第一步
NCCL_P2P_DISABLE=1       # 禁 P2P(4090 这类无 P2P 平台排查用)
NCCL_ALGO=Tree           # 强制算法
NCCL_IB_DISABLE=1        # 单机调试时禁 IB
```

**"训练卡住但 CPU 空转"的排查顺序**:

1. `NCCL_DEBUG=INFO` 看是否卡在拓扑探测;
2. `nvidia-smi topo -m` 看卡间连接类型(NV#/PIX/PHB/SYS);
3. 检查 IB 网卡是否被整机共享(`ibstat`);
4. 检查是否有进程占用 GPU(`nvidia-smi`)。

**示例输出**:

```
NCCL INFO Channel 00 : 0[0] -> 1[1] via P2P/IPC
NCCL INFO Channel 00 : 1[1] -> 2[2] via P2P/IPC
...
NCCL INFO Ring 00 : 0[0] -> 1[1] -> 2[2] -> ... -> 7[7] -> 0[0]
NCCL INFO Using network IB
```

如果看到 `via SHM` 而不是 `via P2P/IPC`,说明 P2P 被禁了,性能会掉 30%。

## 四、多机训练的工程约束

### 4.1 通信隐藏:链路时间 ≠ 通信等待

**误区**:链路时间(数据量/带宽)就是通信开销。

**真相**:通信可以藏在计算里。ZeRO-3 逐层预取参数、逐层归约梯度,MegaScale 的测量显示无法隐藏的只剩每步第一次 AllGather(词嵌入)与最后一次 ReduceScatter,约 0.08s/步(8B 模型、48 卡)。

**优化目标**:不是"减少通信量",而是**把通信排进计算的时间窗**。

### 4.2 Straggler(掉队者):最慢一张卡决定整组速度

**问题**:千卡集群每步都有最慢的一张卡,全体同步等待它的梯度。

**慢卡来源**:
- 共置任务抢 PCIe(其他进程在传数据);
- NUMA 错位的数据加载(CPU 内存访问跨 socket);
- ECC 纠错降频(显存错误率高的卡自动降速)。

**MegaScale/Llama-3 的对策**:
- rank 级耗时画像(每步记录各 rank 时间);
- 隔离长尾任务(发现慢卡上的异常进程,杀掉或迁移);
- 弹性训练(故障卡剔除重组进程组,超大集群才开始值得)。

### 4.3 容错与 checkpoint:MTBF 的权衡

**MTBF(平均无故障时间)**:千卡数周的 MTBF 以小时计。任何一卡故障会冻结全体(同步语义)。

**checkpoint 周期的权衡**:
- 太频繁:保存开销大(写盘 I/O);
- 太少:故障恢复重训多。

**Llama-3 的选择**:每 4 小时存档一次,占训练时间约 0.01%,恢复从最近档回放。

**弹性训练**:PyTorch Elastic、DeepSpeed Elastic,故障卡剔除后重组进程组继续训,不用从头来。但超大集群(>1000 卡)才值得,小集群重启更快。

## 五、算一笔完整的账:8×4090 ZeRO-3 的通信

### 5.1 场景设定

8B 模型、ZeRO-3、micro-batch 8 条序列/卡、8 卡 4090(无 NVLink/P2P,PCIe 4.0 x16 32GB/s)。

### 5.2 通信量计算

每 micro-batch 三次集合通信:

1. **前向 AllGather**:收集 BF16 权重 16.4GB,每卡收发 $16.4 \times 7/8 = 14.3$GB;
2. **反向 AllGather**:同上;
3. **ReduceScatter**:梯度 16.4GB,每卡收发 $14.3$GB。

一步(8 micro-batch)每卡累计:$3 \times 14.3 \times 8 = 343$GB。

### 5.3 链路时间

4090 无 NVLink/P2P,全部流量走 PCIe 4.0 x16(32GB/s):

$$
T_{\text{链路}} = 343\text{GB} / 32\text{GB/s} = 10.7\text{s/步}
$$

### 5.4 计算时间

8B 模型、batch=8、seq=1024、6ND:

$$
T_{\text{计算}} = 6 \times 8.2\text{B} \times 8 \times 1024 / (8 \times 165\text{TFLOPS} \times 0.3) = 5.2\text{s/步}
$$

### 5.5 通信隐藏的效果

如果通信完全暴露:步时间 $10.7 + 5.2 = 15.9$s,通信占比 67%。

靠预取重叠(ZeRO-3 逐层预取参数、逐层归约梯度),实际通信等待可压到亚秒级:

- 第一次 AllGather(词嵌入,第一层):$14.3$GB / 32GB/s = 0.45s;
- 最后一次 ReduceScatter(词嵌入,最后一层):0.45s;
- 其余通信藏在计算里。

实际步时间约 $5.2 + 0.9 = 6.1$s,通信占比 15%——可接受。

**但注意**:PCIe 同时还要服务数据加载与激活搬运,余量远比 H100 + NVLink 机器紧张。这就是为什么 4090 集群的 MFU 普遍低于 H100(第五篇实测 30% vs 40%)。

## 六、一个完整的多机训练配置示例

### 6.1 场景

128 卡 A100(16 台 8 卡机),训练 13B 模型,TP=8、PP=2、DP=8、ZeRO-1。

### 6.2 网络拓扑

- 机内:8×A100 NVLink(600GB/s);
- 跨机:16 台机 × 8×200G IB,rail-optimized;
- 存储:共享 NFS,checkpoint 写到中央存储。

### 6.3 NCCL 配置

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3  # RoCE v2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_ALGO=Ring  # 大消息强制 Ring
```

### 6.4 启动命令

```bash
# 每台机跑
torchrun --nnodes=16 --nproc_per_node=8 \
         --rdzv_id=123 --rdzv_backend=c10d \
         --rdzv_endpoint=worker0:29500 \
         train.py \
         --tensor-parallel-size 8 \
         --pipeline-parallel-size 2 \
         --zero-stage 1
```

### 6.5 预期性能

- 计算时间:13B、batch=16、seq=4096、6ND,128 卡,MFU 40%,约 2.1s/步;
- TP 通信:每层 4 次 AllReduce,NVLink 1ms/层,26 层 26ms,PP 后 13 层 13ms;
- PP 通信:每 micro-batch 段间 268MB,IB 25GB/s,10.7ms;
- DP 通信:梯度 AllReduce,ZeRO-1 后 $2 \times 13\text{B}/8 \times 7/8 = 5.7$GB,Hierarchical 后跨机 1.4GB,IB 25GB/s,56ms,梯度桶隐藏 80%,暴露 11ms。

步时间:$2.1 + 0.013 + 0.011 + 0.011 = 2.135$s,通信占比 1.6%——优秀。

## 小结

- **层级带宽**:NVLink 450-900 GB/s → PCIe 32-64 GB/s → IB/网卡 25-50 GB/s,三个数量级决定 TP/PP/DP 的空间布局;
- **PS vs AllReduce**:PS 的中心带宽瓶颈使其退出 LLM 主流;AllReduce 架构 + NCCL 是同步训练的标准执行层;
- **NCCL 原语**:AllReduce = ReduceScatter + AllGather;Ring 带宽优、Tree 时延优,NCCL 按拓扑自动选择,`NCCL_DEBUG=INFO` 是排障第一入口;
- **多机三约束**:通信隐藏(排进计算窗)、straggler(同步等待最慢卡)、容错(checkpoint 周期权衡);
- **Hierarchical**:机内 NVLink + 跨机 IB 分层归约,跨机流量降到 $1/8$。

下一篇:[LLM分布式训练5——DeepSpeed 实操:ZeRO 全阶段实测](/posts/eac6a656.html)
