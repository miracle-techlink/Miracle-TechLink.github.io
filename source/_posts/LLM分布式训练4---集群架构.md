---
title: LLM分布式训练4---集群架构
date: 2025-04-23 00:00:00
updated: 2026-10-25 00:00:00
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

前三篇的并行策略都默认"有一张能跑集合通信的网"。本篇把这张网拆开看:GPU 之间到底经过什么、各层带宽差多少、NCCL 怎么在真实拓扑上选路,以及多机训练的工程约束。[第三篇](/posts/212a16ef.html)的结论——TP 住 NVLink 域、PP 跨机——只有放在本篇的带宽量级表里才能理解。

## 一、层级互联:带宽差三个数量级

一块 GPU 到另一块 GPU 的路径,按代价从低到高:

| 层级 | 典型介质 | 单向带宽量级 | 特点 |
| --- | --- | --- | --- |
| 卡内 | HBM3e 显存 | 3–5 TB/s(H100/B200) | 访问即算,不构成通信 |
| 卡间(同机 NVLink) | NVLink 4/5 + NVSwitch | 450–900 GB/s | GPU 显存直连,全域一致性,TP 的家 |
| 卡间(无 NVLink,同机) | PCIe 4.0/5.0 x16 | 32–64 GB/s | 走主机内存中转,无 P2P 时两次搬运 |
| 机器间 | IB / RoCE(NDR 400G) | 25–50 GB/s/网卡 | 可聚合(多网卡分担),时延微秒级 |
| 机器间(降级) | TCP 以太网 | 1–12.5 GB/s | 拥塞控制抖动大,训练应尽量避免 |

三个要点:

1. **NVLink 与 PCIe 差一个数量级,机器网络与 NVLink 又差一个数量级**——这就是 TP 必须锁在单机内的物理根据;
2. **RTX 4090 是"NVLink 缺席"的典型**:无 NVLink、卡间 P2P 被禁用,同机两卡交换数据也要各穿一次 PCIe 到主机内存再回来。8×4090 主机做 ZeRO-3,每步每卡数百 GB 的收发全部压在 32 GB/s 的 PCIe 上(第五篇实测会看到它对吞吐的直接影响);
3. 机器间带宽靠**网卡聚合**补:8 卡机配 8×200 Gbit/s 网卡,环形集合通信把跨机流量摊到 8 张卡各自的网卡,单网卡 25 GB/s 不构成瓶颈——前提是**流量按卡对称**(Ring 天然满足)。

组网上,大集群普遍采用 **rail-optimized(轨道优化)**:每张 GPU 直连一张独立网卡,同轨(同卡序)的网卡接同一台叶子交换机,跨机同卡序通信一跳直达;配合 fat-tree 无收敛比架构,Any-to-Any 路径带宽不缩水。

## 二、参数服务器 vs 集合通信架构

**参数服务器(PS)**:worker 算梯度,推给 server;server 聚合、更新参数,worker 拉回。问题在带宽集中:server 侧要吃下 $O(d \cdot \Psi)$ 流量,千亿模型下没有任何单点扛得住,只能再做 server 分片,复杂度雪崩。PS 的真实优势只剩**异步更新**(无需等最慢的 worker),但异步引入梯度陈旧(staleness),对 LLM 训练的收敛性伤害大于收益,现代 LLM 栈已基本放弃。

**AllReduce 架构**:没有中心,梯度经环形/树形集合通信在全体卡间流动,每卡收发量对称且与卡数近似无关(第二篇推导)。同步语义、无单点、与 NVLink/IB 拓扑天然契合——NCCL 就是这一架构的工程化。

## 三、NCCL:集合通信的执行层

### 原语清单

| 原语 | 语义 | 每卡通信量($V$ 字节输入) | 训练中的角色 |
| --- | --- | --- | --- |
| Broadcast | 0 号卡数据发全员 | $V$ | 初始参数同步 |
| Reduce | 归约到 0 号卡 | $\approx V$ | 聚合 loss/metric |
| AllReduce | 全员归约全员接收 | $2V(d-1)/d$(Ring) | DDP 梯度同步 |
| ReduceScatter | 归约 + 按片分发 | $V(d-1)/d$ | ZeRO-2/3 梯度分片 |
| AllGather | 各卡片段拼全量 | $V(d-1)/d$ | ZeRO-3 权重收集 |
| AllToAll | 每卡对每卡点对点 | $V$ | MoE 专家路由、TP 头重排 |

AllReduce $=$ ReduceScatter $+$ AllGather,这个分解正是 ZeRO-2/3 省通信的原理(第二篇)。

### Ring vs Tree

- **Ring**:带宽最优(链路均摊、每卡收发量恒定),但同步步数 $2(d-1)$,小消息被启动时延支配;
- **Tree**(递归半倍增):步数 $2\log_2 d$,小消息时延最优,但根节点附近链路负载集中;
- NCCL 运行时按消息大小、拓扑、卡数自动选择,也可 `NCCL_ALGO=Ring/Tree/PAT` 强制。多机场景还有 ** hierarchical 模式:机内先 NVLink Ring 归约出每机一份,再跨机网卡 Ring 归约单卡代表,最后机内 Broadcast——把昂贵路径上的流量降到 $1/8$。

### 拓扑感知与排障

NCCL 启动时探测系统拓扑(PCIe 桥树、NVLink、网卡亲和),构建 channel 把通信绑到最优路径。两个高频环境变量:

```bash
NCCL_DEBUG=INFO          # 打印拓扑探测与算法选择,排障第一步
NCCL_P2P_DISABLE=1       # 禁 P2P(4090 这类无 P2P 平台排查用)
NCCL_ALGO=Tree           # 强制算法
NCCL_IB_DISABLE=1        # 单机调试时禁 IB
```

"训练卡住但 CPU 空转"的排查顺序:`NCCL_DEBUG=INFO` 看是否卡在拓扑探测 → 看 `nvidia-smi topo -m` 的卡间连接类型(NV#/PIX/PHB/SYS)→ 检查 IB 网卡是否被整机共享(`ibstat`)。

## 四、多机训练的工程约束

**通信隐藏**。链路时间 ≠ 通信等待:ZeRO-3 逐层预取参数、逐层归约梯度,MegaScale 的测量显示无法隐藏的只剩每步第一次 AllGather(词嵌入)与最后一次 ReduceScatter,约 0.08 s/步(8B 模型、48 卡)。优化目标因此不是"减少通信量",而是**把通信排进计算的时间窗**。

**Straggler(掉队者)**。千卡集群每步都有最慢的一张卡,全体同步等待它的梯度。DDP 的步耗时由最慢 rank 决定,慢卡来源包括:共置任务抢 PCIe、NUMA 错位的数据加载、ECC 纠错降频。MegaScale/Llama-3 级别的运维会做 rank 级耗时画像并隔离长尾任务。

**容错与 checkpoint**。千卡数周的 MTBF(平均无故障时间)以小时计:梯度同步语义下任何一卡故障会冻结全体,所以 checkpoint 周期是吞吐与恢复成本的权衡——Llama-3 每 4 小时存档一次,占训练时间约 0.01%,恢复从最近档回放。弹性训练(故障卡剔除重组进程组)在超大集群上才开始值得。

## 五、算一笔完整的账:8×4090 ZeRO-3 的通信

把本篇的量级表用起来。8B 模型、ZeRO-3、micro-batch 8 条序列/卡:

- 每 micro-batch 三次集合通信:前向 AllGather(BF16 权重 16.4 GB 的 $1/8$ 每卡收发)+ 反向 AllGather + ReduceScatter(梯度);
- 一步(8 micro-batch)每卡累计约 385 GB 收发;
- 4090 无 NVLink/P2P,全部流量走 PCIe 4.0 x16(32 GB/s):链路时间约 12 s/步;
- 而同样的卡计算只需约 52 s/步:靠预取重叠,实际通信等待可压到亚秒级,但 PCIe 同时还要服务数据加载与激活搬运,**余量远比 H100 + NVLink 机器紧张**。

这笔账的每个数字都会在[第五篇](/posts/eac6a656.html)的实测里以吞吐形式落地。

## 小结

- 层级带宽:NVLink 450–900 GB/s → PCIe 32–64 GB/s → IB/网卡 25–50 GB/s,三个数量级决定 TP/PP/DP 的空间布局;
- PS 的中心带宽瓶颈使其退出 LLM 主流;AllReduce 架构 + NCCL 是同步训练的标准执行层;
- AllReduce $=$ ReduceScatter $+$ AllGather;Ring 带宽优、Tree 时延优,NCCL 按拓扑自动选择,`NCCL_DEBUG=INFO` 是排障第一入口;
- 多机三约束:通信隐藏(排进计算窗)、straggler(同步等待最慢卡)、容错(checkpoint 周期权衡)。

下一篇:[LLM分布式训练5——DeepSpeed 实操:ZeRO 全阶段实测](/posts/eac6a656.html)
