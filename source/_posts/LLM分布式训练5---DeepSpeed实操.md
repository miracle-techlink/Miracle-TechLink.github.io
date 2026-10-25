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

前四篇把公式推完了,本篇把公式放到 GPU 上对账:在 4×RTX 4090 上逐阶段实测 ZeRO 的显存与吞吐,验证[第二篇](/posts/3a50363.html)的 $16\Psi$ 账单和通信代价模型。所有数字来自真实运行(随机初始化的 GPT-2 架构模型 + 合成数据,消除数据管线干扰),脚本与原始日志在文末环境块可复现。

<!-- BENCHMARK_RESULTS_PLACEHOLDER -->

## 一、DeepSpeed 在训练栈里的位置

DeepSpeed 不是"另一个 PyTorch":它是一层**训练执行引擎**,接管三件事——参数/梯度/优化器状态的分布方式(ZeRO)、通信调度(预取、分桶、overlap)、以及精度管理(fp16/bf16 的 loss scaling 与主权重维护)。模型定义、前向反向计算仍由你的 PyTorch 代码执行。

最小接入只有一行:

```python
import deepspeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,                # 你的 nn.Module
    optimizer=optimizer,        # 已绑定的 AdamW 等
    config="ds_zero2.json",     # 一切策略都在这个 json 里
)
```

`initialize` 之后,`model` 换成了 `DeepSpeedEngine`:前向照常 `output = model(input)`,反向照常 `loss.backward()`,但梯度归约、参数分片、优化器步进全部由引擎按 json 策略执行;`optimizer.step()` 也可省略(engine 在 `backward` 末尾自动触发)。

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

以 ZeRO-2 + 优化器卸载为例,逐键注释:

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

逐项说明:

- **三个 batch 的恒等式**:`train_batch_size = num_gpus × train_micro_batch_size_per_gpu × gradient_accumulation_steps`,DeepSpeed 会强校验——写成不一致直接拒绝启动;
- **`bf16.enabled`**:4090/A100/H100 都建议 bf16(指数位与 FP32 对齐,不需要 fp16 的 loss scaling);`fp16.enabled` 是老卡路线,须配 `loss_scale`;
- **`stage`**: 1=优化器状态分片,2=+梯度分片,3=+权重分片,显存/通信权衡见第二篇;
- **`overlap_comm` / `contiguous_gradients` / `reduce_bucket_size`**:梯度桶三件套——归约与反向计算重叠、梯度写进连续缓冲(避免碎片)、桶 500 MB 默认值对小模型可调小;
- **`offload_optimizer.device: cpu`**:Adam 的 $m/v$ 与主权重搬到主存,换 PCIe 流量;`pin_memory` 锁页让 H2D 拷贝走 DMA;
- **stage 3 追加项**(本篇实测用到):`stage3_gather_16bit_weights_on_model_save`(存档时聚合完整权重)、`stage3_prefetch_bucket_size`、`stage3_param_persistence_threshold`(前向内多次使用的参数留在卡上)。

## 四、实测设计

**平台**:4×RTX 4090 24GB(无 NVLink、无卡间 P2P,卡间流量走 PCIe 4.0 x16,见[第四篇](/posts/74a233f5.html)的账);主机内存 503 GB;CUDA 12.6。

**模型**:GPT-2 架构、随机初始化、vocab 50257、seq_len 1024、合成 token 数据:

- **模型 A ≈ 1.5B**(gpt2-xl 级):单卡 DDP 可跑,考察各阶段在"放得下"时的开销;
- **模型 B ≈ 6.8B**(hidden 4096 × 32 层):单卡 DDP 必然 OOM,考察 ZeRO 的"从不可行到可行"。

**矩阵**:每模型 × {ZeRO-0(DDP), 1, 2, 3, 2+CPU offload, 3+CPU offload} × {1, 2, 4 卡};micro-batch 逐配置二分取最大可行值;warmup 10 步后测 30 步吞吐(tokens/s)与 `torch.cuda.max_memory_reserved`(峰值显存)。

**对照理论**:模型 B 状态量 $16\Psi = 16 \times 6.8\text{B} = 108.8$ GB,4 卡 ZeRO-3 每卡常驻状态应为 $108.8/4 = 27.2$ GB——已经超过 24 GB 显存,故模型 B 在 4 卡 ZeRO-3 下必须叠加重计算或 offload 才能跑;ZeRO-2 每卡 $2\Psi + 14\Psi/d = 13.6 + 38.1 = 51.7$ GB,不可行。这些"纸面预测→实测对账"正是本篇的目的。

## 五、实测结果

<!-- RESULTS_TABLE_PLACEHOLDER -->

<!-- RESULTS_ANALYSIS_PLACEHOLDER -->

## 六、踩坑实录

<!-- PITFALLS_PLACEHOLDER -->

## 小结

<!-- SUMMARY_PLACEHOLDER -->

## 复现环境

<!-- ENV_BLOCK_PLACEHOLDER -->

本系列完。回看:[篇1 基础](/posts/9aac52d9.html) · [篇2 数据并行](/posts/3a50363.html) · [篇3 模型并行](/posts/212a16ef.html) · [篇4 集群架构](/posts/74a233f5.html)
