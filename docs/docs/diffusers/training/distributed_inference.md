# 使用多个 GPU 进行分布式推理

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/distributed_inference>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/distributed_inference>


在分布式设置中，您可以使用 🤗 跨多个 GPU 运行推理
 [加速](https://huggingface.co/docs/accelerate/index)
 或者
 [PyTorch 分布式](https://pytorch.org/tutorials/beginner/dist_overview.html)
 ，这对于并行生成多个提示很有用。


本指南将向您展示如何使用 🤗 Accelerate 和 PyTorch Distributed 进行分布式推理。


## 🤗 加速



🤗
 [加速](https://huggingface.co/docs/accelerate/index)
 是一个旨在轻松跨分布式设置训练或运行推理的库。它简化了设置分布式环境的过程，使您可以专注于 PyTorch 代码。


首先，创建一个 Python 文件并初始化
 `accelerate.PartialState`
 创建分布式环境；您的设置会自动检测到，因此您无需显式定义
 `等级`
 或者
 `世界大小`
 。移动
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 到
 `distributed_state.device`
 为每个进程分配一个 GPU。


现在使用
 `进程之间的分割`
 作为上下文管理器的实用程序可以在多个进程之间自动分配提示。



```
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result\_{distributed\_state.process\_index}.png")
```


使用
 `--num_processes`
 参数指定要使用的 GPU 数量，并调用
 ‘加速发射’
 运行脚本：



```
accelerate launch run_distributed.py --num_processes=2
```


要了解更多信息，请查看
 [使用 🤗 Accelerate 进行分布式推理](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate)
 指导。


## PyTorch 分布式



PyTorch 支持
 [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
 这使得数据并行。


首先，创建一个 Python 文件并导入
 `torch.distributed`
 和
 `torch.multiprocessing`
 设置分布式进程组并在每个 GPU 上生成用于推理的进程。您还应该初始化一个
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 ：



```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

sd = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
```


您需要创建一个函数来运行推理；
 [`init_process_group`](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group)
 处理创建具有要使用的后端类型的分布式环境，
 `等级`
 当前进程的，以及
 `世界大小`
 或参与的进程数。如果您在 2 个 GPU 上并行运行推理，那么
 `世界大小`
 是 2。


移动
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 到
 `等级`
 并使用
 `获取排名`
 为每个进程分配一个 GPU，其中每个进程处理不同的提示：



```
def run\_inference(rank, world\_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    sd.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    image = sd(prompt).images[0]
    image.save(f"./{'\_'.join(prompt)}.png")
```


要运行分布式推理，请调用
 [`mp.spawn`](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn)
 运行
 `运行推理`
 函数中定义的 GPU 数量
 `世界大小`
 ：



```
def main():
    world_size = 2
    mp.spawn(run_inference, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "\_\_main\_\_":
    main()
```


完成推理脚本后，使用
 `--nproc_per_node`
 指定要使用和调用的 GPU 数量的参数
 `torchrun`
 运行脚本：



```
torchrun run_distributed.py --nproc_per_node=2
```