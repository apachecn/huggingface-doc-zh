# 深速

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/accelerate/deepspeed-zero3-offload>
>
> 原始地址：<https://huggingface.co/docs/peft/accelerate/deepspeed-zero3-offload>


[DeepSpeed](https://www.deepspeed.ai/)
 是一个专为具有数十亿参数的大型模型的分布式训练的速度和规模而设计的库。其核心是零冗余优化器 (ZeRO)，它将优化器状态 (ZeRO-1)、梯度 (ZeRO-2) 和参数 (ZeRO-3) 跨数据并行进程进行分片。这大大减少了内存使用量，使您能够将训练扩展到十亿个参数模型。为了释放更高的内存效率，ZeRO-Offload 在优化过程中利用 CPU 资源来减少 GPU 计算和内存。


🤗 Accelerate 支持这两个功能，您可以将它们与 🤗 PEFT 一起使用。本指南将帮助您了解如何使用我们的 DeepSpeed
 [训练脚本](https://github.com/huggingface/peft/blob/main/examples/conditional_ Generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py)
 。您将配置脚本来训练大型模型，以使用 ZeRO-3 和 ZeRO-Offload 进行条件生成。


💡 为了帮助您入门，请查看我们的示例培训脚本
 [因果语言建模](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py)
 和
 [条件生成](https://github.com/huggingface/peft/blob/main/examples/conditional_ Generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py)
 。您可以根据自己的应用程序调整这些脚本，如果您的任务与脚本中的任务类似，甚至可以直接使用它们。


## 配置



首先运行以下命令
 [创建 DeepSpeed 配置文件](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script)
 与 🤗 加速。这
 `--config_file`
 flag 允许您将配置文件保存到特定位置，否则将保存为
 `default_config.yaml`
 🤗 加速缓存中的文件。


配置文件用于在启动训练脚本时设置默认选项。



```
accelerate config --config_file ds_zero3_cpu.yaml
```


系统会询问您一些有关您的设置的问题，并配置以下参数。在此示例中，您将使用 ZeRO-3 和 ZeRO-Offload，因此请确保选择这些选项。



```
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them.
`gradient_clipping`: Enable gradient clipping with value.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. 
```


一个例子
 [配置文件](https://github.com/huggingface/peft/blob/main/examples/conditional_ Generation/accelerate_ds_zero3_cpu_offload_config.yaml)
 可能如下所示。最需要注意的是
 `零阶段`
 被设定为
 `3`
 ， 和
 `offload_optimizer_device`
 和
 `offload_param_device`
 被设置为
 `CPU`
 。



```
compute\_environment: LOCAL\_MACHINE
deepspeed\_config:
  gradient\_accumulation\_steps: 1
  gradient\_clipping: 1.0
  offload\_optimizer\_device: cpu
  offload\_param\_device: cpu
  zero3\_init\_flag: true
  zero3\_save\_16bit\_model: true
  zero\_stage: 3
distributed\_type: DEEPSPEED
downcast\_bf16: 'no'
dynamo\_backend: 'NO'
fsdp\_config: {}
machine\_rank: 0
main\_training\_function: main
megatron\_lm\_config: {}
mixed\_precision: 'no'
num\_machines: 1
num\_processes: 1
rdzv\_backend: static
same\_network: true
use\_cpu: false
```


## 重要部分



让我们更深入地研究一下脚本，这样您就可以看到发生了什么，并了解它是如何工作的。


内
 [`main`](https://github.com/huggingface/peft/blob/2822398fbe896f25d4dac5e468624dc5fd65a51b/examples/conditional_ Generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py#L103)
 函数，该脚本创建一个
 `加速器`
 类来初始化分布式训练的所有必要要求。


💡 随意更改里面的模型和数据集
 `主要`
 功能。如果您的数据集格式与脚本中的格式不同，您可能还需要编写自己的预处理函数。


该脚本还为您正在使用的 🤗 PEFT 方法创建配置，在本例中为 LoRA。这
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 指定任务类型和重要参数，例如低秩矩阵的维度、矩阵缩放因子和 LoRA 层的丢失概率。如果您想使用不同的 🤗 PEFT 方法，请确保替换
 `LoraConfig`
 与适当的
 [类](../package_reference/tuners)
 。



```
 def main():
+ accelerator = Accelerator()
     model_name_or_path = "facebook/bart-large"
     dataset_name = "twitter_complaints"
+ peft\_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
     )
```


在整个脚本中，您将看到
 `main_process_first`
 和
 `等待大家`
 帮助控制和同步进程执行的函数。


这
 `get_peft_model()`
 函数采用一个基本模型，并且
 `peft_config`
 您之前准备创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 ：



```
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+ model = get\_peft\_model(model, peft\_config)
```


将所有相关训练对象传递给🤗 Accelerate
 `准备`
 这确保一切都准备好进行训练：



```
model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
)
```


下一段代码检查是否在中使用了 DeepSpeed 插件
 `加速器`
 ，如果插件存在，那么
 `加速器`
 使用配置文件中指定的 ZeRO-3：



```
is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed\_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
```


在训练循环中，通常
 `loss.backward()`
 被 🤗 Accelerate 取代
 ‘落后’
 它使用正确的
 `向后()`
 根据您的配置的方法：



```
  for epoch in range(num_epochs):
      with TorchTracemalloc() as tracemalloc:
          model.train()
          total_loss = 0
          for step, batch in enumerate(tqdm(train_dataloader)):
              outputs = model(**batch)
              loss = outputs.loss
              total_loss += loss.detach().float()
+ accelerator.backward(loss)
              optimizer.step()
              lr_scheduler.step()
              optimizer.zero_grad()
```


就这些！脚本的其余部分处理训练循环、评估，甚至将其推送到中心。


## 火车



运行以下命令来启动训练脚本。之前，您将配置文件保存到
 `ds_zero3_cpu.yaml`
 ，因此您需要使用以下命令将路径传递给启动器
 `--config_file`
 像这样的论点：



```
accelerate launch --config_file ds_zero3_cpu.yaml examples/peft_lora_seq2seq_accelerate_ds_zero3_offload.py
```


您将看到一些跟踪训练期间内存使用情况的输出日志，一旦完成，脚本就会返回准确性并将预测与标签进行比较：



```
GPU Memory before entering the train : 1916
GPU Memory consumed at the end of the train (end-begin): 66
GPU Peak Memory consumed during the train (max-begin): 7488
GPU Total Peak Memory consumed during the train (max): 9404
CPU Memory before entering the train : 19411
CPU Memory consumed at the end of the train (end-begin): 0
CPU Peak Memory consumed during the train (max-begin): 0
CPU Total Peak Memory consumed during the train (max): 19411
epoch=4: train_ppl=tensor(1.0705, device='cuda:0') train_epoch_loss=tensor(0.0681, device='cuda:0')
100%|████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:27<00:00,  3.92s/it]
GPU Memory before entering the eval : 1982
GPU Memory consumed at the end of the eval (end-begin): -66
GPU Peak Memory consumed during the eval (max-begin): 672
GPU Total Peak Memory consumed during the eval (max): 2654
CPU Memory before entering the eval : 19411
CPU Memory consumed at the end of the eval (end-begin): 0
CPU Peak Memory consumed during the eval (max-begin): 0
CPU Total Peak Memory consumed during the eval (max): 19411
accuracy=100.0
eval_preds[:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
```