# 完全分片数据并行

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/accelerate/fsdp>
>
> 原始地址：<https://huggingface.co/docs/peft/accelerate/fsdp>


[完全分片数据并行](https://pytorch.org/docs/stable/fsdp.html)
 (FSDP) 是为高达 1T 参数的大型预训练模型的分布式训练而开发的。 FSDP 通过跨数据并行进程对模型参数、梯度和优化器状态进行分片来实现这一点，并且它还可以将分片的模型参数卸载到 CPU。 FSDP 提供的内存效率允许您将训练扩展到更大的批次或模型大小。


目前，FSDP 不会减少 GPU 内存使用量，并且具有 CPU 卸载功能的 FSDP 在训练期间实际上会消耗 1.65 倍的 GPU 内存。你可以追踪这个 PyTorch
 [问题](https://github.com/pytorch/pytorch/issues/91165)
 如有任何更新。


🤗 Accelerate 支持 FSDP，您可以将其与 🤗 PEFT 一起使用。本指南将帮助您了解如何使用我们的 FSDP
 [训练脚本](https://github.com/huggingface/peft/blob/main/examples/conditional_ Generation/peft_lora_seq2seq_accelerate_fsdp.py)
 。您将配置脚本来训练大型模型以进行条件生成。


## 配置



首先运行以下命令
 [创建 FSDP 配置文件](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
 与 🤗 加速。使用
 `--config_file`
 标志将配置文件保存到特定位置，否则保存为
 `default_config.yaml`
 🤗 加速缓存中的文件。


配置文件用于在启动训练脚本时设置默认选项。



```
accelerate config --config_file fsdp_config.yaml
```


系统会询问您一些有关您的设置的问题，并配置以下参数。对于此示例，请确保完全分片模型参数、梯度、优化器状态，​​利用 CPU 进行卸载，并根据 Transformer 层类名称包装模型层。



```
`Sharding Strategy`: [1] FULL_SHARD (shards optimizer states, gradients and parameters), [2] SHARD_GRAD_OP (shards optimizer states and gradients), [3] NO_SHARD
`Offload Params`: Decides Whether to offload parameters and gradients to CPU
`Auto Wrap Policy`: [1] TRANSFORMER_BASED_WRAP, [2] SIZE_BASED_WRAP, [3] NO_WRAP 
`Transformer Layer Class to Wrap`: When using `TRANSFORMER_BASED_WRAP`, user specifies comma-separated string of transformer layer class names (case-sensitive) to wrap ,e.g, 
`BertLayer`, `GPTJBlock`, `T5Block`, `BertLayer,BertEmbeddings,BertSelfOutput`...
`Min Num Params`: minimum number of parameters when using `SIZE_BASED_WRAP`
`Backward Prefetch`: [1] BACKWARD_PRE, [2] BACKWARD_POST, [3] NO_PREFETCH
`State Dict Type`: [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT  
```


例如，您的 FSDP 配置文件可能如下所示：



```
command\_file: null
commands: null
compute\_environment: LOCAL\_MACHINE
deepspeed\_config: {}
distributed\_type: FSDP
downcast\_bf16: 'no'
dynamo\_backend: 'NO'
fsdp\_config:
  fsdp\_auto\_wrap\_policy: TRANSFORMER\_BASED\_WRAP
  fsdp\_backward\_prefetch\_policy: BACKWARD\_PRE
  fsdp\_offload\_params: true
  fsdp\_sharding\_strategy: 1
  fsdp\_state\_dict\_type: FULL\_STATE\_DICT
  fsdp\_transformer\_layer\_cls\_to\_wrap: T5Block
gpu\_ids: null
machine\_rank: 0
main\_process\_ip: null
main\_process\_port: null
main\_training\_function: main
megatron\_lm\_config: {}
mixed\_precision: 'no'
num\_machines: 1
num\_processes: 2
rdzv\_backend: static
same\_network: true
tpu\_name: null
tpu\_zone: null
use\_cpu: false
```


## 重要部分



让我们更深入地研究训练脚本以了解它是如何工作的。


这
 [`main()`](https://github.com/huggingface/peft/blob/2822398fbe896f25d4dac5e468624dc5fd65a51b/examples/conditional_ Generation/peft_lora_seq2seq_accelerate_fsdp.py#L14)
 函数首先初始化一个
 `加速器`
 类处理分布式训练的所有内容，例如自动检测您的训练环境。


💡 随意更改里面的模型和数据集
 `主要`
 功能。如果您的数据集格式与脚本中的格式不同，您可能还需要编写自己的预处理函数。


该脚本还会创建与您正在使用的 🤗 PEFT 方法相对应的配置。对于 LoRA，您将使用
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 指定任务类型以及其他几个重要参数，例如低秩矩阵的维度、矩阵缩放因子和 LoRA 层的丢失概率。如果您想使用不同的 🤗 PEFT 方法，请替换
 `LoraConfig`
 与适当的
 [类](../package_reference/tuners)
 。


接下来，脚本包装基本模型并
 `peft_config`
 与
 `get_peft_model()`
 函数来创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。



```
 def main():
+ accelerator = Accelerator()
     model_name_or_path = "t5-base"
     base_path = "temp/data/FinancialPhraseBank-v1.0"
+ peft\_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
     )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+ model = get\_peft\_model(model, peft\_config)
```


在整个脚本中，您将看到
 `main_process_first`
 和
 `等待大家`
 帮助控制和同步进程执行的函数。


准备好数据集并加载所有必要的训练组件后，脚本会检查您是否正在使用
 `fsdp_插件`
 。 PyTorch 提供了两种在 FSDP 中包装模型层的方法：自动或手动。最简单的方法是让 FSDP 自动递归包装模型层，而无需更改任何其他代码。您可以选择根据图层名称或大小（参数数量）来包装模型图层。在 FSDP 配置文件中，它使用
 `TRANSFORMER_BASED_WRAP`
 选项来包装
 `T5块`
 层。



```
if getattr(accelerator.state, "fsdp\_plugin", None) is not None:
    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
```


接下来，使用 🤗 Accelerate
 `准备`
 函数来准备模型、数据集、优化器和调度器以进行训练。



```
model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
)
```


从这里开始，脚本的其余部分处理训练循环、评估并将模型共享到中心。


## 火车



运行以下命令来启动训练脚本。之前，您将配置文件保存到
 `fsdp_config.yaml`
 ，因此您需要使用以下命令将路径传递给启动器
 `--config_file`
 像这样的论点：



```
accelerate launch --config_file fsdp_config.yaml examples/peft_lora_seq2seq_accelerate_fsdp.py
```


训练完成后，脚本会返回准确性并将预测与标签进行比较。