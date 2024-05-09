

# 配置和模型

> 译者：[糖醋鱼](https://github.com/now-101)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/tutorial/peft_model_config>
>
> 原始地址：<https://huggingface.co/docs/peft/tutorial/peft_model_config>

## PEFT配置与模型概述
当今的大型预训练模型通常拥有数十亿个参数，其庞大的规模给训练带来了显著的挑战，因为它们需要更多的存储空间和更强的计算能力来处理所有这些计算。你需要访问强大的GPU或TPU来训练这些大型预训练模型，这是昂贵的，不是每个人都能轻易获得的，对环境也不友好，而且并不是非常实际的。PEFT方法解决了许多这些挑战。PEFT方法有几种类型（软提示、矩阵分解、适配器），但它们都专注于同一件事，即减少可训练参数的数量。这使得在消费者硬件上训练和存储大型模型更加容易。

PEFT库旨在帮助你在免费或低成本的GPU上快速训练大型模型，在本教程中，你将学习如何设置一个配置，将PEFT方法应用于预训练基础模型进行训练。设置完成PEFT的配置后，你可以使用任何喜欢的训练框架（Transformer的Trainer类、Accelerate、自定义的PyTorch训练循环）。


## PEFT配置
> 要了解更多关于每种PEFT方法可以配置的参数，请参阅它们各自的API参考页面。

配置文件存储了指定特定PEFT方法应如何应用的重要参数。

例如，查看以下用于应用LoRA的LoraConfig和用于应用p-tuning的PromptEncoderConfig（这些配置文件已经是JSON序列化的）。无论何时加载PEFT适配器，检查它是否有一个必需的关联adapter_config.json文件都是一个好主意。

**LoraConfig**
```json
{
  "base_model_name_or_path": "facebook/opt-350m", #base model to apply LoRA to
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA", #PEFT method type
  "r": 16,
  "revision": null,
  "target_modules": [
    "q_proj", #model modules to apply LoRA to (query and value projection layers)
    "v_proj"
  ],
  "task_type": "CAUSAL_LM" #type of task to train model on
}
```
可以通过初始化 LoraConfig 来创建自己的训练配置。
例如，以下是如何创建一个LoraConfig实例的示例代码：
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
```
## PEFT 模型

有了PEFT配置之后，你现在可以将其应用于任何预训练模型，以创建一个`PeftModel`。你可以从Transformers库中的任何最先进的模型中选择，也可以选择自定义模型，甚至是新的和不支持的transformer架构。

对于本教程，我们将加载一个基础的`facebook/opt-350m`模型进行微调。以下是如何加载模型并应用PEFT配置的示例代码：
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```

使用 `get_peft_model()` 函数从基础的 `facebook/opt-350m` 模型和之前创建的 `lora_config` 来创建一个 PeftModel。

```python
from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
"trainable params: 1,572,864 || all params: 332,769,280 || trainable%: 0.472659014678278"
```

现在你可以使用你喜欢的训练框架对PeftModel进行训练！训练完成后，你可以使用 `save_pretrained()` 方法将模型保存在本地，或使用 `push_to_hub` 方法将其上传到Hub。

```python
# save locally
lora_model.save_pretrained("your-name/opt-350m-lora")

# push to Hub
lora_model.push_to_hub("your-name/opt-350m-lora")
```

要加载 `PeftModel` 进行推理，您需要提供用于创建它的 `PeftConfig` 以及训练它的基本模型。

```python
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
````

>默认情况下，PeftModel被设置为推断模式，但如果您想要进一步训练适配器，您可以将 `is_trainable=True`。
> ```python 
>lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora", >is_trainable=True)
> ```

`PeftModel.from_pretrained()` 方法是加载 `PeftModel` 最灵活的方式，因为它不关心使用了哪种模型框架（Transformers、timm、通用的PyTorch模型）。其他类，如 `AutoPeftModel`，只是基础 `PeftModel` 的便捷包装，使得直接从 Hub 或本地加载 PEFT 模型更加容易，其中存储了 PEFT 权重。

```python
from peft import AutoPeftModelForCausalLM

lora_model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
```
查看 AutoPeftModel API 参考以了解有关 AutoPeftModel 类的更多信息。

## 下一步
使用适当的 `PeftConfig`，您可以将其应用于任何预训练模型以创建一个 `PeftModel` ，并在免费提供的 GPU 上更快地训练大型强大模型！要了解更多关于 PEFT 配置和模型的信息，以下指南可能会有所帮助：

> 学习如何为不是来自 Transformers 的模型配置 PEFT 方法，请参阅“使用自定义模型”指南。
