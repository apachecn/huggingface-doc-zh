# PEFT 作为实用程序库

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/developer_guides/low_level_api>
>
> 原始地址：<https://huggingface.co/docs/peft/developer_guides/low_level_api>


让我们在本节中介绍如何利用 PEFT 的低级 API 将可训练适配器注入任何
 `torch`
 模块。
开发此 API 的动机是超级用户不需要依赖 PEFT 库中公开的建模类，并且仍然能够使用 LoRA、IA3 和 AdaLoRA 等适配器方法。


## 支持的调谐器类型



目前支持的适配器类型是“可注入”适配器，这意味着对模型进行就地修改足以正确执行微调的适配器。如此一来，仅
 [LoRA](./conceptual_guides/lora)
 、AdaLoRA 和
 [IA3](./conceptual_guides/ia3)
 目前此 API 支持。


## 注入\_adapter\_in\_model方法



要执行适配器注入，只需使用
 `在模型中注入适配器`
 该方法采用 3 个参数：PEFT 配置和模型本身以及可选的适配器名称。如果多次调用，您还可以在模型中附加多个适配器
 `在模型中注入适配器`
 具有不同的适配器名称。


下面是如何将 LoRA 适配器注入子模块的基本示例用法
 `线性`
 模块的
 `虚拟模型`
 。



```
import torch
from peft import inject_adapter_in_model, LoraConfig


类 DummyModel(torch.nn.Module):
    def \__\_init\_\_(自身):
        超级().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.线性 = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)

    defforward(self, 输入\_ids):
        x = self.embedding(input_ids)
        x = 自线性(x)
        x = self.lm_head(x)
        返回x


lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["linear"],
)

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)

dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
dummy_outputs = model(dummy_inputs)
```


如果打印模型，您会注意到适配器已正确注入模型中



```
DummyModel(
  (embedding): Embedding(10, 10)
  (linear): Linear(
    in_features=10, out_features=10, bias=True
    (lora_dropout): ModuleDict(
      (default): Dropout(p=0.1, inplace=False)
    )
    (lora_A): ModuleDict(
      (default): Linear(in_features=10, out_features=64, bias=False)
    )
    (lora_B): ModuleDict(
      (default): Linear(in_features=64, out_features=10, bias=False)
    )
    (lora_embedding_A): ParameterDict()
    (lora_embedding_B): ParameterDict()
  )
  (lm_head): Linear(in_features=10, out_features=10, bias=True)
)
```


请注意，应该由用户来正确保存适配器（如果他们只想保存适配器），因为
 `model.state_dict()`
 将返回模型的完整状态字典。
如果您想提取适配器状态字典，您可以使用
 `get_peft_model_state_dict`
 方法：



```
from peft import get_peft_model_state_dict

peft_state_dict = get_peft_model_state_dict(model)
print(peft_state_dict)
```


## 优点和缺点



什么时候使用这个API，什么时候不使用它？让我们在本节中讨论优点和缺点


优点：


* 模型就地修改，这意味着模型将保留其所有原始属性和方法
* 适用于任何torch模块和任何模式（视觉、文本、多模式）


缺点：


* 需要手动编写Huging Face
 `from_pretrained`
 和
 `保存预训练`
 如果您想轻松地从 Hugging Face Hub 保存/加载适配器，请使用实用方法。
* 您不能使用任何提供的实用方法
 `佩夫特模型`
 例如禁用适配器、合并适配器等。