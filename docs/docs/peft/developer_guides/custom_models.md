# 使用自定义模型

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/developer_guides/custom_models>
>
> 原始地址：<https://huggingface.co/docs/peft/developer_guides/custom_models>


一些微调技术（例如提示调整）特定于语言模型。这意味着在 🤗 PEFT 中，它是
假设正在使用 🤗 Transformers 模型。然而，其他微调技术 - 例如
 [LoRA](./conceptual_guides/lora)
 - 不限于特定型号。


在本指南中，我们将了解 LoRA 如何应用于多层感知器和计算机视觉模型
 [蒂姆](https://huggingface.co/docs/timm/index)
 图书馆。


## 多层感知器



假设我们想要使用 LoRA 微调多层感知器。这是定义：



```
from torch import nn


class MLP(nn.Module):
    def \_\_init\_\_(self, num\_units\_hidden=2000):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(20, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        return self.seq(X)
```


这是一个简单的多层感知器，具有输入层、隐藏层和输出层。


对于这个玩具示例，我们选择了大量的隐藏单元来突出效率增益
来自 PEFT，但这些收益与更现实的例子相符。


该模型中有一些线性层可以使用 LoRA 进行调整。使用常见的 🤗 变形金刚时
模型中，PEFT 会知道将 LoRA 应用到哪些层，但在这种情况下，由我们作为用户来选择层。
要确定要调整的层的名称：



```
print([(n, type(m)) for n, m in MLP().named_modules()])
```


这应该打印：



```
[('', __main__.MLP),
 ('seq', torch.nn.modules.container.Sequential),
 ('seq.0', torch.nn.modules.linear.Linear),
 ('seq.1', torch.nn.modules.activation.ReLU),
 ('seq.2', torch.nn.modules.linear.Linear),
 ('seq.3', torch.nn.modules.activation.ReLU),
 ('seq.4', torch.nn.modules.linear.Linear),
 ('seq.5', torch.nn.modules.activation.LogSoftmax)]
```


假设我们要将 LoRA 应用于输入层和隐藏层，这些是
 `'seq.0'`
 和
 `'seq.2'`
 。而且，
假设我们想要在没有 LoRA 的情况下更新输出层，那就是
 `'seq.4'`
 。相应的配置将
是：



```
from peft import LoraConfig

config = LoraConfig(
    target_modules=["seq.0", "seq.2"],
    modules_to_save=["seq.4"],
)
```


这样，我们就可以创建 PEFT 模型并检查训练参数的比例：



```
from peft import get_peft_model

model = MLP()
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# prints trainable params: 56,164 || all params: 4,100,164 || trainable%: 1.369798866581922
```


最后，我们可以使用任何我们喜欢的训练框架，或者编写我们自己的拟合循环来训练
 `peft_model`
 。


有关完整的示例，请查看
 [本笔记本](https://github.com/huggingface/peft/blob/main/examples/multilayer_perceptron/multilayer_perceptron_lora.ipynb)
 。


## 蒂姆模型



这
 [蒂姆](https://huggingface.co/docs/timm/index)
 库包含大量预训练的计算机视觉模型。
这些也可以通过 PEFT 进行微调。让我们看看这在实践中是如何运作的。


首先，确保 Python 环境中安装了 timm：



```
python -m pip install -U timm
```


接下来我们为图像分类任务加载 timm 模型：



```
import timm

num_classes = ...
model_id = "timm/poolformer\_m36.sail\_in1k"
model = timm.create_model(model_id, pretrained=True, num_classes=num_classes)
```


同样，我们需要决定将 LoRA 应用到哪些层。由于 LoRA 支持 2D 转换层，并且由于
这些是该模型的主要构建块，我们应该将 LoRA 应用于 2D 转换层。来识别姓名
这些层，让我们看看所有层的名称：



```
print([(n, type(m)) for n, m in MLP().named_modules()])
```


这将打印一个很长的列表，我们只显示前几个：



```
[('', timm.models.metaformer.MetaFormer),
 ('stem', timm.models.metaformer.Stem),
 ('stem.conv', torch.nn.modules.conv.Conv2d),
 ('stem.norm', torch.nn.modules.linear.Identity),
 ('stages', torch.nn.modules.container.Sequential),
 ('stages.0', timm.models.metaformer.MetaFormerStage),
 ('stages.0.downsample', torch.nn.modules.linear.Identity),
 ('stages.0.blocks', torch.nn.modules.container.Sequential),
 ('stages.0.blocks.0', timm.models.metaformer.MetaFormerBlock),
 ('stages.0.blocks.0.norm1', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.0.token\_mixer', timm.models.metaformer.Pooling),
 ('stages.0.blocks.0.token\_mixer.pool', torch.nn.modules.pooling.AvgPool2d),
 ('stages.0.blocks.0.drop\_path1', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.layer\_scale1', timm.models.metaformer.Scale),
 ('stages.0.blocks.0.res\_scale1', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.norm2', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.0.mlp', timm.layers.mlp.Mlp),
 ('stages.0.blocks.0.mlp.fc1', torch.nn.modules.conv.Conv2d),
 ('stages.0.blocks.0.mlp.act', torch.nn.modules.activation.GELU),
 ('stages.0.blocks.0.mlp.drop1', torch.nn.modules.dropout.Dropout),
 ('stages.0.blocks.0.mlp.norm', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.mlp.fc2', torch.nn.modules.conv.Conv2d),
 ('stages.0.blocks.0.mlp.drop2', torch.nn.modules.dropout.Dropout),
 ('stages.0.blocks.0.drop\_path2', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.layer\_scale2', timm.models.metaformer.Scale),
 ('stages.0.blocks.0.res\_scale2', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.1', timm.models.metaformer.MetaFormerBlock),
 ('stages.0.blocks.1.norm1', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.1.token\_mixer', timm.models.metaformer.Pooling),
 ('stages.0.blocks.1.token\_mixer.pool', torch.nn.modules.pooling.AvgPool2d),
 ...
 ('head.global\_pool.flatten', torch.nn.modules.linear.Identity),
 ('head.norm', timm.layers.norm.LayerNorm2d),
 ('head.flatten', torch.nn.modules.flatten.Flatten),
 ('head.drop', torch.nn.modules.linear.Identity),
 ('head.fc', torch.nn.modules.linear.Linear)]
 ]
```


经过仔细检查，我们发现 2D 卷积层的名称如下
 `“stages.0.blocks.0.mlp.fc1”`
 和
 `“stages.0.blocks.0.mlp.fc2”`
 。我们如何具体匹配这些图层名称？你可以写一个
 [常规的
表达式](https://docs.python.org/3/library/re.html)
 以匹配图层名称。对于我们的例子，正则表达式
 `r".*\.mlp\.fc\d"`
 应该做这项工作。


此外，与第一个示例一样，我们应该确保输出层（在本例中为分类头）是
也更新了。查看上面打印的列表的末尾，我们可以看到它的名称是
 `'head.fc'`
 。考虑到这一点，
这是我们的 LoRA 配置：



```
config = LoraConfig(target_modules=r".\*\.mlp\.fc\d", modules_to_save=["head.fc"])
```


然后我们只需要将基本模型和配置传递给即可创建 PEFT 模型
 `get_peft_model`
 :



```
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# prints trainable params: 1,064,454 || all params: 56,467,974 || trainable%: 1.88505789139876
```


这表明我们只需要训练不到 2% 的参数，这是一个巨大的效率增益。


有关完整的示例，请查看
 [本笔记本](https://github.com/huggingface/peft/blob/main/examples/image_classification/image_classification_timm_peft_lora.ipynb)
 。