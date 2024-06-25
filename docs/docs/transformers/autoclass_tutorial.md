# 使用AutoClass加载预训练实例

> 译者：[BrightLi](https://github.com/brightli)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/autoclass_tutorial>
>
> 原始地址：<https://huggingface.co/docs/transformers/autoclass_tutorial>

鉴于存在众多不同的Transformer架构，为您的检查点创建一个可能颇具挑战。作为 🤗 Transformers 核心理念的一部分，旨在使库易于使用、简单且灵活，AutoClass 能够自动推断并从给定检查点加载正确的架构。from_pretrained() 方法允许你快速加载任何架构的预训练模型，因此你不必花费时间和资源从头开始训练模型。生成这种与检查点无关的代码意味着，如果你的代码适用于一个检查点，即使架构不同，它也将适用于另一个检查点——只要该检查点是针对类似任务训练的。

> 请记住，架构是指模型的骨架，而检查点是给定架构的权重。例如，[BERT](https://huggingface.co/google-bert/bert-base-uncased) 是一个架构，而 google-bert/bert-base-uncased 是一个检查点。模型是一个通用术语，可以指代架构或检查点。

在本教程中，学习如何：
* 加载预训练的分词器。
* 加载预训练的图像处理器。
* 加载预训练的特征提取器。
* 加载预训练的处理器。
* 加载预训练的模型。
* 将模型作为主干网络加载。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

然后按照下面的方式对你的输入进行分词：

```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

**AutoImageProcessor**

对于视觉任务，图像处理器将图像处理为正确的输入格式。

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

**AutoBackbone**

<img src='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stages.png' width=100% />

一个具有多个阶段用于输出特征图的Swin骨架网络。

[AutoBackbone](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/backbones#transformers.AutoBackbone)允许您使用预训练模型作为骨架网络，以从骨架的不同阶段获取特征图。您应该在[from_pretrained()](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)中指定以下参数之一：

* out_indices 是您想要获取特征图的层的索引
* out_features 是您想要获取特征图的层的名称

这些参数可以互换使用，但如果您同时使用两者，请确保它们彼此一致！如果您不传递这些参数中的任何一个，骨架将返回最后一层的特征图。
在机器学习或深度学习上下文中，这段文字描述的是如何使用一个预训练的模型（通常是一个神经网络的一部分，负责提取低级到中级的特征）来作为新模型的基础。通过调整out_indices或out_features参数，用户能够选择骨架网络中不同层输出的特征图。如果两个参数都未指定，模型默认输出最后一个层的特征图。这对于需要在模型的不同层次上进行特征提取的任务非常有用，如物体检测或语义分割等。

<img src='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stage%201.png' width=100% />

特征图来自骨架的第一个阶段。补丁分区指的是模型的主干。

例如，在上图所示的图中，要从Swin骨架的第一级返回特征图，您可以设置 out_indices=(1,)：

```py
>>> from transformers import AutoImageProcessor, AutoBackbone
>>> import torch
>>> from PIL import Image
>>> import requests
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = AutoBackbone.from_pretrained("microsoft/>>> swin-tiny-patch4-window7-224", out_indices=(1,))

>>> inputs = processor(image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> feature_maps = outputs.feature_maps
```

现在，您可以从骨架的第一级访问 feature_maps 对象：
复制

```py
>>> list(feature_maps[0].shape)
[1, 96, 56, 56]
```

**AutoFeatureExtractor**

对于音频任务，特征提取器以正确的输入格式处理音频信号。
使用 [AutoFeatureExtractor.from_pretrained()](https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoFeatureExtractor.from_pretrained) 加载特征提取器：
复制

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
"ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
```

**AutoProcessor**

多模态任务需要一个处理器，该处理器结合了两种类型的预处理工具。例如，[LayoutLMV2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)模型需要一个图像处理器来处理图像和一个分词器来处理文本；一个处理器将两者结合起来。

使用[AutoProcessor.from_pretrained()](https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)加载处理器：

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

**AutoModel**

AutoModelFor 类允许您为给定任务加载预训练模型（[此处](https://huggingface.co/docs/transformers/model_doc/auto)查看可用任务的完整列表）。例如，使用 [AutoModelForSequenceClassification.from_pretrained()](https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel.from_pretrained) 加载一个用于序列分类的模型：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

轻松重用相同的检查点来加载不同任务的架构：

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

> 对于 PyTorch 模型，from_pretrained() 方法使用 torch.load()，后者在内部使用 pickle，已知这是不安全的。通常，永远不要加载可能来自不受信任的来源的模型，或可能已被篡改的模型。对于托管在 Hugging Face Hub 上的公共模型，这种安全风险在一定程度上得到了缓解，因为每次提交都会扫描恶意软件。有关最佳实践，如使用 GPG 进行签名提交验证，请参阅 Hub 文档。
TensorFlow 和 Flax 检查点不受影响，并且可以使用 from_pretrained 方法的 from_tf 和 from_flax kwargs 在 PyTorch 架构内加载，以绕过此问题。

通常，我们建议使用 AutoTokenizer 类和 AutoModelFor 类来加载模型的预训练实例。这将确保您每次加载正确的架构。在下一个[教程](https://huggingface.co/docs/transformers/preprocessing)中，学习如何使用您新加载的分词器、图像处理器、特征提取器和处理器来预处理数据集以进行微调。

TensorFlow

最后，TFAutoModelFor 类允许您为给定任务加载预训练模型（[此处](https://huggingface.co/docs/transformers/model_doc/auto)查看可用任务的完整列表）。例如，使用 [TFAutoModelForSequenceClassification.from_pretrained()](https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/auto#transformers.AutoModel.from_pretrained) 加载一个用于序列分类的模型：

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

轻松地重用相同的检查点来加载不同任务的架构：

```py
from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

通常，我们建议使用AutoTokenizer类和TFAutoModelFor类来加载模型的预训练实例。这将确保您每次都能加载到正确的架构。在下一个[教程](https://huggingface.co/docs/transformers/preprocessing)中，学习如何使用新加载的分词器、图像处理器、特征提取器和处理器对数据集进行微调预处理。

