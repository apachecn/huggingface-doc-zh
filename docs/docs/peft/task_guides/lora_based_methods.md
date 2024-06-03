# LoRA 方法

> 译者：[糖醋鱼](https://github.com/now-101)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/lora_based_methods>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/lora_based_methods>

# LoRA 方法
一种流行的高效训练大型模型的方法是（通常在注意力块中）插入较小的可训练矩阵，这些矩阵是对微调期间要学习的权重增量矩阵的低秩分解。预训练模型原有的权重矩阵被冻结，仅更新这些较小的矩阵。这样做减少了可训练参数的数量，从而降低了内存使用量和训练时间，这对于大型模型而言可以大幅节省成本。  

有多种方法可以将权重矩阵表示为低秩分解，但[低秩适应（Low-Rank Adaptation，LoRA）](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora)是最常见的方法。PEFT库支持几种其他的LoRA变体，例如低秩[Hadamard积（Low-Rank Hadamard Product，LoHa）](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-hadamard-product-loha)、[低秩Kronecker积（Low-Rank Kronecker Product，LoKr）](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-kronecker-product-lokr)和[自适应低秩适应（Adaptive Low-Rank Adaptation，AdaLoRA）](https://huggingface.co/docs/peft/conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora)。您可以在[适配器](https://huggingface.co/docs/peft/conceptual_guides/adapter)指南中了解这些方法的概念。如果您有兴趣将这些方法应用于其他任务和用例，如语义分割、标记分类，请查看我们的[笔记本集合](https://huggingface.co/collections/PEFT/notebooks-6573b28b33e5a4bf5b157fc1)！  

本指南将向您展示如何运用低秩分解方法，快速训练一个图像分类模型，以识别图像中所示食物的类别。    

> 对训练图像分类模型的一般流程有所了解将非常有帮助，这样你可以更专注于低秩分解方法。如果你是新手，我们建议你先从Transformers文档中的图像分类指南开始学习。当你准备好了，回过头来看看将PEFT融入到你的训练流程中是多么简单！


在开始之前，请确保已经安装了所有必需的库。
```py
pip install -q peft transformers datasets
```

# 数据集
在本指南中，你将使用[Food-101](https://huggingface.co/datasets/food101)数据集，其中包含101类食物的图像（可以通过[数据集查看器](https://huggingface.co/datasets/food101/viewer/default/train)来更好地了解数据集的样貌）。

使用load_dataset函数加载数据集。
```py
from datasets import load_dataset

ds = load_dataset("food101")
```
每个食物类别都被赋予了一个整数标签，为了更直观地理解这些整数代表的意义，你将创建一个label2id和id2label字典，用于将整数映射到其对应的类别标签。
```py
labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]
"baklava"
```
加载一个图像处理器，以便正确调整训练和评估图像的像素值大小并对其进行归一化。  
```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
```
你还可以使用图像处理器来准备一些变换函数，用于数据增强和像素缩放。  
```py
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
```
定义训练和验证数据集，并使用 `set_transform` 函数来实时应用这些变换。
```py
train_ds = ds["train"]
val_ds = ds["validation"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```

```py
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```
# 模型
现在让我们加载一个预训练模型作为基模型。本指南使用的是
[google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)模型，但你可以选择任何你想要的图像分类模型。将 `label2id` 和 `id2label` 字典传递给模型，以便它知道如何将整数标签映射到它们对应的类别标签。如果你正在微调一个已经被微调过的checkpoint，可以选择性地传递 `ignore_mismatched_sizes=True` 参数。
```py
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
```
# PEFT 配置 and 模型

每种PEFT方法都需要一个配置，用于保存指定PEFT方法应用方式的所有参数。一旦配置设置完毕，将其与基模型一起传递给 [get_peft_model()](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/peft_model#peft.get_peft_model) 函数，以创建一个可训练的[PeftModel](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/peft_model#peft.PeftModel)。  

> 调用 [print_trainable_parameters()](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/peft_model#peft.PeftModel.print_trainable_parameters) 方法来对比PeftModel与基模型中的参数数量

## LoRA
[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora)将权重更新矩阵分解为两个更小的矩阵。这些低秩矩阵的大小由其秩或r决定。更高的秩意味着模型有更多的参数需要训练，但也意味着模型具有更强的学习能力。你还需要指定`target_modules`，这决定了较小的矩阵插入的位置。在这个指南中，我们将目标定位在注意力模块的 `query` 和 `value` 矩阵上。其他需要设定的重要参数包括 `lora_alpha`（缩放因子）、`bias`（是否训练无偏置、所有偏置或仅LoRA偏置参数）以及 `modules_to_save`（除LoRA层外，需要训练和保存的模块）。所有这些参数，以及更多，都可以在 [LoraConfig](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/lora#peft.LoraConfig)中找到。
```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 667,493 || all params: 86,543,818 || trainable%: 0.7712775047664294"
```

## LoHa
[LoHa](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-hadamard-product-loha)将权重更新矩阵分解为四个更小的矩阵，每对较小的矩阵通过Hadamard乘积结合。这使得与LoRA相比，权重更新矩阵在保持相同数量的可训练参数的同时，具有更高的秩（LoHA为r^2，而LoRA为2*r）。这些较小矩阵的大小由其秩或r决定。同样，你需要指定 `target_modules` ，以确定这些较小矩阵的插入位置。在这个指南中，我们将目标定位在注意力模块的 `query` 和 `value` 矩阵上。其他重要的参数包括alpha（缩放因子）以及modules_to_save（除LoHa层之外，需要训练和保存的模块）。所有这些参数，以及其他更多选项，都可以在[LoHaConfig](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/loha#peft.LoHaConfig)中找到。
```py
from peft import LoHaConfig, get_peft_model

config = LoHaConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 1,257,317 || all params: 87,133,642 || trainable%: 1.4429753779831676"
```
## LoKr
[LoKr](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-kronecker-product-lokr)将权重更新矩阵表示为Kronecker乘积的分解，形成一个能够保持原权重矩阵秩的块矩阵。这些较小矩阵的大小由其秩或r决定。同样，你需要指定 `target_modules` ，以确定这些较小矩阵的插入位置。在这个指南中，我们将目标定位在注意力模块的 `query` 和 `value` 矩阵上。其他重要的参数包括alpha（缩放因子），以及 `modules_to_save`（除LoKr层之外，需要训练和保存的模块）。所有这些参数，以及其他更多选项，都可以在 [LoKrConfig](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/lokr#peft.LoKrConfig) 中找到。

```py
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 116,069 || all params: 87,172,042 || trainable%: 0.13314934162033282"
```
## AdaLoRA
[AdaLoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora)
通过为重要的权重矩阵分配更多参数并剪枝较不重要的矩阵，从而有效地管理LoRA参数预算。相比之下，LoRA在所有模块中均匀分配参数。你可以控制矩阵的平均期望秩或r，以及通过 `target_modules` 参数来指定将AdaLoRA应用于哪些模块。其他需要设置的重要参数包括lora_alpha（缩放因子），以及 `modules_to_save`（除AdaLoRA层之外，需要训练和保存的模块）。所有这些参数，以及其他更多选项，都可以在
 [AdaLoraConfig](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/adalora#peft.AdaLoraConfig) 中找到。
```py
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 520,325 || all params: 87,614,722 || trainable%: 0.5938785036606062"
```


# 训练
对于训练，我们使用来自Transformers库的 [Trainer](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/trainer#transformers.Trainer)
 类。Trainer类包含了PyTorch的训练循环，当一切准备就绪，只需调用 [train](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/trainer#transformers.Trainer.train) 方法即可开始训练。为了定制训练过程，你可以在 [TrainingArguments](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/trainer#transformers.TrainingArguments) 类中配置训练超参数。使用类似LoRA的方法，你可以负担得起使用更高的批次大小和学习率。

>> AdaLoRA有一个[update_and_allocate()](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/adalora#peft.AdaLoraModel.update_and_allocate)方法，需要在每个训练步骤中调用，以更新参数预算和掩码，否则不会执行适应步骤。这需要编写自定义训练循环或继承Trainer类来整合这个方法。作为一个例子，可以参考这个自定义训练循环[ custom training loop](https://github.com/huggingface/peft/blob/912ad41e96e03652cabf47522cd876076f7a0c4f/examples/conditional_generation/peft_adalora_seq2seq.py#L120)。


```py
from transformers import TrainingArguments, Trainer

account = "stevhliu"
peft_model_id = f"{account}/google/vit-base-patch16-224-in21k-lora"
batch_size = 128

args = TrainingArguments(
    peft_model_id,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    label_names=["labels"],
)
```
开始训练，调用train方法。
```py
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    data_collator=collate_fn,
)
trainer.train()
```

# 分享你的模型
完成训练后，你可以使用 [push_to_hub](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)方法将你的模型上传到模型中心。首先，你需要登录到你的Hugging Face账户，并在提示时输入你的访问令牌。
```py
from huggingface_hub import notebook_login

notebook_login()
```
调用 `push_to_hub` 方法将你的模型保存到你的仓库中。
```py
model.push_to_hub(peft_model_id)
```

# 推理
让我们从模型中心加载模型，并在一个食物图像上测试它的效果。
```py
from peft import PeftConfig, PeftModel
from transfomers import AutoImageProcessor
from PIL import Image
import requests

config = PeftConfig.from_pretrained("stevhliu/vit-base-patch16-224-in21k-lora")
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(model, "stevhliu/vit-base-patch16-224-in21k-lora")

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image
```
![](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg  "beignets")

将图像转换为RGB格式，并返回底层的PyTorch张量。
```
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
```
现在运行模型，返回预测的类别！
```py
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
"Predicted class: beignets"
```
