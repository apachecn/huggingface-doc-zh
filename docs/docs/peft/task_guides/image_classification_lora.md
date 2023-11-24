# 使用 LoRA 进行图像分类

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/image_classification_lora>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/image_classification_lora>


本指南演示如何使用 LoRA（一种低秩近似技术）来微调图像分类模型。
通过使用🤗 PEFT 的 LoRA，我们可以将模型中可训练参数的数量减少到只有原始参数的 0.77%。


LoRA 通过向模型的特定块（例如注意力）添加低秩“更新矩阵”来实现这种减少
块。微调时，仅训练这些矩阵，而原始模型参数保持不变。
在推理时，更新矩阵与原始模型参数合并以产生最终的分类结果。


有关 LoRA 的更多信息，请参阅
 [LoRA 原始论文](https://arxiv.org/abs/2106.09685)
 。


## 安装依赖项



安装模型训练所需的库：



```
!pip install transformers accelerate evaluate datasets peft -q
```


检查所有必需库的版本以确保您是最新的：



```
import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.\_\_version\_\_}")
print(f"Accelerate version: {accelerate.\_\_version\_\_}")
print(f"PEFT version: {peft.\_\_version\_\_}")
"Transformers version: 4.27.4"
"Accelerate version: 0.18.0"
"PEFT version: 0.2.0"
```


## 验证以共享您的模型



要在培训结束时与社区分享微调后的模型，请使用您的 🤗 令牌进行身份验证。
您可以从您的
 [账户设置](https://huggingface.co/settings/token)
 。



```
from huggingface_hub import notebook_login

notebook_login()
```


## 选择模型检查点进行微调



从支持的任何模型架构中选择一个模型检查点
 [图像分类](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads)
 。如有疑问，请参阅
这
 [图像分类任务指南](https://huggingface.co/docs/transformers/v4.27.2/en/tasks/image_classification)
 在
🤗 变形金刚文档。



```
model_checkpoint = "google/vit-base-patch16-224-in21k"
```


## 加载数据集



为了缩短此示例的运行时间，我们只加载训练集中的前 5000 个实例
 [Food-101 数据集](https://huggingface.co/datasets/food101)
 :



```
from datasets import load_dataset

dataset = load_dataset("food101", split="train[:5000]")
```


## 数据集准备



要准备用于训练和评估的数据集，请创建
 `标签2id`
 和
 `id2标签`
 字典。这些都会进来
在执行推理和元数据信息时很方便：



```
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]
"baklava"
```


接下来，加载您正在微调的模型的图像处理器：



```
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
```


这
 `图像处理器`
 包含有关应调整训练和评估图像大小的有用信息
以及用于标准化像素值的值。使用
 `图像处理器`
 , 准备转换
数据集的函数。这些功能将包括数据增强和像素缩放：



```
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


def 预处理\_train（示例\_batch）：
    """跨批次应用训练\_变换。"""
    example_batch["pixel\_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    返回 example_batch


def preprocess\_val(example\_batch):
    """Apply val\_transforms across a batch."""
    example_batch["pixel\_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
```


将数据集分为训练集和验证集：



```
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]
```


最后，相应地设置数据集的转换函数：



```
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```


## 加载并准备模型



在加载模型之前，我们定义一个辅助函数来检查模型的参数总数
其中有多少是可训练的。



```
def print\_trainable\_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable\_params} || all params: {all\_param} || trainable%: {100 \* trainable\_params / all\_param:.2f}"
    )
```


正确初始化原始模型非常重要，因为它将用作创建模型的基础
 `佩夫特模型`
 你会
实际上微调。指定
 `标签2id`
 和
 `id2标签`
 以便
 [AutoModelForImageClassification](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/auto#transformers.AutoModelForImageClassification)
 可以附加一个分类
前往适合该数据集的底层模型。您应该看到以下输出：



```
Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.weight', 'classifier.bias']
```



```
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
```


在创建之前
 `佩夫特模型`
 ，您可以检查原始模型中可训练参数的数量：



```
print_trainable_parameters(model)
"trainable params: 85876325 || all params: 85876325 || trainable%: 100.00"
```


接下来，使用
 `get_peft_model`
 包装基本模型，以便将“更新”矩阵添加到相应的位置。



```
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)
"trainable params: 667493 || all params: 86466149 || trainable%: 0.77"
```


让我们来解开这里发生的事情。
要使用LoRA，您需要在中指定目标模块
 `LoraConfig`
 以便
 `get_peft_model()`
 知道哪些模块
我们的模型内部需要用 LoRA 矩阵进行修改。在此示例中，我们只对定位查询感兴趣，并且
基础模型的注意力块的值矩阵。由于这些矩阵对应的参数是“命名”的
分别是“query”和“value”，我们在
 `目标模块`
 的论证
 `LoraConfig`
 。


我们还指定
 `要保存的模块`
 。包装基本模型后
 `get_peft_model()`
 随着
 `配置`
 ，我们得到
一种新模型，其中仅 LoRA 参数可训练（所谓的“更新矩阵”），而预训练参数
被冷冻保存。然而，我们希望在微调基础模型时也训练分类器参数
自定义数据集。为了确保分类器参数也经过训练，我们指定
 `要保存的模块`
 。这也是
确保在使用类似实用程序时，这些模块与 LoRA 可训练参数一起序列化
 `save_pretrained()`
 和
 `push_to_hub()`
 。


其他参数的含义如下：


* `r`
 ：LoRA 更新矩阵使用的维度。
* `阿尔法`
 ：比例因子。
*`偏见`
 ：指定是否
 `偏见`
 应该训练参数。
 `无`
 表示没有一个
 `偏见`
 参数将被训练。


`r`
 和
 `阿尔法`
 使用 LoRA 时共同控制最终可训练参数的总数，为您提供灵活性
平衡最终性能和计算效率之间的权衡。


通过查看可训练参数的数量，您可以了解我们实际正在训练的参数数量。既然目标是
为了实现参数高效的微调，您应该期望在
 `劳拉模型`
 与原始模型相比，这里确实是这样。


## 定义训练参数



对于模型微调，使用
 [Trainer](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer)
 。它接受
您可以使用几个参数来包装
 [TrainingArguments](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments)
 。



```
from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    f"{model\_name}-finetuned-lora-food101",
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
    metric_for_best_model="accuracy",
    push_to_hub=True,
    label_names=["labels"],
)
```


与非 PEFT 方法相比，您可以使用更大的批量大小，因为需要训练的参数更少。
您还可以设置比正常情况更大的学习率（例如 1e-5）。


这还可能减少进行昂贵的超参数调整实验的需要。


## 准备评估指标




```
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute\_metrics(eval\_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```


这
 `计算指标`
 函数采用命名元组作为输入：
 `预测`
 ，它们是 Numpy 数组形式的模型的 logits，
和
 `label_ids`
 ，它们是 Numpy 数组的真实标签。


## 定义排序规则函数



使用排序函数
 [Trainer](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer)
 收集一批培训和评估示例并准备好
底层模型可接受的格式。



```
import torch


def collate\_fn(examples):
    pixel_values = torch.stack([example["pixel\_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel\_values": pixel_values, "labels": labels}
```


## 训练和评估



将所有内容放在一起 - 模型、训练参数、数据、整理函数等。然后，开始训练！



```
trainer = Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()
```


在短短几分钟内，经过微调的模型即使在这么小的数据上也显示出 96% 的验证准确率
训练数据集的子集。



```
trainer.evaluate(val_ds)
{
    "eval\_loss": 0.14475855231285095,
    "eval\_accuracy": 0.96,
    "eval\_runtime": 3.5725,
    "eval\_samples\_per\_second": 139.958,
    "eval\_steps\_per\_second": 1.12,
    "epoch": 5.0,
}
```


## 分享您的模型并运行推理



微调完成后，与社区共享 LoRA 参数，如下所示：



```
repo_name = f"sayakpaul/{model\_name}-finetuned-lora-food101"
lora_model.push_to_hub(repo_name)
```


打电话时
 [push\_to\_hub](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)
 于
 `劳拉模型`
 ，仅 LoRA 参数以及中指定的任何模块
 `要保存的模块`
 被保存。看看
 [训练好的LoRA参数](https://huggingface.co/sayakpaul/vit-base-patch16-224-in21k-finetuned-lora-food101/blob/main/adapter_model.bin)
 。
您会发现它只有 2.6 MB！这极大地有助于可移植性，特别是在使用非常大的模型进行微调时（例如
 [绽放](https://huggingface.co/bigscience/bloom)
 ）。


接下来，让我们看看如何加载 LoRA 更新参数以及我们的推理基础模型。当您包装基础模型时
和
 `佩夫特模型`
 ，修改完成
 *到位*
 。为了减轻就地修改可能引起的任何担忧，
像之前一样初始化基本模型并构建推理模型。



```
from peft import PeftConfig, PeftModel


config = PeftConfig.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
# Load the LoRA model
inference_model = PeftModel.from_pretrained(model, repo_name)
```


现在让我们获取一个示例图像进行推理。



```
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image
```


![贝奈特饼图像](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg)


 首先，实例化一个
 `图像处理器`
 来自底层模型存储库。



```
image_processor = AutoImageProcessor.from_pretrained(repo_name)
```


然后，准备用于推理的示例。



```
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
```


最后，运行推理！



```
with torch.no_grad():
    outputs = inference_model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", inference_model.config.id2label[predicted_class_idx])
"Predicted class: beignets"
```