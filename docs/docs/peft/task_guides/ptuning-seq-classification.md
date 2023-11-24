# 用于序列分类的 P-tuning

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/ptuning-seq-classification>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/ptuning-seq-classification>


为下游任务微调大型语言模型具有挑战性，因为它们有太多参数。要解决此问题，您可以使用
 *提示*
 在不完全微调模型的情况下将模型引导至特定的下游任务。通常，这些提示是手工制作的，这可能不切实际，因为您需要非常大的验证集才能找到最佳提示。
 *P调*
 是一种在连续空间中自动搜索和优化更好提示的方法。


💡阅读
 [GPT 也能理解](https://arxiv.org/abs/2103.10385)
 了解有关 p 调整的更多信息。


本指南将向您展示如何训练
 [`roberta-large`](https://huggingface.co/roberta-large)
 模型（但您也可以使用任何 GPT、OPT 或 BLOOM 模型），并在
 `mrpc`
 配置的
 [胶水](https://huggingface.co/datasets/glue)
 基准。


在开始之前，请确保已安装所有必需的库：



```
!pip install -q peft transformers datasets evaluate
```


## 设置



首先，导入 🤗 Transformers 以创建基本模型，🤗 Datasets 来加载数据集，🤗 Evaluate 来加载评估指标，🤗 PEFT 来创建
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 并设置 p-tuning 的配置。


定义模型、数据集和一些基本训练超参数：



```
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)
from datasets import load_dataset
import evaluate
import torch

model_name_or_path = "roberta-large"
task = "mrpc"
num_epochs = 20
lr = 1e-3
batch_size = 32
```


## 加载数据集和指标



接下来，加载
 `mrpc`
 配置 - 根据语义是否等价标记的句子对语料库 - 来自
 [胶水](https://huggingface.co/datasets/glue)
 基准：



```
dataset = load_dataset("glue", task)
dataset["train"][0]
{
    "sentence1": 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    "sentence2": 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
    "label": 1,
    "idx": 0,
}
```


从🤗评估中，加载用于评估模型性能的指标。评估模块返回与该特定任务相关的准确性和 F1 分数。



```
metric = evaluate.load("glue", task)
```


现在您可以使用
 `公制`
 编写一个计算准确度和 F1 分数的函数。这
 `计算指标`
 函数根据模型预测和标签计算分数：



```
import numpy as np


def compute\_metrics(eval\_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
```


## 预处理数据集



初始化标记生成器并配置要使用的填充标记。如果您使用 GPT、OPT 或 BLOOM 模型，则应该设置
 `填充侧`
 向左转;否则它将被设置在右侧。对句子对进行分词并将其截断至最大长度。



```
if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad\_token\_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize\_function(examples):
    # max\_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    return outputs
```


使用
 `地图`
 来应用
 `标记化函数`
 到数据集，并删除未处理的列，因为模型不需要这些列。您还应该重命名
 `标签`
 列至
 `标签`
 因为这是 🤗 Transformers 库中模型标签的预期名称。



```
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```


创建一个整理器函数
 [DataCollat​​orWithPadding](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/data_collat​​or#transformers.DataCollat​​orWithPadding)
 将批次中的示例填充到
 ‘最长’
 批次中的顺序：



```
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
```


## 火车



P-tuning 使用提示编码器来优化提示参数，因此您需要初始化
 [PromptEncoderConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEncoderConfig)
 有几个参数：


* `任务类型`
 ：您正在训练的任务类型，在本例中是序列分类或
 `SEQ_CLS`
* `num_virtual_tokens`
 ：要使用的虚拟令牌的数量，或者换句话说，提示
* `编码器隐藏大小`
 ：用于优化提示参数的编码器的隐藏大小



```
peft_config = PromptEncoderConfig(task_type="SEQ\_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
```


创建基地
 “罗伯塔·大号”
 模型来自
 [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/auto#transformers.AutoModelForSequenceClassification)
 ，然后包装基础模型并
 `peft_config`
 和
 `get_peft_model()`
 创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。如果您想知道与所有模型参数的训练相比，您实际训练了多少参数，您可以使用以下命令打印出来
 [print\_trainable\_parameters()](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel.print_trainable_parameters)
 :



```
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 1351938 || all params: 355662082 || trainable%: 0.38011867680626127"
```


从 🤗 Transformers 库中，设置
 [TrainingArguments](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments)
 包含要将模型保存到的位置、训练超参数、如何评估模型以及何时保存检查点的类：



```
training_args = TrainingArguments(
    output_dir="your-name/roberta-large-peft-p-tuning",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```


然后通过模型，
 《训练论据》
 、数据集、分词器、数据整理器和评估函数
 [Trainer](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer)
 课程，它将为您处理整个培训循环。准备好后，请致电
 [火车](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer.train)
 开始训练！



```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```


## 分享模型



如果您愿意，您可以在 Hub 上存储和共享您的模型。登录您的 Hugging Face 帐户并在出现提示时输入您的令牌：



```
from huggingface_hub import notebook_login

notebook_login()
```


使用以下命令将模型上传到 Hub 上的特定模型存储库
 [push\_to\_hub](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)
 功能：



```
model.push_to_hub("your-name/roberta-large-peft-p-tuning", use_auth_token=True)
```


## 推理



模型上传到 Hub 后，任何人都可以轻松使用它进行推理。加载配置和模型：



```
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

peft_model_id = "smangrul/roberta-large-peft-p-tuning"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, peft_model_id)
```


获取一些文本并将其标记化：



```
classes = ["not equivalent", "equivalent"]

sentence1 = "Coast redwood trees are the tallest trees on the planet and can grow over 300 feet tall."
sentence2 = "The coast redwood trees, which can attain a height of over 300 feet, are the tallest trees on earth."

inputs = tokenizer(sentence1, sentence2, truncation=True, padding="longest", return_tensors="pt")
```


将输入传递给模型以对句子进行分类：



```
with torch.no_grad():
    outputs = model(**inputs).logits
    print(outputs)

paraphrased_text = torch.softmax(outputs, dim=1).tolist()[0]
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrased\_text[i] \* 100))}%")
"not equivalent: 4%"
"equivalent: 96%"
```