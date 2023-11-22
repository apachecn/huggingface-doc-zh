# LoRA 用于代币分类

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/token-classification-lora>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/token-classification-lora>


低秩适应（LoRA）是一种重参数化方法，旨在减少具有低秩表示的可训练参数的数量。权重矩阵被分解为经过训练和更新的低秩矩阵。所有预训练的模型参数保持冻结。训练后，低秩矩阵被添加回原始权重。这使得存储和训练 LoRA 模型更加高效，因为参数明显减少。


💡阅读
 [LoRA：大型语言模型的低秩适应](https://arxiv.org/abs/2106.09685)
 了解有关 LoRA 的更多信息。


本指南将向您展示如何训练
 [`roberta-large`](https://huggingface.co/roberta-large)
 带有 LoRA 的模型
 [BioNLP2004](https://huggingface.co/datasets/tner/bionlp2004)
 用于标记分类的数据集。


在开始之前，请确保已安装所有必需的库：



```
!pip install -q peft transformers datasets evaluate seqeval
```


## 设置



让我们首先导入您需要的所有必要的库：


* 🤗 用于装载底座的变压器
 “罗伯塔·大号”
 模型和分词器，以及处理训练循环
* 🤗 用于加载和准备的数据集
 `bionlp2004`
 训练数据集
* 🤗 Evaluate 用于评估模型的性能
* 🤗 PEFT 用于设置 LoRA 配置并创建 PEFT 模型



```
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np

model_checkpoint = "roberta-large"
lr = 1e-3
batch_size = 16
num_epochs = 10
```


## 加载数据集和指标



这
 [BioNLP2004](https://huggingface.co/datasets/tner/bionlp2004)
 数据集包括 DNA、RNA 和蛋白质等生物结构的标记和标签。加载数据集：



```
bionlp = load_dataset("tner/bionlp2004")
bionlp["train"][0]
{
    "tokens": [
        "Since",
        "HUVECs",
        "released",
        "superoxide",
        "anions",
        "in",
        "response",
        "to",
        "TNF",
        ",",
        "and",
        "H2O2",
        "induces",
        "VCAM-1",
        ",",
        "PDTC",
        "may",
        "act",
        "as",
        "a",
        "radical",
        "scavenger",
        ".",
    ],
    "tags": [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}
```


这
 `标签`
 值在标签 id 中定义
 [字典](https://huggingface.co/datasets/tner/bionlp2004#label-id)
 。每个标签前面的字母表示标记位置：
 'B'
 是实体的第一个令牌，
 ‘我’
 用于实体内部的令牌，并且
 `0`
 用于不属于实体的令牌。



```
{
    "O": 0,
    "B-DNA": 1,
    "I-DNA": 2,
    "B-protein": 3,
    "I-protein": 4,
    "B-cell\_type": 5,
    "I-cell\_type": 6,
    "B-cell\_line": 7,
    "I-cell\_line": 8,
    "B-RNA": 9,
    "I-RNA": 10,
}
```


然后加载
 [`seqeval`](https://huggingface.co/spaces/evaluate-metric/seqeval)
 框架，其中包括用于评估序列标记任务的多个指标（精确度、准确性、F1 和召回率）。



```
seqeval = evaluate.load("seqeval")
```


现在，您可以编写一个评估函数来计算模型预测和标签的指标，并返回精度、召回率、F1 和准确度分数：



```
label_list = [
    "O",
    "B-DNA",
    "I-DNA",
    "B-protein",
    "I-protein",
    "B-cell\_type",
    "I-cell\_type",
    "B-cell\_line",
    "I-cell\_line",
    "B-RNA",
    "I-RNA",
]


def compute\_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall\_precision"],
        "recall": results["overall\_recall"],
        "f1": results["overall\_f1"],
        "accuracy": results["overall\_accuracy"],
    }
```


## 预处理数据集



初始化分词器并确保设置
 `is_split_into_words=True`
 因为文本序列已经被分割成单词。然而，这并不意味着它已经被标记化了（尽管它可能看起来像这样！），并且您需要进一步将单词标记为子词。



```
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
```


您还需要编写一个函数来：


1. 使用以下命令将每个标记映射到各自的单词
 [word\_ids](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/tokenizer#transformers.BatchEncoding.word_ids)
 方法。
2. 忽略特殊标记，将其设置为
 `-100`
 。
3. 标记给定实体的第一个标记。



```
def tokenize\_and\_align\_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```


使用
 `地图`
 来应用
 `tokenize_and_align_labels`
 数据集的函数：



```
tokenized_bionlp = bionlp.map(tokenize_and_align_labels, batched=True)
```


最后，创建一个数据整理器将示例填充到批次中的最长长度：



```
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```


## 火车



现在您已准备好创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。首先加载底座
 “罗伯塔·大号”
 模型、预期标签的数量以及
 `id2标签`
 和
 `标签2id`
 字典：



```
id2label = {
    0: "O",
    1: "B-DNA",
    2: "I-DNA",
    3: "B-protein",
    4: "I-protein",
    5: "B-cell\_type",
    6: "I-cell\_type",
    7: "B-cell\_line",
    8: "I-cell\_line",
    9: "B-RNA",
    10: "I-RNA",
}
label2id = {
    "O": 0,
    "B-DNA": 1,
    "I-DNA": 2,
    "B-protein": 3,
    "I-protein": 4,
    "B-cell\_type": 5,
    "I-cell\_type": 6,
    "B-cell\_line": 7,
    "I-cell\_line": 8,
    "B-RNA": 9,
    "I-RNA": 10,
}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=11, id2label=id2label, label2id=label2id
)
```


定义
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 和：


* `任务类型`
 ，令牌分类（
 `任务类型.TOKEN_CLS`
 ）
* `r`
 ，低秩矩阵的维数
* `lora_alpha`
 , 权重矩阵的比例因子
* `lora_dropout`
 , LoRA 层的丢失概率
*`偏见`
 ， 设置
 `全部`
 训练所有偏置参数


💡 权重矩阵按比例缩放
 `lora_alpha/r`
 ，以及更高的
 `lora_alpha`
 value 为 LoRA 激活分配更多权重。为了性能，我们建议设置
 `偏见`
 到
 `无`
 首先，然后
 `仅劳拉`
 ，在尝试之前
 `全部`
 。



```
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)
```


通过基本模型并
 `peft_config`
 到
 `get_peft_model()`
 函数来创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。您可以查看一下训练效率有多少
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 与通过打印出可训练参数来完全训练基础模型进行比较：



```
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 1855499 || all params: 355894283 || trainable%: 0.5213624069370061"
```


从 🤗 Transformers 库中，创建一个
 [TrainingArguments](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments)
 class 并指定要将模型保存到的位置、训练超参数、如何评估模型以及何时保存检查点：



```
training_args = TrainingArguments(
    output_dir="roberta-large-lora-token-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```


通过模型，
 《训练论据》
 、数据集、分词器、数据整理器和评估函数
 [培训师](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer)
 班级。这
 「教练」
 为您处理训练循环，当您准备好时，请致电
 [火车](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer.train)
 开始！



```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bionlp["train"],
    eval_dataset=tokenized_bionlp["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```


## 分享模型



训练完成后，您可以根据需要在 Hub 上存储和共享您的模型。登录您的 Hugging Face 帐户并在出现提示时输入您的令牌：



```
from huggingface_hub import notebook_login

notebook_login()
```


使用以下命令将模型上传到 Hub 上的特定模型存储库
 [push\_to\_hub](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)
 方法：



```
model.push_to_hub("your-name/roberta-large-lora-token-classification")
```


## 推理



要使用模型进行推理，请加载配置和模型：



```
peft_model_id = "stevhliu/roberta-large-lora-token-classification"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForTokenClassification.from_pretrained(
    config.base_model_name_or_path, num_labels=11, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, peft_model_id)
```


获取一些要标记化的文本：



```
text = "The activation of IL-2 gene expression and NF-kappa B through CD28 requires reactive oxygen production by 5-lipoxygenase."
inputs = tokenizer(text, return_tensors="pt")
```


将输入传递给模型，并打印出每个标记的模型预测：



```
with torch.no_grad():
    logits = model(**inputs).logits

tokens = inputs.tokens()
predictions = torch.argmax(logits, dim=2)

for token, prediction in zip(tokens, predictions[0].numpy()):
    print((token, model.config.id2label[prediction]))
("<s>", "O")
("The", "O")
("Ġactivation", "O")
("Ġof", "O")
("ĠIL", "B-DNA")
("-", "O")
("2", "I-DNA")
("Ġgene", "O")
("Ġexpression", "O")
("Ġand", "O")
("ĠNF", "B-protein")
("-", "O")
("k", "I-protein")
("appa", "I-protein")
("ĠB", "I-protein")
("Ġthrough", "O")
("ĠCD", "B-protein")
("28", "I-protein")
("Ġrequires", "O")
("Ġreactive", "O")
("Ġoxygen", "O")
("Ġproduction", "O")
("Ġby", "O")
("Ġ5", "B-protein")
("-", "O")
("lip", "I-protein")
("oxy", "I-protein")
("gen", "I-protein")
("ase", "I-protein")
(".", "O")
("</s>", "O")
```