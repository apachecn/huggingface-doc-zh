# 条件生成的前缀调整

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/seq2seq-prefix-tuning>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/seq2seq-prefix-tuning>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


前缀调整是一种附加方法，其中仅将一系列连续的特定于任务的向量附加到输入的开头，或者
 *字首*
 。仅优化前缀参数并将其添加到模型每一层的隐藏状态中。输入序列的标记仍然可以关注前缀：
 *虚拟代币*
 。因此，前缀调优存储的参数比完全微调的模型少 1000 倍，这意味着您可以使用一个大型语言模型来完成许多任务。


💡阅读
 [前缀调优：优化生成连续提示](https://arxiv.org/abs/2101.00190)
 了解有关前缀调整的更多信息。


本指南将向您展示如何应用前缀调整来训练
 [`t5-large`](https://huggingface.co/t5-large)
 模型上的
 `sentences_allagree`
 的子集
 [financial\_phrasebank](https://huggingface.co/datasets/financial_phrasebank)
 数据集。


在开始之前，请确保已安装所有必需的库：



```
!pip install -q peft transformers datasets
```


## 设置



首先定义模型和分词器、文本和标签列以及一些超参数，以便以后更容易更快地开始训练。设置环境变量
 `TOKENIZERS_PARALLELSIM`
 到
 `假`
 禁用默认情况下并行处理数据的基于 Rust 的快速分词器，以便您可以在 Python 中使用多重处理。



```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

os.environ["TOKENIZERS\_PARALLELISM"] = "false"
os.environ["CUDA\_VISIBLE\_DEVICES"] = "3"

device = "cuda"
model_name_or_path = "t5-large"
tokenizer_name_or_path = "t5-large"

text_column = "sentence"
label_column = "text\_label"
max_length = 128
lr = 1e-2
num_epochs = 5
batch_size = 8
```


## 加载数据集



对于本指南，您将接受以下培训：
 `sentences_allagree`
 的子集
 [`financial_phrasebank`](https://huggingface.co/datasets/financial_phrasebank)
 数据集。该数据集包含按情绪分类的财经新闻。


使用🤗
 [数据集](https://huggingface.co/docs/datasets/index)
`train_test_split`
 函数创建训练和验证分割并转换
 `标签`
 更具可读性的价值
 `文本标签`
 。所有更改都可以应用
 `地图`
 功能：



```
from datasets import load_dataset

dataset = load_dataset("financial\_phrasebank", "sentences\_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text\_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

dataset["train"][0]
{"sentence": "Profit before taxes was EUR 4.0 mn , down from EUR 4.9 mn .", "label": 0, "text\_label": "negative"}
```


## 预处理数据集



初始化一个分词器，并创建一个函数来填充和截断
 `模型输入`
 和
 `标签`
 :



```
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def preprocess\_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max\_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=2, padding="max\_length", truncation=True, return_tensors="pt")
    labels = labels["input\_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs
```


使用
 `地图`
 函数来应用
 `预处理函数`
 到数据集。您可以删除未处理的列，因为模型不再需要它们：



```
processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```


创建一个
 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
 来自
 `火车`
 和
 `评估`
 数据集。放
 `pin_memory=True`
 如果数据集中的样本位于 CPU 上，则可以在训练期间加快数据传输到 GPU 的速度。



```
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```


## 火车模型



现在您可以设置模型并确保它已准备好进行训练。指定任务在
 [PrefixTuningConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PrefixTuningConfig)
 ，创建基础
 `t5-大号`
 模型来自
 [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/auto#transformers.AutoModelForSeq2SeqLM)
 ，然后将模型和配置包装在
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。请随意打印
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 的参数并与完全训练所有模型参数进行比较，看看效率提高了多少！



```
peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 983040 || all params: 738651136 || trainable%: 0.13308583065659835"
```


设置优化器和学习率调度器：



```
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```


将模型移至 GPU，然后编写训练循环来开始！



```
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train\_ppl=} {train\_epoch\_loss=} {eval\_ppl=} {eval\_epoch\_loss=}")
```


让我们看看模型在验证集上的表现如何：



```
correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["validation"]["text\_label"]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval\_preds[:10]=}")
print(f"{dataset['validation']['text\_label'][:10]=}")
"accuracy=97.3568281938326 % on the evaluation dataset"
"eval\_preds[:10]=['neutral', 'positive', 'neutral', 'positive', 'neutral', 'negative', 'negative', 'neutral', 'neutral', 'neutral']"
"dataset['validation']['text\_label'][:10]=['neutral', 'positive', 'neutral', 'positive', 'neutral', 'negative', 'negative', 'neutral', 'neutral', 'neutral']"
```


只需几分钟即可达到 97% 的准确率；不错！


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
peft_model_id = "your-name/t5-large\_PREFIX\_TUNING\_SEQ2SEQ"
model.push_to_hub("your-name/t5-large\_PREFIX\_TUNING\_SEQ2SEQ", use_auth_token=True)
```


如果您检查存储库中的模型文件大小，您会发现它只有 3.93MB！ 🤏


## 推理



模型上传到 Hub 后，任何人都可以轻松使用它进行推理。加载配置和模型：



```
from peft import PeftModel, PeftConfig

peft_model_id = "stevhliu/t5-large\_PREFIX\_TUNING\_SEQ2SEQ"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
```


获取并标记一些有关财经新闻的文本：



```
inputs = tokenizer(
    "The Lithuanian beer market made up 14.41 million liters in January , a rise of 0.8 percent from the year-earlier figure , the Lithuanian Brewers ' Association reporting citing the results from its members .",
    return_tensors="pt",
)
```


将模型放在 GPU 上并
 *产生*
 预测的文本情绪：



```
model.to(device)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input\_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
["positive"]
```