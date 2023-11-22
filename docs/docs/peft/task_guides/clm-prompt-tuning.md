# 因果语言建模的及时调整

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/clm-prompt-tuning>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning>


![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)


![在 Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)


提示通过添加一些特定于任务的输入文本来帮助指导语言模型行为。提示调整是一种附加方法，仅用于训练和更新新添加的提示标记到预训练模型。这样，您可以使用一个权重被冻结的预训练模型，并为每个下游任务训练和更新一组较小的提示参数，而不是完全微调单独的模型。随着模型变得越来越大，及时调整会更加有效，并且随着模型参数的扩展，结果会更好。


💡阅读
 [参数高效提示调整的规模力量](https://arxiv.org/abs/2104.08691)
 了解有关提示调整的更多信息。


本指南将向您展示如何应用提示调整来训练
 [`bloomz-560m`](https://huggingface.co/bigscience/bloomz-560m)
 模型上的
 `twitter_complaints`
 的子集
 [RAFT](https://huggingface.co/datasets/ought/raft)
 数据集。


在开始之前，请确保已安装所有必需的库：



```
!pip install -q peft transformers datasets
```


## 设置



首先定义模型和分词器、要训练的数据集和数据集列、一些训练超参数以及
 [PromptTuningConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptTuningConfig)
 。这
 [PromptTuningConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptTuningConfig)
 包含有关任务类型、初始化提示嵌入的文本、虚拟标记的数量以及要使用的标记生成器的信息：



```
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda"
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "twitter\_complaints"
checkpoint_name = f"{dataset\_name}\_{model\_name\_or\_path}\_{peft\_config.peft\_type}\_{peft\_config.task\_type}\_v1.pt".replace(
    "/", "\_"
)
text_column = "Tweet text"
label_column = "text\_label"
max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8
```


## 加载数据集



对于本指南，您将加载
 `twitter_complaints`
 的子集
 [RAFT](https://huggingface.co/datasets/ought/raft)
 数据集。该子集包含标记为
 `投诉`
 或者
 `没有抱怨`
 ：



```
dataset = load_dataset("ought/raft", dataset_name)
dataset["train"][0]
{"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2}
```


为了使
 `标签`
 列更具可读性，替换
 `标签`
 值与相应的标签文本并将它们存储在
 `文本标签`
 柱子。您可以使用
 `地图`
 函数一步将这一更改应用于整个数据集：



```
classes = [k.replace("\_", " ") for k in dataset["train"].features["Label"].names]
dataset = dataset.map(
    lambda x: {"text\_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
dataset["train"][0]
{"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2, "text\_label": "no complaint"}
```


## 预处理数据集



接下来，您将设置一个分词器；配置适当的填充标记以用于填充序列，并确定标记化标签的最大长度：



```
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input\_ids"]) for class_label in classes])
print(target_max_length)
3
```


创建一个
 `预处理函数`
 到：


1. 对输入文本和标签进行标记。
2. 对于批次中的每个示例，使用分词器填充标签
 `pad_token_id`
 。
3. 将输入文本和标签连接到
 `模型输入`
 。
4. 创建一个单独的注意力蒙版
 `标签`
 和
 `模型输入`
 。
5. 再次循环遍历批次中的每个示例，将输入 id、标签和注意掩码填充到
 `最大长度`
 并将它们转换为 PyTorch 张量。



```
def preprocess\_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text\_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input\_ids"][i]
        label_input_ids = labels["input\_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample\_input\_ids, label\_input\_ids)
        model_inputs["input\_ids"][i] = sample_input_ids + label_input_ids
        labels["input\_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention\_mask"][i] = [1] * len(model_inputs["input\_ids"][i])
    # print(model\_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input\_ids"][i]
        label_input_ids = labels["input\_ids"][i]
        model_inputs["input\_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention\_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention\_mask"
        ][i]
        labels["input\_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input\_ids"][i] = torch.tensor(model_inputs["input\_ids"][i][:max_length])
        model_inputs["attention\_mask"][i] = torch.tensor(model_inputs["attention\_mask"][i][:max_length])
        labels["input\_ids"][i] = torch.tensor(labels["input\_ids"][i][:max_length])
    model_inputs["labels"] = labels["input\_ids"]
    return model_inputs
```


使用
 `地图`
 函数来应用
 `预处理函数`
 到整个数据集。您可以删除未处理的列，因为模型不需要它们：



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
eval_dataset = processed_datasets["test"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```


## 火车



您几乎已准备好设置模型并开始训练！


初始化基本模型
 [AutoModelForCausalLM](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/auto#transformers.AutoModelForCausalLM)
 ，并通过它和
 `peft_config`
 到
 `get_peft_model()`
 函数来创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。您可以打印新的
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 的可训练参数，看看它比训练原始模型的完整参数效率高多少！



```
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
"trainable params: 8192 || all params: 559222784 || trainable%: 0.0014648902430985358"
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


将模型移至 GPU，然后编写训练循环开始训练！



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


## 分享模型



如果您愿意，您可以在 Hub 上存储和共享您的模型。登录您的 Hugging Face 帐户并在出现提示时输入您的令牌：



```
from huggingface_hub import notebook_login

notebook_login()
```


使用
 [push\_to\_hub](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)
 将模型上传到 Hub 上的模型存储库的函数：



```
peft_model_id = "your-name/bloomz-560m\_PROMPT\_TUNING\_CAUSAL\_LM"
model.push_to_hub("your-name/bloomz-560m\_PROMPT\_TUNING\_CAUSAL\_LM", use_auth_token=True)
```


模型上传后，您会看到模型文件大小只有 33.5kB！ 🤏


## 推理



让我们在示例输入上尝试该模型以进行推理。如果您查看将模型上传到的存储库，您将看到
 `adapter_config.json`
 文件。将此文件加载到
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 指定
 `peft_type`
 和
 `任务类型`
 。然后就可以加载提示调整的模型权重，并将配置放入
 [来自_pretrained()](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel.from_pretrained)
 来创建
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 ：



```
from peft import PeftModel, PeftConfig

peft_model_id = "stevhliu/bloomz-560m\_PROMPT\_TUNING\_CAUSAL\_LM"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
```


获取一条推文并将其标记化：



```
inputs = tokenizer(
    f'{text\_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)
```


将模型放在 GPU 上并
 *产生*
 预测标签：



```
model.to(device)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input\_ids"], attention_mask=inputs["attention\_mask"], max_new_tokens=10, eos_token_id=3
    )
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
[
    "Tweet text : @nationalgridus I have no water and the bill is current and paid. Can you do something about this? Label : complaint"
]
```