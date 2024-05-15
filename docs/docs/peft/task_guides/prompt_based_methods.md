# 基于提示的方法

> 译者：[糖醋鱼](https://github.com/now-101)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/prompt_based_methods>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/prompt_based_methods>

## 基于提示的方法

提示可以描述任务或提供你希望模型学习的任务示例。 软提示方法不是手动创建这些提示，而是将可学习的参数添加到输入 embedding 中，这些参数可以针对特定任务进行优化，同时保持预训练模型的参数冻结。 这使得为新的下游任务微调大型语言模型 (LLM) 变得更快、更容易。

PEFT 库支持多种类型的提示方法（ p-tuning, prefix tuning, prompt tuning ），你可以在[软提示指南](https://huggingface.co/docs/peft/conceptual_guides/prompting)中了解有关这些方法在概念上如何工作的更多信息。 如果你有兴趣将这些方法应用于其他任务和用例，请查看我们的[笔记本集合](https://huggingface.co/spaces/PEFT/soft-prompting)！

本指南将向你展示如何使用软提示方法训练因果语言模型，以生成推文是否属于投诉的分类。

> 熟悉训练[因果语言模型](https://huggingface.co/docs/transformers/tasks/language_modeling)的一般过程将会非常有帮助，并且可以让你专注于软提示方法。 如果你是新手，我们建议你首先查看 Transformers 文档中的因果语言建模指南。 当你准备好后，回来看看将 PEFT 纳入你的训练是多么容易！

在开始之前，请确保已安装所有必需的库。
```python
pip install -q peft transformers datasets
```

## 数据集
在本指南中，你将使用 [RAFT](https://huggingface.co/datasets/ought/raft) 数据集的 twitter_complaints 子集。 twitter_complaints 子集包含标记为投诉和无投诉的推文，你可以查看数据集查看器以更好地了解[数据的具体情况](https://huggingface.co/datasets/ought/raft/viewer/twitter_complaints)。

使用 [load_dataset](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset) 函数加载数据集并创建新的 text_label 列，以便更容易理解标签值 1 和 2 的含义。
```python
from datasets import load_dataset

ds = load_dataset("ought/raft", "twitter_complaints")

classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
ds = ds.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
ds["train"][0]
{"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2, "text_label": "no complaint"}
```

加载分词器，定义要使用的填充令牌，并确定令牌化标签的最大长度。
```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)
```
要创建一个预处理函数，该函数对推文文本和标签进行令牌化，对每个批次中的输入和标签进行填充，创建注意力掩码，并将序列截断到 max_length，然后转换 input_ids、attention_mask 和 labels 为 PyTorch 张量，你可以按照以下步骤编写函数：
```py
import torch

max_length = 64

def preprocess_function(examples, text_column="Tweet text", label_column="text_label"):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```
使用 [map](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Dataset.map) 函数将预处理函数应用于整个数据集，并移除未经处理的列，因为模型不需要这些列。
```py
processed_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```
最后，创建一个训练和评估DataLoader。 如果数据集中的样本位于 CPU 上，则可以设置 pin_memory=True 以加快在训练期间数据传输到 GPU 的速度。
```py
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_ds = processed_ds["train"]
eval_ds = processed_ds["test"]

batch_size = 16

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```

## 模型
现在让我们加载一个预训练模型作为软提示方法的基础模型。 本指南使用 bigscience/bloomz-560m 模型，但你可以使用任何你想要的因果语言模型。
```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
```

## PEFT配置和模型
对于任何 PEFT 方法，你需要创建一个配置，其中包含指定如何应用 PEFT 方法的所有参数。 设置配置后，将其与基本模型一起传递给 [get_peft_model()](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model#peft.get_peft_model) 函数以创建可训练的 [PeftModel](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model#peft.PeftModel)。

> 调用 [print_trainable_parameters()](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model#peft.PeftModel.print_trainable_parameters) 方法来比较 PeftModel 的可训练参数数量与基础模型中的参数数量！
### p-tuning （参数调整）
[P-tuning](https://huggingface.co/docs/peft/conceptual_guides/prompting#p-tuning)（参数调整）添加了一个可训练的嵌入张量，其中提示标记可以添加到输入序列的任何位置。你可以创建一个 [PromptEncoderConfig](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/p_tuning#peft.PromptEncoderConfig)，用于指定任务类型、要添加和学习的虚拟标记数量，以及用于学习提示参数的编码器的隐藏大小。
```py
from peft import PromptEncoderConfig, get_peft_model

peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"
```
### prefix tuning （前缀调整）
[Prefix tuning](https://huggingface.co/docs/peft/conceptual_guides/prompting#prefix-tuning)（前缀调整）在模型的所有层中添加了任务特定的参数，这些参数由一个单独的前馈网络进行优化。你可以创建一个 [PrefixTuningConfig](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/prefix_tuning#peft.PrefixTuningConfig)，用于指定任务类型和要添加和学习的虚拟标记数量。
```py
from peft import PrefixTuningConfig, get_peft_model

peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"
```
### prompt tuning (提示调整）
[Prompt tuning](https://huggingface.co/docs/peft/conceptual_guides/prompting#prompt-tuning)（提示调整）将所有任务都建模为生成任务，并向输入添加了一个任务特定的提示，该提示独立更新。Prompt_tuning_init_text 参数指定如何微调模型（在本例中，它是对推文是否是投诉进行分类）。 为了获得最佳结果，prompt_tuning_init_text 应具有与预测相同数量的标记。 为此，你可以将 num_virtual_tokens 设置为 Prompt_tuning_init_text 的标记数量。

使用任务类型、用于训练模型的初始提示调整文本、要添加和学习的虚拟标记数量以及标记生成器创建 [PromptTuningConfig](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/prompt_tuning#peft.PromptTuningConfig)。
```py
from` peft import PromptTuningConfig, PromptTuningInit, get_peft_model

prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358"
```

## 训练
设置优化器和学习率调度器。
```py
from transformers import get_linear_schedule_with_warmup

lr = 3e-2
num_epochs = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```
将模型移至 GPU 并创建一个训练循环，报告每个 epoch 的损失和困惑度。
```py
from tqdm import tqdm

device = "cuda"
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
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```    
## 共享你的模型
要将训练完成的模型上传到 [Hugging Face Hub](https://huggingface.co/docs/transformers/v4.39.0/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)，你可以使用 push_to_hub 方法。在此之前，你需要先登录到你的 Hugging Face 帐号，并在提示时输入你的访问令牌（token）
```py
from huggingface_hub import notebook_login

account = <your-hf-account-name>
peft_model_id = f"{account}/bloomz-560-m-peft-method"
model.push_to_hub(peft_model_id)
```
如果你检查存储库中的模型文件大小，你会发现它比全尺寸模型小很多！

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png "For example, the adapter weights for a opt-350m model stored on the Hub are only ~6MB compared to the full model size which can be ~700MB.")

## 推理
让我们加载模型进行推理并在推文上进行测试！
```py
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("peft_model_id").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

i = 15
inputs = tokenizer(f'{text_column} : {ds["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
print(ds["test"][i]["Tweet text"])
"@NYTsupport i have complained a dozen times &amp; yet my papers are still thrown FAR from my door. Why is this so hard to resolve?"
```

调用 [generate](https://huggingface.co/docs/transformers/v4.39.0/en/model_doc/phi#transformers.PhiForCausalLM.generate) 方法生成预测的分类标签。

```py
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
"['Tweet text : @NYTsupport i have complained a dozen times &amp; yet my papers are still thrown FAR from my door. Why is this so hard to resolve? Label : complaint']"
```


