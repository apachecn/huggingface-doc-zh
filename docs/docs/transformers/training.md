# 微调预训练模型

> 译者：[BrightLi](https://github.com/brightli)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/training>
>
> 原始地址：<https://huggingface.co/docs/transformers/training>

使用预训练模型有显著的好处。它可以降低计算成本、减少碳足迹，并允许您在无需从头开始训练的情况下使用最先进的模型。🤗 Transformers 提供了数千个针对广泛任务的预训练模型。当您使用预训练模型时，您会针对特定于您的任务的数据集对其进行训练。这被称为微调，是一种极其强大的训练技术。在本教程中，您将使用您选择的深度学习框架微调一个预训练模型：

* 使用 🤗 Transformers Trainer 微调预训练模型。
* 在 TensorFlow 中使用 Keras 微调预训练模型。
* 在原生 PyTorch 中微调预训练模型。

**准备数据集**

<iframe width="509" height="320" src="https://www.youtube.com/embed/_BZearw7f0w" title="Hugging Face Datasets overview (Pytorch)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

在您能够微调预训练模型之前，需要下载一个数据集并为其训练做好准备。之前的教程向您展示了如何处理训练数据，现在您有机会将这些技能付诸实践！
首先，加载[Yelp Reviews](https://huggingface.co/datasets/yelp_review_full)数据集：

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

正如您现在所知，您需要一个分词器来处理文本，并包括一个填充和截断策略来处理任何可变序列长度。要一步处理您的数据集，请使用 🤗 Datasets 的[map](https://huggingface.co/docs/datasets/process#map)方法在整个数据集上应用预处理函数：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

如果您愿意，可以创建一个完整数据集的较小子集来进行微调，以减少所需时间：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

**训练**

此时，您应该遵循与您想要使用的框架相对应的部分。您可以使用右侧边栏中的链接跳转到您想要的部分 - 如果您想隐藏给定框架的所有内容，只需使用该框架块右上角的按钮！

<iframe width="475" height="320" src="https://www.youtube.com/embed/nvBXf7s7vTI" title="The Trainer API" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**用 PyTorch Trainer 训练**

🤗 Transformers 提供了一个针对训练 🤗 Transformers 模型进行优化的[Trainer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer)类，使得开始训练变得更加容易，无需手动编写自己的训练循环[Trainer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer)API支持广泛的训练选项和功能，如日志记录、梯度累积和混合精度。

首先，加载您的模型并指定预期标签的数量。从 Yelp Review[数据集卡片](https://huggingface.co/datasets/yelp_review_full#data-fields)中，您知道有五个标签：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

> 您会看到一个关于一些预训练权重未被使用和一些权重被随机初始化的警告。别担心，这完全正常！BERT模型的预训练头部被丢弃，并替换为一个随机初始化的分类头部。您将在序列分类任务上微调这个新模型头部，将预训练模型的知识转移到它上面。

**训练超参数**

接下来，创建一个[TrainingArguments](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments)类，其中包含您可以调整的所有超参数以及激活不同训练选项的标志。在本教程中，您可以从默认的训练[超参数](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)开始，但可以自由尝试这些参数以找到您的最佳设置。

指定保存训练期间检查点的位置：

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

**评估**

在训练过程中，[Trainer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer)不会自动评估模型性能。你需要向[Trainer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer) 传递一个函数来计算和报告指标。🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库提供了一个简单的准确性函数，你可以使用 evaluate.load（更多信息请参见此[快速教程](https://huggingface.co/docs/evaluate/a_quick_tour)）函数加载：

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

在你的预测上调用 compute 来计算准确性。在将你的预测传递给 compute 之前，你需要将 logits 转换为预测（记住所有 🤗 Transformers 模型都返回 logits）：

```py
>>> def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

如果你想在微调过程中监控你的评估指标，请在训练参数中指定 eval_strategy 参数，以便在每个epoch结束时报告评估指标：

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

**训练器**

使用你的模型、训练参数、训练和测试数据集以及评估函数创建一个[Trainer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer)对象：

```py
>>> trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

然后通过调用[train()](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer.train)来微调你的模型：

```py
>>> trainer.train()
```

<iframe width="381" height="320" src="https://www.youtube.com/embed/rnTGBy2ax1c" title="Keras introduction" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**用Keras训练TensorFlow模型**

你还可以使用Keras API在TensorFlow中训练 🤗 Transformers 模型！

**为Keras加载数据**

当你想用Keras API训练一个 🤗 Transformers 模型时，你需要将你的数据集转换为Keras能理解的格式。如果你的数据集很小，你可以将其全部转换为NumPy数组并传递给Keras。在我们进行更复杂的操作之前，先尝试一下这个方法。

首先，加载一个数据集。我们将使用来自[GLUE基准测试](https://huggingface.co/datasets/glue)的CoLA数据集，因为它是一个简单二元文本分类任务，现在只采用训练划分。

```py
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # Just take the training split for now
```

接下来，加载一个分词器并将数据分词化为 NumPy 数组。注意，标签已经是0和1的列表，所以我们可以直接将其转换为 NumPy 数组，无需分词！

```py
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1
```

最后，加载、[编译](https://keras.io/api/models/model_training_apis/#compile-method)并拟合模型。注意，Transformers 模型都有一个默认的任务相关损失函数，所以你不需要指定一个，除非你想这么做：

```py
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tokenized_data, labels)
```

> 当你编译模型时，不必向模型传递 loss 参数！如果这个参数留空，Hugging Face 模型会自动选择一个适合其任务和模型架构的损失函数。如果你想的话，你可以通过指定一个损失函数来覆盖这个设置！

这种方法对于较小的数据集效果很好，但对于较大的数据集，你可能会发现它开始成为一个问题。为什么？因为分词数组和标签必须完全加载到内存中，而且由于 NumPy 不处理“锯齿”数组，所以每个分词样本都必须填充到整个数据集中最长样本的长度。这将使你的数组变得更大，而这些填充标记也会减慢训练速度！

**将数据加载为 tf.data.Dataset**

如果你想避免减慢训练速度，你可以将你的数据加载为 tf.data.Dataset。尽管如果你愿意，你可以编写自己的 tf.data 管道，但我们有两个方便的方法来做到这一点：

[prepare_tf_dataset()](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset)：这是我们在大多数情况下推荐的方法。因为它是你模型上的一个方法，它可以检查模型以自动找出哪些列可以用作模型输入，并丢弃其他列以创建更简单、性能更高的数据集。

to_tf_dataset：这个方法更底层，当你想精确控制如何创建数据集时很有用，通过指定要包含的确切列和标签列。

在使用[prepare_tf_dataset()](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset)之前，你需要将分词器的输出作为列添加到你的数据集中，如下所示代码示例：

```py
def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

请记住，Hugging Face 数据集默认情况下存储在磁盘上，所以这不会占用你的内存！一旦添加了列，你可以从数据集中流式传输批次，并为每个批次添加填充，与为整个数据集添加填充相比，这极大地减少了填充标记的数量。

```py
>>> tf_dataset = model.prepare_tf_dataset(dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer)
```

请注意，在上面的代码示例中，你需要将分词器传递给 prepare_tf_dataset，以便它可以在加载时正确地填充批次。如果你的数据集中的样本长度都相同且不需要填充，你可以跳过这个参数。如果你需要做一些比简单的样本填充更复杂的事情（例如为掩码语言建模损坏标记），你可以使用 collate_fn 参数来传递一个函数，该函数将被调用以将样本列表转换为批次并应用任何你想要的预处理。请参阅我们的[示例](https://github.com/huggingface/transformers/tree/main/examples)或[笔记本](https://huggingface.co/docs/transformers/notebooks)以查看此方法的实际运行情况。

一旦你创建了一个 tf.data.Dataset，你就可以像之前一样编译和拟合模型：

```py
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tf_dataset)
```

**在原生PyTorch中训练**

<iframe width="381" height="320" src="https://www.youtube.com/embed/Dh9CL8fyG80" title="Write your training loop in PyTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[Trainer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer)负责处理训练循环，并允许你用一行代码微调模型。对于喜欢编写自己的训练循环的用户，你也可以在原生 PyTorch 中微调一个 🤗 Transformers 模型。
此时，你可能需要重新启动你的笔记本或执行以下代码以释放一些内存：

```py
del model
del trainer
torch.cuda.empty_cache()
```

接下来，手动对 tokenized_dataset 进行后处理以准备训练。

1.删除文本列，因为模型不接受原始文本作为输入：

```py
>>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
```

2.将标签列重命名为 labels，因为模型期望参数名为 labels：

```py
>>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```

3.将数据集的格式设置为返回 PyTorch 张量而不是列表：

```py
>>> tokenized_datasets.set_format("torch")
```

然后，如前所示创建一个数据集的较小子集，以加快微调速度：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

**DataLoader**

为你的训练和测试数据集创建一个 DataLoader，以便你可以迭代数据批次：

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

使用预期标签的数量加载你的模型：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

**优化器和学习率调度器**

创建一个优化器和学习率调度器来微调模型。让我们使用 PyTorch 的[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)优化器：

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

创建 Trainer 中的默认学习率调度器：

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
```

最后，指定设备以使用 GPU（如果你有访问权限）。否则，在 CPU 上训练可能需要几个小时，而不是几分钟。

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

>如果你没有 GPU，可以通过像[Colaboratory](https://colab.research.google.com/)或[SageMaker StudioLab](https://studiolab.sagemaker.aws/)这样的托管笔记本免费获得云 GPU。

太好了，现在你准备好训练了！🥳

**训练循环**

为了跟踪你的训练进度，使用[tqdm](https://tqdm.github.io/)库在训练步骤数上添加一个进度条：

```py
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

**评估**

就像你为[Trainer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer)添加了一个评估函数一样，当你编写自己的训练循环时也需要做同样的事情。但是这次不是在每个epoch结束时计算和报告度量，而是使用 add_batch 累积所有批次，并在最后计算度量。

```py
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```

**附加资源**

有关更多微调示例，请参阅：

🤗 [Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples)包括在 PyTorch 和 TensorFlow 中训练常见 NLP 任务的脚本。

🤗 [Transformers Notebooks](https://huggingface.co/docs/transformers/notebooks)包含各种笔记本，介绍如何在 PyTorch 和 TensorFlow 中为特定任务微调模型。