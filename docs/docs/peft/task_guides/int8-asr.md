# 自动语音识别的int8训练

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/task_guides/int8-asr>
>
> 原始地址：<https://huggingface.co/docs/peft/task_guides/int8-asr>


量化会降低浮点数据类型的精度，从而减少存储模型权重所需的内存。但是，量化会降低推理性能，因为降低精度时会丢失信息。 8 位或
 `int8`
 量化仅使用四分之一精度，但它不会降低性能，因为它不只是丢弃位或数据。反而，
 `int8`
 量化
 *回合*
 从一种数据类型转换为另一种数据类型。


💡 阅读
 [LLM.int8()：大规模 Transformer 的 8 位矩阵乘法](https://arxiv.org/abs/2208.07339)
 论文了解更多，或者你可以看看相应的
 [博客文章](https://huggingface.co/blog/hf-bitsandbytes-integration)
 进行更温和的介绍。


本指南将向您展示如何训练
 [`openai/whisper-large-v2`](https://huggingface.co/openai/whisper-large-v2)
 使用以下组合的多语言自动语音识别 (ASR) 模型
 `int8`
 量化和 LoRA。您将从以下位置训练 Whisper 进行马拉地语多语言 ASR：
 [通用语音 11.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
 数据集。


在开始之前，请确保已安装所有必需的库：



```
!pip install -q peft transformers datasets accelerate evaluate jiwer bitsandbytes
```


## 设置



让我们先进行一些设置，以便您稍后可以更快地开始训练。设置
 `CUDA_VISIBLE_DEVICES`
 到
 `0`
 使用机器上的第一个 GPU。然后，您可以指定模型名称（Hub 模型存储库 ID 或包含模型的目录的路径）、要训练的语言和语言缩写、任务类型和数据集名称：



```
import os

os.environ["CUDA\_VISIBLE\_DEVICES"] = "0"
model_name_or_path = "openai/whisper-large-v2"
language = "Marathi"
language_abbr = "mr"
task = "transcribe"
dataset_name = "mozilla-foundation/common\_voice\_11\_0"
```


如果您愿意，您还可以登录您的 Hugging Face 帐户，在 Hub 上保存并分享经过训练的模型：



```
from huggingface_hub import notebook_login

notebook_login()
```


## 加载数据集和指标



这
 [通用语音 11.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
 数据集包含许多小时的多种不同语言的录音语音。本指南使用
 [马拉地语](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/mr/train)
 语言作为示例，但请随意使用您感兴趣的任何其他语言。


初始化一个
 `数据集字典`
 结构，并加载
 `火车`
 （加载两个
 `训练+验证`
 分成
 `火车`
 ） 和
 `测试`
 从数据集分割成：



```
from datasets import load_dataset
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)
common_voice["train"][0]
```


## 预处理数据集



让我们准备训练数据集。加载特征提取器、分词器和处理器。您还应该将语言和任务传递给分词器和处理器，以便它们知道如何处理输入：



```
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)
```


您将只接受以下方面的培训：
 `句子`
 和
 `音频`
 列，因此您可以使用以下命令删除其余元数据
 `删除_列`
 ：



```
common_voice = common_voice.remove_columns(
    ["accent", "age", "client\_id", "down\_votes", "gender", "locale", "path", "segment", "up\_votes"]
)
common_voice["train"][0]
{
    "audio": {
        "path": "/root/.cache/huggingface/datasets/downloads/extracted/f7e1ef6a2d14f20194999aad5040c5d4bb3ead1377de3e1bbc6e9dba34d18a8a/common\_voice\_mr\_30585613.mp3",
        "array": array(
            [1.13686838e-13, -1.42108547e-13, -1.98951966e-13, ..., 4.83472422e-06, 3.54798703e-06, 1.63231743e-06]
        ),
        "sampling\_rate": 48000,
    },
    "sentence": "आईचे आजारपण वाढत चालले, तसतशी मथीही नीट खातपीतनाशी झाली.",
}
```


如果你看一下
 `采样率`
 ，您会看到音频是以 48kHz 采样的。 Whisper 模型在 16kHZ 的音频输入上进行了预训练，这意味着您需要对音频输入进行下采样以匹配模型的预训练内容。使用以下命令对音频进行下采样
 `cast_column`
 方法上的
 `音频`
 列，并设置
 `采样率`
 至 16kHz。下次调用时，音频输入会立即重新采样：



```
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice["train"][0]
{
    "audio": {
        "path": "/root/.cache/huggingface/datasets/downloads/extracted/f7e1ef6a2d14f20194999aad5040c5d4bb3ead1377de3e1bbc6e9dba34d18a8a/common\_voice\_mr\_30585613.mp3",
        "array": array(
            [-3.06954462e-12, -3.63797881e-12, -4.54747351e-12, ..., -7.74800901e-06, -1.74738125e-06, 4.36312439e-06]
        ),
        "sampling\_rate": 16000,
    },
    "sentence": "आईचे आजारपण वाढत चालले, तसतशी मथीही नीट खातपीतनाशी झाली.",
}
```


清理数据集后，您可以编写一个函数来生成正确的模型输入。该函数应该：


1. 通过加载将音频输入重新采样到 16kHZ
 `音频`
 柱子。
2. 计算音频的输入特征
 `数组`
 使用特征提取器。
3. 代币化
 `句子`
 列到输入标签。



```
def prepare\_dataset(batch):
    audio = batch["audio"]
    batch["input\_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling\_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
```


应用
 `准备数据集`
 函数到数据集
 `地图`
 功能，并设置
 `num_proc`
 论证
 `2`
 启用多处理（如果
 `地图`
 挂起，然后设置
 `num_proc=1`
 ）：



```
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)
```


最后，创建一个
 `数据整理器`
 类将每个批次中的标签填充到最大长度，并将填充替换为
 `-100`
 所以它们被损失函数忽略了。然后初始化数据整理器的实例：



```
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@数据类
类 DataCollat​​orSpeechSeq2SeqWithPadding：
    处理器：任何

    def \_\_call\_\_(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input\_features": feature["input\_features"]} 用于特征中的特征]
        批处理= self.processor.feature_extractor.pad（input_features，return_tensors =“pt”）

        label_features = [{"input\_ids": feature["labels"]} 用于特征中的特征]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        标签 = labels_batch["input\_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            标签 = 标签[:, 1:]

        批次[“标签”] = 标签

        退货批次


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```


## 火车



现在数据集已准备就绪，您可以将注意力转向模型。首先加载预训练的
 `openai/whisper-large-v2`
 模型来自
 [AutoModelForSpeechSeq2Seq](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/auto#transformers.AutoModelForSpeechSeq2Seq)
 ，并确保设置
 `load_in_8bit`
 论证
 '真实'
 启用
 `int8`
 量化。这
 `device_map=自动`
 参数自动确定如何加载和存储模型权重：



```
from transformers import AutoModelForSpeechSeq2Seq

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
```


你应该配置
 `forced_decoder_ids=无`
 因为在采样之前不使用任何标记，并且您也不需要在生成过程中抑制任何标记：



```
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
```


为模型做好准备
 `int8`
 量化，使用效用函数
 [`prepare_model_for_int8_training`](https://github.com/huggingface/peft/blob/34027fe813756897767b9a6f19ae7f1c4c7b418c/src/peft/utils/other.py#L35)
 处理以下事项：


* 投射所有非
 `int8`
 模块达到全精度（
 `fp32`
 ）为了稳定性
* 在输入嵌入层添加一个前向钩子来计算输入隐藏状态的梯度
* 启用梯度检查点以实现更高效的内存训练



```
from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model)
```


我们还将 LoRA 应用到训练中，使其更加高效。加载一个
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 并配置以下参数：


* `r`
 ，低秩矩阵的维数
* `lora_alpha`
 , 权重矩阵的比例因子
* `目标模块`
 ，应用 LoRA 的注意力矩阵的名称（
 `q_proj`
 和
 `v_proj`
 ，或本例中的查询和值）
* `lora_dropout`
 , LoRA 层的丢失概率
*`偏见`
 ， 设置
 `无`


💡 权重矩阵按比例缩放
 `lora_alpha/r`
 ，以及更高的
 `lora_alpha`
 value 为 LoRA 激活分配更多权重。为了性能，我们建议将偏差设置为
 `无`
 首先，然后
 `仅劳拉`
 ，在尝试之前
 `全部`
 。



```
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q\_proj", "v\_proj"], lora_dropout=0.05, bias="none")
```


设置完成后
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 ，将其和基本模型包裹起来
 `get_peft_model()`
 函数来创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。打印出可训练参数的数量，看看 LoRA 与完全训练模型相比效率提高了多少！



```
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 15728640 || all params: 1559033600 || trainable%: 1.0088711365810203"
```


现在您已准备好在中定义一些训练超参数
 [Seq2SeqTrainingArguments](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments)
 类，例如模型的保存位置、批量大小、学习率和训练周期数。这
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 与基本模型没有相同的签名，因此您需要显式设置
 `remove_unused_columns=False`
 和
 `label_names=["标签"]`
 。



```
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="your-name/int8-whisper-large-v2-asr",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
)
```


编写自定义也是一个好主意
 [TrainerCallback](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/callback#transformers.TrainerCallback)
 在训练期间保存模型检查点：



```
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on\_save(
 self,
 args: TrainingArguments,
 state: TrainerState,
 control: TrainerControl,
 \*\*kwargs,
 ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX\_CHECKPOINT\_DIR}-{state.global\_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter\_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch\_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control
```


通过
 `Seq2SeqTrainingArguments`
 、模型、数据集、数据整理器、分词器和回调
 [Seq2SeqTrainer](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Seq2SeqTrainer)
 。您可以选择设置
 `model.config.use_cache = False`
 消除任何警告。一切准备就绪后，致电
 [火车](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer.train)
 开始训练！



```
from transformers import Seq2SeqTrainer, TrainerCallback, Seq2SeqTrainingArguments, TrainerState, TrainerControl

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False
trainer.train()
```


## 评价



[单词错误率](https://huggingface.co/spaces/evaluate-metric/wer)
 (WER) 是评估 ASR 模型的常用指标。从 🤗 评估加载 WER 指标：



```
import evaluate

metric = evaluate.load("wer")
```


编写一个循环来评估模型性能。首先将模型设置为评估模式，然后编写循环
 [`torch.cuda.amp.autocast()`](https://pytorch.org/docs/stable/amp.html)
 因为
 `int8`
 训练需要自动施法。然后，将一批示例传递给模型进行评估。获取解码后的预测和标签，并在调用之前将它们批量添加到 WER 指标中
 `计算`
 得到最终的WER分数：



```
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc

eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input\_features"].to("cuda"),
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric.compute()
print(f"{wer=}")
```


## 分享模型



一旦您对结果感到满意，您可以使用以下命令将模型上传到 Hub：
 [push\_to\_hub](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)
 方法：



```
model.push_to_hub("your-name/int8-whisper-large-v2-asr")
```


## 推理



现在让我们测试一下模型吧！


实例化模型配置
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ，从这里，您可以使用配置来加载基础和
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 、分词器、处理器和特征提取器。请记住定义
 `语言`
 和
 `任务`
 在分词器、处理器和
 `forced_decoder_ids`
 ：



```
from peft import PeftModel, PeftConfig

peft_model_id = "smangrul/openai-whisper-large-v2-LORA-colab"
language = "Marathi"
task = "transcribe"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
```


加载音频样本（您可以在
 [数据集预览](https://huggingface.co/datasets/stevhliu/dummy)
 ) 进行转录，并且
 [AutomaticSpeechRecognitionPipeline](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
 ：



```
from transformers import AutomaticSpeechRecognitionPipeline

audio = "https://huggingface.co/datasets/stevhliu/dummy/resolve/main/mrt\_01523\_00028548203.wav"
pipeline = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
```


然后使用带有 autocast 的管道作为音频样本的上下文管理器：



```
with torch.cuda.amp.autocast():
    text = pipe(audio, generate_kwargs={"forced\_decoder\_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
text
"मी तुमच्यासाठी काही करू शकतो का?"
```