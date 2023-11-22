# 快速游览

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/quicktour>
>
> 原始地址：<https://huggingface.co/docs/peft/quicktour>


🤗 PEFT 包含用于训练大型预训练模型的参数高效微调方法。传统的范例是针对每个下游任务微调模型的所有参数，但由于当今模型中的参数数量巨大，这变得非常昂贵且不切实际。相反，训练较少数量的提示参数或使用低秩自适应（LoRA）等重新参数化方法来减少可训练参数的数量会更有效。


本快速浏览将向您展示 🤗 PEFT 的主要功能，并帮助您训练通常在消费设备上无法访问的大型预训练模型。您将看到如何训练 1.2B 参数
 [`bigscience/mt0-large`](https://huggingface.co/bigscience/mt0-large)
 使用 LoRA 进行模型生成分类标签并将其用于推理。


## 佩夫特配置



每个 🤗 PEFT 方法都由一个定义
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 存储用于构建的所有重要参数的类
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。


因为您要使用 LoRA，所以您需要加载并创建一个
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 班级。之内
 `LoraConfig`
 ，指定以下参数：


* 这
 `任务类型`
 ，或本例中的序列到序列语言建模
* `推理模式`
 ，无论您是否使用模型进行推理
* `r`
 ，低秩矩阵的维数
* `lora_alpha`
 ，低秩矩阵的缩放因子
* `lora_dropout`
 ，LoRA 层的丢失概率



```
from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
```


💡参见
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 有关您可以调整的其他参数的更多详细信息，请参考。


## 佩夫特模型



A
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 是由
 `get_peft_model()`
 功能。它需要一个基本模型 - 您可以从 🤗 Transformers 库加载 - 以及
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 包含如何为特定 🤗 PEFT 方法配置模型的说明。


首先加载您想要微调的基本模型。



```
from transformers import AutoModelForSeq2SeqLM

model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
```


包裹你的基础模型并
 `peft_config`
 与
 `get_peft_model`
 函数来创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。要了解模型中可训练参数的数量，请使用
 `打印可训练参数`
 方法。在这种情况下，您只训练了模型参数的 0.19%！ 🤏



```
from peft import get_peft_model

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
```


就是这样🎉！现在您可以使用 🤗 Transformers 训练模型
 [培训师](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.Trainer)
 、🤗 加速或任何自定义 PyTorch 训练循环。


## 保存并加载模型



模型完成训练后，您可以使用以下命令将模型保存到目录中
 [保存\_pretrained](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)
 功能。您还可以使用以下命令将模型保存到 Hub（请确保先登录您的 Hugging Face 帐户）：
 [push\_to\_hub](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)
 功能。



```
model.save_pretrained("output\_dir")

# if pushing to Hub
from huggingface_hub import notebook_login

notebook_login()
model.push_to_hub("my\_awesome\_peft\_model")
```


这只会保存训练后的增量 🤗 PEFT 权重，这意味着存储、传输和加载非常高效。例如，这个
 [`bigscience/T0_3B`](https://huggingface.co/smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM)
 使用 LoRA 训练的模型
 [`twitter_complaints`](https://huggingface.co/datasets/ought/raft/viewer/twitter_complaints/train)
 RAFT 的子集
 [数据集](https://huggingface.co/datasets/ought/raft)
 仅包含两个文件：
 `adapter_config.json`
 和
 `adapter_model.bin`
 。后一个文件只有 19MB！


使用以下命令轻松加载模型进行推理
 [来自\_pretrained](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
 功能：



```
  from transformers import AutoModelForSeq2SeqLM
+ from peft import PeftModel, PeftConfig

+ peft\_model\_id = "smangrul/twitter\_complaints\_bigscience\_T0\_3B\_LORA\_SEQ\_2\_SEQ\_LM"
+ config = PeftConfig.from\_pretrained(peft\_model\_id)
  model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
+ model = PeftModel.from\_pretrained(model, peft\_model\_id)
  tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

  model = model.to(device)
  model.eval()
  inputs = tokenizer("Tweet text : @HondaCustSvc Your customer service has been horrible during the recall process. I will never purchase a Honda again. Label :", return_tensors="pt")

  with torch.no_grad():
      outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
      print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
  'complaint'
```


## 轻松加载自动类



如果您已在本地或集线器上保存适配器，则可以利用
 `AutoPeftModelForxxx`
 类并使用一行代码加载任何 PEFT 模型：



```
- from peft import PeftConfig, PeftModel
- from transformers import AutoModelForCausalLM
+ from peft import AutoPeftModelForCausalLM

- peft\_config = PeftConfig.from\_pretrained("ybelkada/opt-350m-lora") 
- base\_model\_path = peft\_config.base\_model\_name\_or\_path
- transformers\_model = AutoModelForCausalLM.from\_pretrained(base\_model\_path)
- peft\_model = PeftModel.from\_pretrained(transformers\_model, peft\_config)
+ peft\_model = AutoPeftModelForCausalLM.from\_pretrained("ybelkada/opt-350m-lora")
```


目前，支持的汽车类别有：
 `CausalLM 的 AutoPeft 模型`
 ,
 `AutoPeftModelForSequenceClassification`
 ,
 `AutoPeftModelForSeq2SeqLM`
 ,
 `AutoPeftModelForTokenClassification`
 ,
 `AutoPeftModelForQuestionAnswering`
 和
 `用于特征提取的 AutoPeft 模型`
 。对于其他任务（例如 Whisper、StableDiffusion），您可以使用以下命令加载模型：



```
- from peft import PeftModel, PeftConfig, AutoPeftModel
+ from peft import AutoPeftModel
- from transformers import WhisperForConditionalGeneration

- model\_id = "smangrul/openai-whisper-large-v2-LORA-colab"

peft_model_id = "smangrul/openai-whisper-large-v2-LORA-colab"
- peft\_config = PeftConfig.from\_pretrained(peft\_model\_id)
- model = WhisperForConditionalGeneration.from\_pretrained(
- peft\_config.base\_model\_name\_or\_path, load\_in\_8bit=True, device\_map="auto"
- )
- model = PeftModel.from\_pretrained(model, peft\_model\_id)
+ model = AutoPeftModel.from\_pretrained(peft\_model\_id)
```


## 下一步



现在您已经了解了如何使用其中一种 🤗 PEFT 方法训练模型，我们鼓励您尝试其他一些方法，例如提示调整。这些步骤与本快速入门中显示的步骤非常相似；准备一个
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 对于 🤗 PEFT 方法，并使用
 `get_peft_model`
 创建一个
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 从配置和基本模型。然后你就可以随心所欲地训练它了！


如果您有兴趣使用 🤗 PEFT 方法训练模型以执行特定任务（例如语义分割、多语言自动语音识别、DreamBooth 和标记分类），请随意查看任务指南。