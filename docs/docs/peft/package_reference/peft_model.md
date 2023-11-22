# 楷模

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/package_reference/peft_model>
>
> 原始地址：<https://huggingface.co/docs/peft/package_reference/peft_model>


[PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 是用于指定要应用 PEFT 方法的基本 Transformer 模型和配置的基本模型类。基地
 `佩夫特模型`
 包含从 Hub 加载和保存模型的方法，并支持
 [PromptEncoder](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEncoder)
 以便及时学习。


## 佩夫特模型




### 


班级
 

 佩夫特。
 

 佩夫特模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L83)


（


 模型
 
 ：预训练模型


pft\_config
 
 : 佩夫特配置


适配器\_名称
 
 : str = '默认'


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 用于 Peft 的基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 模型的配置。
* **适配器\_名称**
 （
 `str`
 ) — 适配器的名称，默认为
 `“默认”`
 。


 包含各种 Peft 方法的基础模型。


**属性**
 ：


* **基础\_模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 用于 Peft 的基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 模型的配置。
* **模块\_要\_保存**
 （
 `列表`
 的
 `str`
 ) — 时要保存的子模块名称列表
保存模型。
* **提示\_编码器**
 （
 [PromptEncoder](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEncoder)
 ) — 用于 Peft 的提示编码器 if
使用
 [PromptLearningConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PromptLearningConfig)
 。
* **提示\_tokens**
 （
 `torch.张量`
 ) — 用于 Peft 的虚拟提示标记，如果
使用
 [PromptLearningConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PromptLearningConfig)
 。
* **变压器\_backbone\_name**
 （
 `str`
 ) — 变压器的名称
基本模型中的主干（如果使用）
 [PromptLearningConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PromptLearningConfig)
 。
* **词\_嵌入**
 （
 `torch.nn.Embedding`
 ) — Transformer 主干的词嵌入
在基本模型中，如果使用
 [PromptLearningConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PromptLearningConfig)
 。



#### 


创建\_或\_更新\_模型\_卡


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L697)


（


 输出\_dir
 
 : 字符串


）


更新或创建模型卡以包含有关 peft 的信息：


1. 新增
 `佩夫特`
 库标签
2.增加peft版本
3.添加基础型号信息
4. 添加量化信息（如果使用）




#### 


禁用\_适配器


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L526)


（
 

 ）


禁用适配器模块。




#### 


向前


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L512)


（


 \*参数
 
 ： 任何


\*\*kwargs
 
 ： 任何


）


模型的前向传递。




#### 


来自\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L263)


（


 模型
 
 ：预训练模型


型号\_id
 
 : 联合[str, os.PathLike]


适配器\_名称
 
 : str = '默认'


是\_可训练的
 
 ：布尔=假


配置
 
 : 可选[PeftConfig] = 无


\*\*kwargs
 
 ： 任何


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 )—
要适应的模型。该模型应使用以下命令进行初始化
 [来自\_pretrained](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
 🤗 Transformers 库中的方法。
* **型号\_id**
 （
 `str`
 或者
 `os.PathLike`
 )—
要使用的 PEFT 配置的名称。可以是：
 
+ 一个字符串，
`型号 ID`
Hugging Face 上模型存储库内托管的 PEFT 配置的示例
中心。
+ 包含使用保存的 PEFT 配置文件的目录的路径
`保存预训练`
方法 （
`./my_peft_config_directory/`
）。
* **适配器\_名称**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“默认”`
 )—
要加载的适配器的名称。这对于加载多个适配器非常有用。
* **可训练**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
适配器是否应该可训练。如果
 ‘假’
 ，适配器将被冻结并用于
推理
* **配置**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ,
 *选修的*
 )—
要使用的配置对象，而不是自动加载的配置。这个配置
对象是互斥的
 `型号_id`
 和
 `夸格斯`
 。当配置已经完成时这很有用
调用前已加载
 `from_pretrained`
 。
夸格斯——（
 `可选的`
 ）：
传递给特定 PEFT 配置类的其他关键字参数。


 从预训练模型和加载的 PEFT 权重实例化 PEFT 模型。


请注意，通过的
 `模型`
 可以就地修改。




#### 


获取\_base\_model


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L549)


（
 

 ）


返回基本模型。




#### 


获取\_nb\_可训练\_参数


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L471)


（
 

 ）


返回模型中可训练参数的数量和所有参数的数量。




#### 


得到\_提示


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L425)


（


 批量\_大小
 
 ：整数


任务\_ids
 
 : 可选[torch.Tensor] = 无


）


返回用于 Peft 的虚拟提示。仅适用于以下情况
 `peft_config.peft_type != PeftType.LORA`
 。




#### 


获取\_提示\_嵌入\_以\_保存


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L406)


（


 适配器\_名称
 
 : 字符串


）


返回保存模型时要保存的提示嵌入。仅适用于以下情况
 `peft_config.peft_type != PeftType.LORA`
 。




#### 


打印\_可训练\_参数


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L495)


（
 

 ）


打印模型中可训练参数的数量。




#### 


保存\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L157)


（


 保存\_目录
 
 : 字符串


安全\_序列化
 
 ：布尔=假


选定\_适配器
 
 : 可选[列表[str]] = 无


\*\*kwargs
 
 ： 任何


）


 参数


* **保存\_目录**
 （
 `str`
 )—
适配器模型和配置文件的保存目录（如果没有则创建）
存在）。
* **安全\_序列化**
 （
 `布尔`
 ,
 *选修的*
 )—
是否以 safetensors 格式保存适配器文件。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
传递给的附加关键字参数
 `推送到集线器`
 方法。


 该函数将适配器模型和适配器配置文件保存到一个目录中，以便于调用
使用重新加载
 [PeftModel.from\_pretrained()](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel.from_pretrained)
 类方法，也被
 `PefModel.push_to_hub()`
 方法。




#### 


设置\_适配器


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L678)


（


 适配器\_名称
 
 : 字符串


）


设置活动适配器。


## 序列分类的 Peft 模型



A
 `佩夫特模型`
 用于序列分类任务。



### 


班级
 

 佩夫特。
 

 序列分类的 Peft 模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L746)


（


 模型


 pft\_config
 
 : 佩夫特配置


适配器\_名称
 
 ='默认'


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 配置。


 用于序列分类任务的 Pft 模型。


**属性**
 ：


* **配置**
 （
 [PretrainedConfig](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/configuration#transformers.PretrainedConfig)
 ) — 基础模型的配置对象。
* **cls\_layer\_name**
 （
 `str`
 ) — 分类层的名称。


 例子：



```
>>> from transformers import AutoModelForSequenceClassification
>>> from peft import PeftModelForSequenceClassification, get_peft_config

>>> config = {
...     "peft\_type": "PREFIX\_TUNING",
...     "task\_type": "SEQ\_CLS",
...     "inference\_mode": False,
...     "num\_virtual\_tokens": 20,
...     "token\_dim": 768,
...     "num\_transformer\_submodules": 1,
...     "num\_attention\_heads": 12,
...     "num\_layers": 12,
...     "encoder\_hidden\_size": 768,
...     "prefix\_projection": False,
...     "postprocess\_past\_key\_value\_function": None,
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForSequenceClassification(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
```


## 令牌分类的 Pft 模型



A
 `佩夫特模型`
 用于标记分类任务。



### 


班级
 

 佩夫特。
 

 令牌分类的 Pft 模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L1349)


（


 模型


 pft\_config
 
 : PeftConfig = 无


适配器\_名称
 
 ='默认'


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 配置。


 用于令牌分类任务的 Pft 模型。


**属性**
 ：


* **配置**
 （
 [PretrainedConfig](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/configuration#transformers.PretrainedConfig)
 ) — 基础模型的配置对象。
* **cls\_layer\_name**
 （
 `str`
 ) — 分类层的名称。


 例子：



```
>>> from transformers import AutoModelForSequenceClassification
>>> from peft import PeftModelForTokenClassification, get_peft_config

>>> config = {
...     "peft\_type": "PREFIX\_TUNING",
...     "task\_type": "TOKEN\_CLS",
...     "inference\_mode": False,
...     "num\_virtual\_tokens": 20,
...     "token\_dim": 768,
...     "num\_transformer\_submodules": 1,
...     "num\_attention\_heads": 12,
...     "num\_layers": 12,
...     "encoder\_hidden\_size": 768,
...     "prefix\_projection": False,
...     "postprocess\_past\_key\_value\_function": None,
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForTokenClassification(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
```


## 因果LM的Pef模型



A
 `佩夫特模型`
 用于因果语言建模。



### 


班级
 

 佩夫特。
 

 因果LM的Pef模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L935)


（


 模型


 pft\_config
 
 : 佩夫特配置


适配器\_名称
 
 ='默认'


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 配置。


 用于因果语言建模的 Peft 模型。


 例子：



```
>>> from transformers import AutoModelForCausalLM
>>> from peft import PeftModelForCausalLM, get_peft_config

>>> config = {
...     "peft\_type": "PREFIX\_TUNING",
...     "task\_type": "CAUSAL\_LM",
...     "inference\_mode": False,
...     "num\_virtual\_tokens": 20,
...     "token\_dim": 1280,
...     "num\_transformer\_submodules": 1,
...     "num\_attention\_heads": 20,
...     "num\_layers": 36,
...     "encoder\_hidden\_size": 1280,
...     "prefix\_projection": False,
...     "postprocess\_past\_key\_value\_function": None,
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
>>> peft_model = PeftModelForCausalLM(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
```


## PeftModelForSeq2SeqLM



A
 `佩夫特模型`
 用于序列到序列的语言建模。



### 


班级
 

 佩夫特。
 

 PeftModelForSeq2SeqLM


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L1104)


（


 模型


 pft\_config
 
 : 佩夫特配置


适配器\_名称
 
 ='默认'


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 配置。


 用于序列到序列语言建模的 Peft 模型。


 例子：



```
>>> from transformers import AutoModelForSeq2SeqLM
>>> from peft import PeftModelForSeq2SeqLM, get_peft_config

>>> config = {
...     "peft\_type": "LORA",
...     "task\_type": "SEQ\_2\_SEQ\_LM",
...     "inference\_mode": False,
...     "r": 8,
...     "target\_modules": ["q", "v"],
...     "lora\_alpha": 32,
...     "lora\_dropout": 0.1,
...     "fan\_in\_fan\_out": False,
...     "enable\_lora": None,
...     "bias": "none",
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> peft_model = PeftModelForSeq2SeqLM(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
```


## Peft问题解答模型



A
 `佩夫特模型`
 用于回答问题。



### 


班级
 

 佩夫特。
 

 Peft问题解答模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L1522)


（


 模型


 pft\_config
 
 : PeftConfig = 无


适配器\_名称
 
 ='默认'


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 配置。


 用于提取式问答的 Peft 模型。


**属性**
 :


* **配置**
 （
 [PretrainedConfig](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/configuration#transformers.PretrainedConfig)
 ) — 基础模型的配置对象。
* **cls\_layer\_name**
 （
 `str`
 ) — 分类层的名称。


 例子：



```
>>> from transformers import AutoModelForQuestionAnswering
>>> from peft import PeftModelForQuestionAnswering, get_peft_config

>>> config = {
...     "peft\_type": "LORA",
...     "task\_type": "QUESTION\_ANS",
...     "inference\_mode": False,
...     "r": 16,
...     "target\_modules": ["query", "value"],
...     "lora\_alpha": 32,
...     "lora\_dropout": 0.05,
...     "fan\_in\_fan\_out": False,
...     "bias": "none",
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForQuestionAnswering(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 592900 || all params: 108312580 || trainable%: 0.5473971721475013
```


## 特征提取的Peft模型



A
 `佩夫特模型`
 用于从变压器模型中提取特征/嵌入。



### 


班级
 

 佩夫特。
 

 特征提取的Peft模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/peft_model.py#L1714)


（


 模型


 pft\_config
 
 : PeftConfig = 无


适配器\_名称
 
 ='默认'


）


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 基础变压器模型。
* **peft\_config**
 （
 [PeftConfig](/docs/peft/v0.6.2/en/package_reference/config#peft.PeftConfig)
 ) — Peft 配置。


 用于从 Transformer 模型中提取特征/嵌入的 Peft 模型


**属性**
 :


* **配置**
 （
 [PretrainedConfig](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/configuration#transformers.PretrainedConfig)
 ) — 基础模型的配置对象。


 例子：



```
>>> from transformers import AutoModel
>>> from peft import PeftModelForFeatureExtraction, get_peft_config

>>> config = {
...     "peft\_type": "LORA",
...     "task\_type": "FEATURE\_EXTRACTION",
...     "inference\_mode": False,
...     "r": 16,
...     "target\_modules": ["query", "value"],
...     "lora\_alpha": 32,
...     "lora\_dropout": 0.05,
...     "fan\_in\_fan\_out": False,
...     "bias": "none",
... }
>>> peft_config = get_peft_config(config)
>>> model = AutoModel.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForFeatureExtraction(model, peft_config)
>>> peft_model.print_trainable_parameters()
```