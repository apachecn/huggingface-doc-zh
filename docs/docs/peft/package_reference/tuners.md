# 调音器

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/package_reference/tuners>
>
> 原始地址：<https://huggingface.co/docs/peft/package_reference/tuners>


每个调谐器（或 PEFT 方法）都有一个配置和模型。


## 洛拉



用于使用 LoRA 微调模型。



### 


班级
 

 佩夫特。
 

 洛拉配置


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/config.py#L24)


（


 peft\_type
 
 : 打字.Union[str, peft.utils.peft\_types.PefType] = None


自动\_映射
 
 : 打字.可选[dict] =无


基本\_模型\_名称\_或\_路径
 
 ：str=无


修订
 
 ：str=无


任务\_类型
 
 : Typing.Union[str, peft.utils.peft\_types.TaskType] = None


推理\_模式
 
 ：布尔=假


r
 
 ：整数= 8


目标\_模块
 
 : 打字.Union[str, 打字.List[str], NoneType] = 无


劳拉\_alpha
 
 ：整数= 8


劳拉\_dropout
 
 ：浮动= 0.0


扇入_扇出_扇出
 
 ：布尔=假


偏见
 
 : str = '无'


要保存的模块
 
 : 打字.可选[打字.列表[str]] =无


初始化\_lora\_weights
 
 ：布尔=真


层\_to\_变换
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


层\_pattern
 
 : 打字.Union[str, 打字.List[str], NoneType] = 无


等级\_模式
 
 : 打字.可选[dict] = <工厂>


阿尔法\_模式
 
 : 打字.可选[dict] = <工厂>


）


 参数


* **r**
 （
 `int`
 ) — Lora 注意力维度。
* **目标\_模块**
 （
 `联合[列表[str],str]`
 ) — 应用 Lora 的模块的名称。
* **劳拉\_alpha**
 （
 `int`
 ) — Lora 缩放的 alpha 参数。
* **劳拉\_辍学**
 （
 `浮动`
 ) — Lora 层的 dropout 概率。
* **扇入_扇出_扇出**
 （
 `布尔`
 ) — 如果要替换的层存储像 (fan\_in, fan\_out) 这样的权重，则将此设置为 True。
例如，gpt-2 使用
 `Conv1D`
 它存储像 (fan\_in, fan\_out) 这样的权重，因此应该设置它
到
 '真实'
 。
* **偏见**
 （
 `str`
 ) — Lora 的偏差类型。可以是“无”、“全部”或“lora\_only”。如果是“all”或“lora\_only”，则
相应的偏差将在训练期间更新。请注意，这意味着即使禁用
如果使用适配器，模型将不会产生与没有适应的基本模型相同的输出。
* **模块\_要\_保存**
 （
 `列表[str]`
 ) —除了 LoRA 层之外要设置为可训练的模块列表
并保存在最后的检查点。
* **层\_to\_变换**
 （
 `联合[列表[int],int]`
 )—
要变换的层索引，如果指定此参数，它将应用 LoRA 变换
在此列表中指定的图层索引。如果传递单个整数，它将应用 LoRA
在此索引处的图层上的变换。
* **层\_pattern**
 （
 `str`
 )—
图层图案名称，仅在以下情况下使用
 `要变换的图层`
 不同于
 `无`
 如果该层
图案不在公共层图案中。
* **排名\_模式**
 （
 `字典`
 )—
从图层名称或正则表达式到与默认排名不同的排名的映射
由指定
 `r`
 。
* **alpha\_pattern**
 （
 `字典`
 )—
从图层名称或正则表达式到与默认 alpha 不同的 alpha 的映射
由指定
 `lora_alpha`
 。


 这是存储配置的配置类
 [LoraModel](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraModel)
 。




### 


班级
 

 佩夫特。
 

 劳拉模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/model.py#L53)


（


 模型


 配置


 适配器\_名称


）
 

 →


导出常量元数据='未定义';
 

`torch.nn.模块`


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 要调整的模型。
* **配置**
 （
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 ) — Lora 模型的配置。
* **适配器\_名称**
 （
 `str`
 ) — 适配器的名称，默认为
 `“默认”`
 。


退货


导出常量元数据='未定义';
 

`torch.nn.模块`


导出常量元数据='未定义';


劳拉模型。


从预训练的 Transformer 模型创建低阶适配器 (Lora) 模型。


 例子：



```
>>> from transformers import AutoModelForSeq2SeqLM
>>> from peft import LoraModel, LoraConfig

>>> config = LoraConfig(
...     task_type="SEQ\_2\_SEQ\_LM",
...     r=8,
...     lora_alpha=32,
...     target_modules=["q", "v"],
...     lora_dropout=0.01,
... )

>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> lora_model = LoraModel(model, config, "default")
```



```
>>> import transformers
>>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

>>> target_modules = ["q\_proj", "k\_proj", "v\_proj", "out\_proj", "fc\_in", "fc\_out", "wte"]
>>> config = LoraConfig(
...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL\_LM"
... )

>>> model = transformers.GPTJForCausalLM.from_pretrained(
...     "kakaobrain/kogpt",
...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
...     pad_token_id=tokenizer.eos_token_id,
...     use_cache=False,
...     device_map={"": rank},
...     torch_dtype=torch.float16,
...     load_in_8bit=True,
... )
>>> model = prepare_model_for_int8_training(model)
>>> lora_model = get_peft_model(model, config)
```


**属性**
 :


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 要调整的模型。
* **peft\_config**
 （
 [LoraConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.LoraConfig)
 )：Lora模型的配置。



#### 


添加加权适配器


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/model.py#L442)


（


 适配器


 重量


 适配器\_名称


 组合\_类型
 
 = 'svd'


svd\_rank
 
 = 无


svd\_clamp
 
 = 无


svd\_full\_矩阵
 
 = 正确


svd\_驱动程序
 
 = 无


）


 参数


* **适配器**
 （
 `列表`
 )—
要合并的适配器名称列表。
* **权重**
 （
 `列表`
 )—
每个适配器的重量列表。
* **适配器\_名称**
 （
 `str`
 )—
新适配器的名称。
* **组合\_类型**
 （
 `str`
 )—
合并类型。可以是以下之一[
 `svd`
 ,
 `线性`
 ,
 `猫`
 ]。当使用
 `猫`
 组合\_输入你
应该注意，生成的适配器的等级将等于所有适配器等级的总和。所以
混合适配器可能会变得太大并导致 OOM 错误。
* **svd\_rank**
 （
 `int`
 ,
 *选修的*
 )—
svd 输出适配器的等级。如果未提供，将使用合并适配器的最大等级。
* **svd\_clamp**
 （
 `浮动`
 ,
 *选修的*
 )—
用于钳制 SVD 分解输出的分位数阈值。如果没有提供，则不执行
夹紧。默认为无。
* **svd\_full\_矩阵**
 （
 `布尔`
 ,
 *选修的*
 )—
控制是计算完整的还是简化的 SVD，从而控制返回的形状
张量 U 和 Vh。默认为 True。
* **svd\_driver**
 （
 `str`
 ,
 *选修的*
 )—
要使用的 cuSOLVER 方法的名称。此关键字参数仅在 CUDA 上合并时有效。可
其中之一[无，
 `gesvd`
 ,
 `gesvdj`
 ,
 `gesvda`
 ]。欲了解更多信息，请参阅
 `torch.linalg.svd`
 文档。默认为无。


 此方法通过将给定的适配器与给定的权重合并来添加新的适配器。


当使用
 `猫`
 组合\_type您应该知道生成的适配器的等级将等于
所有适配器等级的总和。因此混合适配器可能会变得太大并导致 OOM
错误。




#### 


删除\_适配器


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/model.py#L652)


（


 适配器\_名称
 
 : 字符串


）


 参数


* **适配器\_名称**
 (str) — 要删除的适配器的名称。


 删除现有适配器。




#### 


合并\_和\_卸载


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/model.py#L674)


（


 进度条
 
 ：布尔=假


安全\_merge
 
 ：布尔=假


）


 参数


* **进度条**
 （
 `布尔`
 )—
是否显示进度条指示卸载和合并过程
* **安全\_合并**
 （
 `布尔`
 )—
是否激活安全合并检查来检查适配器中是否存在潜在的Nan
重量


 此方法将 LoRa 层合并到基础模型中。如果有人想使用基本模型，这是必需的
作为一个独立的模型。


 例子：



```
>>> from transformers import AutoModelForCausalLM
>>> from peft import PeftModel

>>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
>>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
>>> model = PeftModel.from_pretrained(base_model, peft_model_id)
>>> merged_model = model.merge_and_unload()
```


#### 


卸下


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/model.py#L700)


（
 

 ）


通过删除所有 lora 模块而不合并来恢复基本模型。这又恢复了原来的基础
模型。


### 


班级
 

 peft.tuners.lora。
 

 劳拉·莱尔


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/layer.py#L28)


（


 在\_功能中
 
 ：整数


输出\_特征
 
 ：整数


\*\*夸格


）


### 


班级
 

 peft.tuners.lora。
 

 线性


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/layer.py#L191)


（


 适配器\_名称
 
 : 字符串


在\_功能中
 
 ：整数


输出\_特征
 
 ：整数


r
 
 ：整数=0


劳拉\_alpha
 
 ：整数= 1


劳拉\_dropout
 
 ：浮动= 0.0


扇入_扇出_扇出
 
 ：布尔=假


是\_target\_conv\_1d\_layer
 
 ：布尔=假


\*\*夸格


）




#### 


获取\_delta\_权重


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/layer.py#L263)


（


 适配器


）


 参数


* **适配器**
 (str)—
应计算增量权重的适配器的名称。


 计算给定适配器的增量权重。




#### 


合并


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/lora/layer.py#L221)


（


 安全\_merge
 
 ：布尔=假


）


 参数


* **安全\_合并**
 （
 `布尔`
 ,
 *选修的*
 )—
如果为 True，合并操作将在原始权重的副本中执行并检查 NaN
在合并权重之前。如果您想检查合并操作是否会产生结果，这很有用
NaN。默认为
 ‘假’
 。


 将活动适配器权重合并到基本权重中


## P-调音




### 


班级
 

 佩夫特。
 

 提示编码器配置


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/p_tuning/config.py#L30)


（


 peft\_type
 
 : 打字.Union[str, peft.utils.peft\_types.PefType] = None


自动\_映射
 
 : 打字.可选[dict] =无


基本\_模型\_名称\_或\_路径
 
 ：str=无


修订
 
 ：str=无


任务\_类型
 
 : Typing.Union[str, peft.utils.peft\_types.TaskType] = None


推理\_模式
 
 ：布尔=假


虚拟令牌数
 
 ：整数=无


令牌\_dim
 
 ：整数=无


num\_transformer\_submodules
 
 : 打字.Optional[int] = None


num\_attention\_heads
 
 : 打字.Optional[int] = None


层数
 
 : 打字.Optional[int] = None


编码器\_重新参数化\_类型
 
 ：typing.Union[str, peft.tuners.p\_tuning.config.PromptEncoderReparameterizationType] = <PromptEncoderReparameterizationType.MLP: 'MLP'>


编码器\_hidden\_size
 
 ：整数=无


编码器\_num\_layers
 
 ：整数= 2


编码器\_dropout
 
 ：浮动= 0.0


）


 参数


* **编码器\_重新参数化\_类型**
 （联盟[
 `PromptEncoderReparameterizationType`
 ,
 `str`
 ])—
要使用的重新参数化类型。
* **编码器\_hidden\_size**
 （
 `int`
 ) — 提示编码器的隐藏大小。
* **编码器\_num\_layers**
 （
 `int`
 ) — 提示编码器的层数。
* **编码器\_dropout**
 （
 `浮动`
 ) — 提示编码器的丢失概率。


 这是存储配置的配置类
 [PromptEncoder](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEncoder)
 。




### 


班级
 

 佩夫特。
 

 提示编码器


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/p_tuning/model.py#L25)


（


 配置


）


 参数


* **配置**
 （
 [PromptEncoderConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEncoderConfig)
 ) — 提示编码器的配置。


 用于生成用于 p-tuning 的虚拟令牌嵌入的提示编码器网络。


 例子：



```
>>> from peft import PromptEncoder, PromptEncoderConfig

>>> config = PromptEncoderConfig(
...     peft_type="P\_TUNING",
...     task_type="SEQ\_2\_SEQ\_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     encoder_reparameterization_type="MLP",
...     encoder_hidden_size=768,
... )

>>> prompt_encoder = PromptEncoder(config)
```


**属性**
 :


* **嵌入**
 （
 `torch.nn.Embedding`
 ) — 提示编码器的嵌入层。
* **mlp\_head**
 （
 `torch.nn.Sequential`
 ) — 提示编码器的 MLP 头，如果
 `inference_mode=False`
 。
* **lstm\_head**
 （
 `torch.nn.LSTM`
 ) — 提示编码器的 LSTM 头，如果
 `inference_mode=False`
 和
 `encoder_reparameterization_type="LSTM"`
 。
* **令牌\_dim**
 （
 `int`
 ) — 基础变压器模型的隐藏嵌入维度。
* **输入\_大小**
 （
 `int`
 ) — 提示编码器的输入大小。
* **输出\_大小**
 （
 `int`
 ) — 提示编码器的输出大小。
* **隐藏\_size**
 （
 `int`
 ) — 提示编码器的隐藏大小。
* **总计\_虚拟\_代币**
 （
 `int`
 ): 虚拟代币总数
提示编码器。
* **编码器\_type**
 （联盟[
 `PromptEncoderReparameterizationType`
 ,
 `str`
 ]): 提示符的编码器类型
编码器。


输入形状：(
 `批量大小`
 ,
 `total_virtual_tokens`
 ）


输出形状：(
 `批量大小`
 ,
 `total_virtual_tokens`
 ,
 `token_dim`
 ）


## 前缀调整




### 


班级
 

 佩夫特。
 

 前缀调整配置


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/prefix_tuning/config.py#L23)


（


 peft\_type
 
 : 打字.Union[str, peft.utils.peft\_types.PefType] = None


自动\_映射
 
 : 打字.可选[dict] =无


基本\_模型\_名称\_或\_路径
 
 ：str=无


修订
 
 ：str=无


任务\_类型
 
 : Typing.Union[str, peft.utils.peft\_types.TaskType] = None


推理\_模式
 
 ：布尔=假


虚拟令牌数
 
 ：整数=无


令牌\_dim
 
 ：整数=无


num\_transformer\_submodules
 
 : 打字.Optional[int] = None


num\_attention\_heads
 
 : 打字.Optional[int] = None


层数
 
 : 打字.Optional[int] = None


编码器\_hidden\_size
 
 ：整数=无


前缀\_投影
 
 ：布尔=假


）


 参数


* **编码器\_hidden\_size**
 （
 `int`
 ) — 提示编码器的隐藏大小。
* **前缀\_投影**
 （
 `布尔`
 ) — 是否投影前缀嵌入。


 这是存储配置的配置类
 [PrefixEncoder](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PrefixEncoder)
 。




### 


班级
 

 佩夫特。
 

 前缀编码器


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/prefix_tuning/model.py#L21)


（


 配置


）


 参数


* **配置**
 （
 [PrefixTuningConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PrefixTuningConfig)
 ) — 前缀编码器的配置。


 这
 `torch.nn`
 模型对前缀进行编码。


 例子：



```
>>> from peft import PrefixEncoder, PrefixTuningConfig

>>> config = PrefixTuningConfig(
...     peft_type="PREFIX\_TUNING",
...     task_type="SEQ\_2\_SEQ\_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     encoder_hidden_size=768,
... )
>>> prefix_encoder = PrefixEncoder(config)
```


**属性**
 :


* **嵌入**
 （
 `torch.nn.Embedding`
 ) — 前缀编码器的嵌入层。
* **转换**
 （
 `torch.nn.Sequential`
 ) — 用于转换前缀嵌入的两层 MLP 如果
 `前缀投影`
 是
 '真实'
 。
* **前缀\_投影**
 （
 `布尔`
 ) — 是否投影前缀嵌入。


输入形状：(
 `批量大小`
 ,
 `num_virtual_tokens`
 ）


输出形状：(
 `批量大小`
 ,
 `num_virtual_tokens`
 ,
 `2*层*隐藏`
 ）


## 及时调整




### 


班级
 

 佩夫特。
 

 提示调整配置


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/prompt_tuning/config.py#L30)


（


 peft\_type
 
 : 打字.Union[str, peft.utils.peft\_types.PefType] = None


自动\_映射
 
 : 打字.可选[dict] =无


基本\_模型\_名称\_或\_路径
 
 ：str=无


修订
 
 ：str=无


任务\_类型
 
 : Typing.Union[str, peft.utils.peft\_types.TaskType] = None


推理\_模式
 
 ：布尔=假


虚拟令牌数
 
 ：整数=无


令牌\_dim
 
 ：整数=无


num\_transformer\_submodules
 
 : 打字.Optional[int] = None


num\_attention\_heads
 
 : 打字.Optional[int] = None


层数
 
 : 打字.Optional[int] = None


提示\_调整\_init
 
 : Typing.Union[peft.tuners.prompt\_tuning.config.PromptTuningInit, str] = <PromptTuningInit.RANDOM: 'RANDOM'>


提示\_调整\_init\_text
 
 : 打字.可选[str] =无


分词器\_name\_or\_path
 
 : 打字.可选[str] =无


）


 参数


* **提示\_tuning\_init**
 （联盟[
 `提示调整初始化`
 ,
 `str`
 ]) — 提示嵌入的初始化。
* **提示\_tuning\_init\_text**
 （
 `str`
 ,
 *选修的*
 )—
用于初始化提示嵌入的文本。仅在以下情况下使用
 `prompt_tuning_init`
 是
 `文本`
 。
* **分词器\_name\_or\_path**
 （
 `str`
 ,
 *选修的*
 )—
分词器的名称或路径。仅在以下情况下使用
 `prompt_tuning_init`
 是
 `文本`
 。


 这是存储配置的配置类
 [PromptEmbedding](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEmbedding)
 。




### 


班级
 

 佩夫特。
 

 提示嵌入


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/prompt_tuning/model.py#L23)


（


 配置


 词嵌入


）


 参数


* **配置**
 （
 [PromptTuningConfig](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptTuningConfig)
 ) — 提示嵌入的配置。
* **词\_嵌入**
 （
 `torch.nn.模块`
 ) — 基础 Transformer 模型的词嵌入。


 将虚拟令牌编码为提示嵌入的模型。


**属性**
 :


* **嵌入**
 （
 `torch.nn.Embedding`
 ) — 提示嵌入的嵌入层。


 例子：



```
>>> from peft import PromptEmbedding, PromptTuningConfig

>>> config = PromptTuningConfig(
...     peft_type="PROMPT\_TUNING",
...     task_type="SEQ\_2\_SEQ\_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     prompt_tuning_init="TEXT",
...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
...     tokenizer_name_or_path="t5-base",
... )

>>> # t5\_model.shared is the word embeddings of the base model
>>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
```


输入形状：(
 `批量大小`
 ,
 `total_virtual_tokens`
 ）


输出形状：(
 `批量大小`
 ,
 `total_virtual_tokens`
 ,
 `token_dim`
 ）


## IA3




### 


班级
 

 佩夫特。
 

 IA3配置


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/ia3/config.py#L24)


（


 peft\_type
 
 : 打字.Union[str, peft.utils.peft\_types.PefType] = None


自动\_映射
 
 : 打字.可选[dict] =无


基本\_模型\_名称\_或\_路径
 
 ：str=无


修订
 
 ：str=无


任务\_类型
 
 : Typing.Union[str, peft.utils.peft\_types.TaskType] = None


推理\_模式
 
 ：布尔=假


目标\_模块
 
 : 打字.Union[str, 打字.List[str], NoneType] = 无


前馈模块
 
 : 打字.Union[str, 打字.List[str], NoneType] = 无


扇入_扇出_扇出
 
 ：布尔=假


要保存的模块
 
 : 打字.可选[打字.列表[str]] =无


初始化\_ia3\_权重
 
 ：布尔=真


）


 参数


* **目标\_模块**
 （
 `联合[列表[str],str]`
 )—
要应用 (IA)^3 的模块的名称。
* **前馈\_模块**
 （
 `联合[列表[str],str]`
 )—
被视为前馈模块的模块的名称，如原始论文中所示。这些模块将
将 (IA)^3 向量乘以输入，而不是输出。 feedforward\_modules 必须是名称或
target\_modules 中存在的名称子集。
* **扇入_扇出_扇出**
 （
 `布尔`
 )—
如果要替换的层存储像 (fan\_in, fan\_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用
 `Conv1D`
 它存储像 (fan\_in, fan\_out) 这样的权重，因此应该设置为
 '真实'
 。
* **模块\_要\_保存**
 （
 `列表[str]`
 )—
除 (IA)^3 层之外的模块列表，要设置为可训练并保存在最终检查点中。
* **初始化\_ia3\_权重**
 （
 `布尔`
 )—
是否初始化(IA)^3层的向量，默认为
 '真实'
 。


 这是存储配置的配置类
 [IA3Model](/docs/peft/v0.6.2/en/package_reference/tuners#peft.IA3Model)
 。




### 


班级
 

 佩夫特。
 

 IA3模型


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/ia3/model.py#L45)


（


 模型


 配置


 适配器\_名称


）
 

 →


导出常量元数据='未定义';
 

`torch.nn.模块`


 参数


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 要调整的模型。
* **配置**
 （
 [IA3Config](/docs/peft/v0.6.2/en/package_reference/tuners#peft.IA3Config)
 ) — (IA)^3 模型的配置。
* **适配器\_名称**
 （
 `str`
 ) — 适配器的名称，默认为
 `“默认”`
 。


退货


导出常量元数据='未定义';
 

`torch.nn.模块`


导出常量元数据='未定义';


(IA)^3 模型。


通过抑制和放大预训练的内部激活 ((IA)^3) 模型来创建注入适配器
变压器模型。该方法详细描述于
 <https://arxiv.org/abs/2205.05638>


例子：



```
>>> from transformers import AutoModelForSeq2SeqLM, ia3Config
>>> from peft import IA3Model, IA3Config

>>> config = IA3Config(
...     peft_type="IA3",
...     task_type="SEQ\_2\_SEQ\_LM",
...     target_modules=["k", "v", "w0"],
...     feedforward_modules=["w0"],
... )

>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> ia3_model = IA3Model(config, model)
```


**属性**
 :


* **模型**
 （
 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.PreTrainedModel)
 ) — 要调整的模型。
* **peft\_config**
 （
 `ia3Config`
 ): (IA)^3 模型的配置。



#### 


合并\_和\_卸载


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/tuners/ia3/model.py#L300)


（


 安全\_merge
 
 ：布尔=假


）


 参数


* **安全\_合并**
 （
 `布尔`
 ,
 `可选的`
 ，默认为
 ‘假’
 )—
如果为 True，合并操作将在原始权重的副本中执行并检查 NaN
在合并权重之前。如果您想检查合并操作是否会产生结果，这很有用
NaN。默认为
 ‘假’
 。


 此方法将 (IA)^3 层合并到基础模型中。如果有人想使用基本模型，这是必需的
作为一个独立的模型。