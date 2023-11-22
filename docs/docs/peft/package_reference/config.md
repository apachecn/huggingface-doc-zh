# 配置

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/peft/package_reference/config>
>
> 原始地址：<https://huggingface.co/docs/peft/package_reference/config>


配置类存储了配置
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 、PEFT 适配器型号以及配置
 `前缀调整`
 ,
 `及时调整`
 ， 和
 [PromptEncoder](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEncoder)
 。它们包含从 Hub 保存和加载模型配置的方法，指定要使用的 PEFT 方法、要执行的任务类型以及模型配置（例如层数和注意力头数量）。


## 佩夫特配置混合




### 


班级
 

 pft.config。
 

 佩夫特配置混合


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/config.py#L28)


（


 peft\_type
 
 : 打字.可选[peft.utils.peft\_types.PefType] = 无


自动\_映射
 
 : 打字.可选[dict] =无


）


 参数


* **peft\_type**
 （联盟[
 `~peft.utils.config.PefType`
 ,
 `str`
 ]) — 要使用的 Peft 方法的类型。


 这是 PEFT 适配器模型的基本配置类。它包含了所有通用的方法
PEFT 适配器型号。这个类继承自
 [PushToHubMixin](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/model#transformers.utils.PushToHubMixin)
 其中包含方法
将您的模型推送到 Hub。方法
 `保存预训练`
 将把适配器模型的配置保存在
目录。方法
 `from_pretrained`
 将从目录加载适配器模型的配置。



#### 


来自\_json\_file


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/config.py#L137)


（


 路径\_json\_file
 
 : 字符串


\*\*夸格


）


 参数


* **路径\_json\_file**
 （
 `str`
 )—
json 文件的路径。


 从 json 文件加载配置文件。




#### 


来自\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/config.py#L79)


（


 预训练\_model\_name\_or\_path
 
 : 字符串


子文件夹
 
 : 打字.可选[str] =无


\*\*夸格


）


 参数


* **预训练\_model\_name\_or\_path**
 （
 `str`
 )—
保存配置的目录或集线器存储库 ID。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
传递给子类初始化的附加关键字参数。


 此方法从目录加载适配器模型的配置。




#### 


保存\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/config.py#L46)


（


 保存\_目录
 
 : 字符串


\*\*夸格


）


 参数


* **保存\_目录**
 （
 `str`
 )—
将保存配置的目录。
* **夸格斯**
 （额外的关键字参数，
 *选修的*
 )—
传递给的附加关键字参数
 [push\_to\_hub](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub)
 方法。


 此方法将适配器模型的配置保存在目录中。


## 佩夫特配置




### 


班级
 

 佩夫特。
 

 佩夫特配置


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/config.py#L206)


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


）


 参数


* **peft\_type**
 （联盟[
 `~peft.utils.config.PefType`
 ,
 `str`
 ]) — 要使用的 Peft 方法的类型。
* **任务\_类型**
 （联盟[
 `~peft.utils.config.TaskType`
 ,
 `str`
 ]) — 要执行的任务类型。
* **推理\_模式**
 （
 `布尔`
 ，默认为
 ‘假’
 ) — 是否在推理模式下使用 Peft 模型。


 这是存储配置的基本配置类
 [PeftModel](/docs/peft/v0.6.2/en/package_reference/peft_model#peft.PeftModel)
 。


## 提示学习配置




### 


班级
 

 佩夫特。
 

 提示学习配置


[<
 

 来源
 

 >](https://github.com/huggingface/peft/blob/v0.6.2/src/peft/config.py#L224)


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


）


 参数


* **num\_virtual\_tokens**
 （
 `int`
 ) — 要使用的虚拟代币数量。
* **令牌\_dim**
 （
 `int`
 ) — 基础变压器模型的隐藏嵌入维度。
* **num\_transformer\_submodules**
 （
 `int`
 ) — 基础变压器模型中变压器子模块的数量。
* **num\_attention\_heads**
 （
 `int`
 ) — 基础 Transformer 模型中注意力头的数量。
* **num\_layers**
 （
 `int`
 ) — 基础变压器模型中的层数。


 这是存储配置的基本配置类
 `前缀调整`
 ,
 [PromptEncoder](/docs/peft/v0.6.2/en/package_reference/tuners#peft.PromptEncoder)
 ， 或者
 `及时调整`
 。