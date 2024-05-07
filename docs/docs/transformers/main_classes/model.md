# 楷模

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/main_classes/model>
>
> 原始地址：<https://huggingface.co/docs/transformers/main_classes/model>


基类
 [PreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel)
 ,
 [TFPreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.TFPreTrainedModel)
 ， 和
 [FlaxPreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.FlaxPreTrainedModel)
 实现从本地加载/保存模型的常用方法
文件或目录，或来自库提供的预训练模型配置（从 HuggingFace 的 AWS 下载
S3 存储库）。


[PreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel)
 和
 [TFPreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.TFPreTrainedModel)
 还实现了一些方法
所有模型的共同点是：


* 当新标记添加到词汇表时调整输入标记嵌入的大小
* 修剪模型的注意力头。


每个模型通用的其他方法定义在
 [ModuleUtilsMixin](/docs/transformers/v4.35.2/en/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin)
 （对于 PyTorch 模型）和
 `~modeling_tf_utils.TFModuleUtilsMixin`
 （对于 TensorFlow 模型）或
对于文本生成，
 [GenerationMixin](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin)
 （对于 PyTorch 模型），
 [TFGenerationMixin](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.TFGenerationMixin)
 （对于 TensorFlow 模型）和
 [FlaxGenerationMixin](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.FlaxGenerationMixin)
 （对于 Flax/JAX 型号）。


## 预训练模型




### 


班级
 

 变压器。
 

 预训练模型


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L1071)



 (
 


 配置
 
 ：预训练配置


\*输入


 \*\*夸格




 )
 


所有模型的基类。


[PreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel)
 负责存储模型的配置并处理加载方法，
下载和保存模型以及所有模型通用的一些方法：


* 调整输入嵌入的大小，
* 修剪自注意力头中的头。


类属性（被派生类覆盖）：


* **配置\_class**
 （
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 ) — 的子类
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 用作配置类
对于这个模型架构。
* **加载\_tf\_权重**
 （
 `可调用`
 ) — 一条蟒蛇
 *方法*
 用于在 PyTorch 模型中加载 TensorFlow 检查点，
以参数：


+ **型号**
（
[PreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel)
) — 要加载 TensorFlow 检查点的模型实例。
+ **配置**
（
`预训练配置`
) — 与模型关联的配置实例。
+ **路径**
（
`str`
) — TensorFlow 检查点的路径。
* **基础\_模型\_前缀**
 （
 `str`
 ) — 一个字符串，指示与派生中的基本模型关联的属性
相同架构的类在基本模型之上添加模块。
* **可并行化**
 （
 `布尔`
 ) — 指示该模型是否支持模型并行化的标志。
* **主\_输入\_名称**
 （
 `str`
 ) — 模型主要输入的名称（通常
 `输入ID`
 自然语言处理
楷模，
 `像素值`
 对于视觉模型和
 `输入值`
 对于语音模型）。



#### 


推送\_到\_集线器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/hub.py#L791)



 (
 


 仓库\_id
 
 : 字符串


使用\_temp\_dir
 
 : 打字.Optional[bool] = None


提交\_消息
 
 : 打字.可选[str] =无


私人的
 
 : 打字.Optional[bool] = None


代币
 
 : 打字.Union[bool, str, NoneType] = None


最大分片大小
 
 : 打字.Union[int, str, NoneType] = '5GB'


创建\_pr
 
 ：布尔=假


安全\_序列化
 
 ：布尔=真


修订
 
 ：str=无


提交\_描述
 
 ：str=无


\*\*已弃用\_kwargs




 )
 


 参数


* **仓库\_id**
 （
 `str`
 )—
您要将模型推送到的存储库的名称。它应包含您的组织名称
当推送到给定组织时。
* **使用\_temp\_dir**
 （
 `布尔`
 ,
 *选修的*
 )—
是否使用临时目录来存储推送到 Hub 之前保存的文件。
将默认为
 '真实'
 如果没有类似名称的目录
 `repo_id`
 ,
 ‘假’
 否则。
* **提交\_消息**
 （
 `str`
 ,
 *选修的*
 )—
推送时要提交的消息。将默认为
 `“上传模型”`
 。
* **私人的**
 （
 `布尔`
 ,
 *选修的*
 )—
创建的存储库是否应该是私有的。
* **代币**
 （
 `布尔`
 或者
 `str`
 ,
 *选修的*
 )—
用作远程文件的 HTTP 承载授权的令牌。如果
 '真实'
 ，将使用生成的令牌
跑步时
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。将默认为
 '真实'
 如果
 `repo_url`
 没有指定。
* **最大\_分片\_大小**
 （
 `int`
 或者
 `str`
 ,
 *选修的*
 ，默认为
 `“5GB”`
 )—
仅适用于型号。分片之前检查点的最大大小。检查点分片
那么每个尺寸都会小于这个尺寸。如果表示为字符串，则后面需要跟数字
按一个单位（例如
 `“5MB”`
 ）。我们默认为
 `“5GB”`
 以便用户可以轻松地在免费层上加载模型
Google Colab 实例没有任何 CPU OOM 问题。
* **创建\_pr**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否使用上传的文件创建 PR 或直接提交。
* **安全\_序列化**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
是否将模型权重转换为安全张量格式以实现更安全的序列化。
* **修订**
 （
 `str`
 ,
 *选修的*
 )—
将上传的文件推送到的分支。
* **提交\_描述**
 （
 `str`
 ,
 *选修的*
 )—
将创建的提交的描述


 将模型文件上传到🤗模型中心。


 例子：



```
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")

# Push the model to your namespace with the name "my-finetuned-bert".
model.push_to_hub("my-finetuned-bert")

# Push the model to an organization with the name "my-finetuned-bert".
model.push_to_hub("huggingface/my-finetuned-bert")
```


#### 


可以\_生成


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L1234)


（
 

 ）
 

 →


导出常量元数据='未定义';
 

`布尔`


退货


导出常量元数据='未定义';
 

`布尔`


导出常量元数据='未定义';


该模型是否可以生成序列
 `.generate()`
 。


返回该模型是否可以生成序列
 `.generate()`
 。




#### 


禁用\_input\_require\_grads


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L1337)


（
 

 ）


删除
 `_require_grads_hook`
 。




#### 


启用\_输入\_需要\_grads


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L1326)


（
 

 ）


启用输入嵌入的渐变。这对于微调适配器重量同时保持
模型权重固定。




#### 


来自\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L2293)



 (
 


 预训练\_model\_name\_or\_path
 
 : 打字.Union[str, os.PathLike, NoneType]


\*模型\_args


 配置
 
 : 打字.Union[transformers.configuration\_utils.PretrainedConfig, str, os.PathLike, NoneType] = None


缓存\_dir
 
 : 打字.Union[str, os.PathLike, NoneType] = None


忽略\_不匹配\_尺寸
 
 ：布尔=假


强制\_下载
 
 ：布尔=假


仅本地\_文件\_
 
 ：布尔=假


代币
 
 : 打字.Union[str, bool, NoneType] = None


修订
 
 : str = '主'


使用\_safetensors
 
 : 布尔 = 无


\*\*夸格




 )
 


 参数


* **预训练\_model\_name\_or\_path**
 （
 `str`
 或者
 `os.PathLike`
 ,
 *选修的*
 )—
可以是：
 
+ 一个字符串，
*型号编号*
托管在 Huggingface.co 上的模型存储库内的预训练模型。
有效的模型 ID 可以位于根级别，例如
`bert-base-uncased`
，或命名空间下
用户或组织名称，例如
`dbmdz/bert-base-德语-大小写`
。
+ 通往a的路径
*目录*
包含使用保存的模型权重
[save\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)
，例如，
`./my_model_directory/`
。
+ 路径或 url
*张量流索引检查点文件*
（例如，
`./tf_model/model.ckpt.index`
）。在
这个案例，
`来自_tf`
应设置为
'真实'
并且应该提供一个配置对象
`配置`
争论。此加载路径比在 a 中转换 TensorFlow 检查点要慢
PyTorch 模型使用提供的转换脚本并随后加载 PyTorch 模型。
+ 包含模型文件夹的路径或 URL
*亚麻检查点文件*
在
*.msgpack*
格式（例如，
`./flax_model/`
含有
`flax_model.msgpack`
）。在这种情况下，
`from_flax`
应设置为
'真实'
。
+ `无`
如果您同时提供配置和状态字典（分别使用关键字
论点
`配置`
和
`state_dict`
）。
* **模型\_args**
 （位置参数的序列，
 *选修的*
 )—
所有剩余的位置参数将传递给底层模型的
 `__init__`
 方法。
* **配置**
 （
 `Union[PretrainedConfig, str, os.PathLike]`
 ,
 *选修的*
 )—
可以是：
 
+ 派生类的实例
[PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
,
+ 作为输入有效的字符串或路径
[来自_pretrained()](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)
。

 模型使用的配置，而不是自动加载的配置。配置还可以
在以下情况下自动加载：


+ 该模型是库提供的模型（加载了
*型号编号*
预训练的字符串
模型）。
+ 模型保存使用
[save\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)
并通过提供重新加载
保存目录。
+ 通过提供本地目录来加载模型
`预训练模型名称或路径`
和一个
配置 JSON 文件名为
*配置.json*
可以在目录中找到。
* **状态\_dict**
 （
 `字典[str, torch.Tensor]`
 ,
 *选修的*
 )—
使用状态字典代替从保存的权重文件加载的状态字典。
 
 如果您想从预训练配置创建模型但加载您自己的模型，则可以使用此选项
重量。但在这种情况下，您应该检查是否使用
 [save\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)
 和
 [来自\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
 这不是一个更简单的选择。
* **缓存\_dir**
 （
 `Union[str, os.PathLike]`
 ,
 *选修的*
 )—
如果是，则应缓存下载的预训练模型配置的目录路径
不应使用标准缓存。
* **来自\_tf**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
从 TensorFlow 检查点保存文件加载模型权重（请参阅文档字符串
 `预训练模型名称或路径`
 争论）。
* **来自\_flax**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
从 Flax 检查点保存文件加载模型权重（请参阅文档字符串
 `预训练模型名称或路径`
 争论）。
* **忽略\_不匹配\_尺寸**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
如果检查点的某些权重大小不同，是否引发错误
作为模型的权重（例如，如果您正在实例化一个具有 10 个标签的模型）
有 3 个标签的检查点）。
* **强制\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否强制（重新）下载模型权重和配置文件，覆盖
缓存版本（如果存在）。
* **继续\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否删除接收不完整的文件。如果出现这种情况，将尝试恢复下载
文件已存在。
* **代理**
 （
 `字典[str, str]`
 ,
 *选修的*
 )—
由协议或端点使用的代理服务器字典，例如，
 `{'http': 'foo.bar:3128', 'http://主机名': 'foo.bar:4012'}`
 。每个请求都会使用代理。
* **输出\_加载\_信息(
 `布尔`
 ,**
*选修的*
 ，默认为
 ‘假’
 )—
是否还返回包含丢失键、意外键和错误消息的字典。
* **本地\_文件\_only(
 `布尔`
 ,**
*选修的*
 ，默认为
 ‘假’
 )—
是否仅查看本地文件（即不尝试下载模型）。
* **代币**
 （
 `str`
 或者
 `布尔`
 ,
 *选修的*
 )—
用作远程文件的 HTTP 承载授权的令牌。如果
 '真实'
 ，或未指定，将使用
运行时生成的token
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **修订**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“主要”`
 )—
要使用的具体型号版本。它可以是分支名称、标签名称或提交 ID，因为我们使用
基于 git 的系统，用于在 Huggingface.co 上存储模型和其他工件，因此
 `修订`
 可以是任何
git 允许的标识符。
 

 要测试您在 Hub 上发出的拉取请求，您可以传递 `revision=”refs/pr/
 
 ”。
* **镜子**
 （
 `str`
 ,
 *选修的*
 )—
镜像源加速中国下载。如果您来自中国并且可以访问
问题，您可以设置此选项来解决它。请注意，我们不保证及时性或安全性。
请参阅镜像站点以获取更多信息。
* **\_fast\_init(
 `布尔`
 ,**
*选修的*
 ，默认为
 '真实'
 )—
是否禁用快速初始化。
 

 应该只禁用
 *\_fast\_init*
 以确保向后兼容
 `变形金刚.__版本__ < 4.6.0`
 用于种子模型初始化。该参数将在下一个主要版本中删除。看
 [拉取请求 11471](https://github.com/huggingface/transformers/pull/11471)
 了解更多信息。


大模型推理参数