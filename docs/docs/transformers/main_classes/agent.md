# 代理和工具

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/main_classes/agent>
>
> 原始地址：<https://huggingface.co/docs/transformers/main_classes/agent>


Transformers Agents 是一个实验性 API，可能随时更改。代理返回的结果
由于 API 或底层模型容易发生变化，因此可能会有所不同。


要了解有关代理和工具的更多信息，请务必阅读
 [入门指南](../transformers_agents)
 。这一页
包含底层类的 API 文档。


## 代理商



我们提供三种类型的代理：
 [HfAgent](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.HfAgent)
 使用开源模型的推理端点，
 [LocalAgent](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.LocalAgent)
 在本地使用您选择的模型并且
 [OpenAiAgent](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.OpenAiAgent)
 使用 OpenAI 封闭模型。


### 


 Hf代理



### 


班级
 

 变压器。
 

 Hf代理


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L588)



 (
 


 url\_端点


 代币
 
 = 无


聊天\_提示\_模板
 
 = 无


运行\_提示\_模板
 
 = 无


额外\_工具
 
 = 无



 )
 


 参数


* **url\_端点**
 （
 `str`
 )—
要使用的 url 端点的名称。
* **代币**
 （
 `str`
 ,
 *选修的*
 )—
用作远程文件的 HTTP 承载授权的令牌。如果未设置，将使用时生成的令牌
跑步
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **聊天\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `聊天`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `chat_prompt_template.txt`
 在本例中的此回购协议中。
* **运行\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `运行`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `run_prompt_template.txt`
 在本例中的此回购协议中。
* **额外\_工具**
 （
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 、工具列表或带有工具值的字典，
 *选修的*
 )—
除了默认工具之外还包含任何其他工具。如果您传递一个同名的工具
默认工具之一，该默认工具将被覆盖。


 使用推理端点生成代码的代理。


 例子：



```
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
```




### 


 本地代理



### 


班级
 

 变压器。
 

 本地代理


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L659)



 (
 


 模型


 分词器


 聊天\_提示\_模板
 
 = 无


运行\_提示\_模板
 
 = 无


额外\_工具
 
 = 无



 )
 


 参数


* **模型**
 （
 [PreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel)
 )—
用于代理的模型。
* **标记器**
 （
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 )—
用于代理的分词器。
* **聊天\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `聊天`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `chat_prompt_template.txt`
 在本例中的此回购协议中。
* **运行\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `运行`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `run_prompt_template.txt`
 在本例中的此回购协议中。
* **额外\_工具**
 （
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 、工具列表或带有工具值的字典，
 *选修的*
 )—
除了默认工具之外还包含任何其他工具。如果您传递一个同名的工具
默认工具之一，该默认工具将被覆盖。


 使用本地模型和分词器生成代码的代理。


 例子：



```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LocalAgent

checkpoint = "bigcode/starcoder"
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

agent = LocalAgent(model, tokenizer)
agent.run("Draw me a picture of rivers and lakes.")
```




#### 


来自\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L704)



 (
 


 预训练\_model\_name\_or\_path


 \*\*夸格




 )
 


 参数


* **预训练\_model\_name\_or\_path**
 （
 `str`
 或者
 `os.PathLike`
 )—
集线器上存储库的名称或包含模型和标记生成器的文件夹的本地路径。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
关键字参数传递给
 [来自\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
 。


 构建一个便捷的方法
 `本地代理`
 来自预先训练的检查点。


 例子：



```
import torch
from transformers import LocalAgent

agent = LocalAgent.from_pretrained("bigcode/starcoder", device_map="auto", torch_dtype=torch.bfloat16)
agent.run("Draw me a picture of rivers and lakes.")
```


### 


 开放人工智能代理



### 


班级
 

 变压器。
 

 开放人工智能代理


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L364)



 (
 


 模型
 
 = '文本-达芬奇-003'


api\_key
 
 = 无


聊天\_提示\_模板
 
 = 无


运行\_提示\_模板
 
 = 无


额外\_工具
 
 = 无



 )
 


 参数


* **模型**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“文本-达芬奇-003”`
 )—
要使用的 OpenAI 模型的名称。
* **api\_key**
 （
 `str`
 ,
 *选修的*
 )—
要使用的 API 密钥。如果未设置，将查找环境变量
 `“OPENAI_API_KEY”`
 。
* **聊天\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `聊天`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `chat_prompt_template.txt`
 在本例中的此回购协议中。
* **运行\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `运行`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `run_prompt_template.txt`
 在本例中的此回购协议中。
* **额外\_工具**
 （
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 、工具列表或带有工具值的字典，
 *选修的*
 )—
除了默认工具之外还包含任何其他工具。如果您传递一个同名的工具
默认工具之一，该默认工具将被覆盖。


 使用 openai API 生成代码的代理。


openAI 模型用于生成模式，因此即使对于
 `聊天()`
 API，最好使用这样的模型
 `“文本-达芬奇-003”`
 通过 chat-GPT 变体。对聊天 GPT 模型的适当支持将在下一个版本中提供。


例子：



```
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key=xxx)
agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
```




### 


 AzureOpenAiAgent



### 


班级
 

 变压器。
 

 AzureOpenAiAgent


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L462)



 (
 


 部署\_id


 api\_key
 
 = 无


资源\_名称
 
 = 无


api\_版本
 
 ='2022-12-01'


是\_聊天\_模型
 
 = 无


聊天\_提示\_模板
 
 = 无


运行\_提示\_模板
 
 = 无


额外\_工具
 
 = 无



 )
 


 参数


* **部署\_id**
 （
 `str`
 )—
要使用的已部署的 Azure openAI 模型的名称。
* **api\_key**
 （
 `str`
 ,
 *选修的*
 )—
要使用的 API 密钥。如果未设置，将查找环境变量
 `“AZURE_OPENAI_API_KEY”`
 。
* **资源\_名称**
 （
 `str`
 ,
 *选修的*
 )—
Azure OpenAI 资源的名称。如果未设置，将查找环境变量
 `“AZURE_OPENAI_RESOURCE_NAME”`
 。
* **api\_版本**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“2022-12-01”`
 )—
用于此代理的 API 版本。
* **是\_聊天\_模式**
 （
 `布尔`
 ,
 *选修的*
 )—
无论您使用的是完成模型还是聊天模型（请参阅上面的注释，聊天模型都不会像
高效的）。将默认为
 `gpt`
 处于
 `部署_id`
 或不。
* **聊天\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `聊天`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `chat_prompt_template.txt`
 在本例中的此回购协议中。
* **运行\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `运行`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `run_prompt_template.txt`
 在本例中的此回购协议中。
* **额外\_工具**
 （
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 、工具列表或带有工具值的字典，
 *选修的*
 )—
除了默认工具之外还包含任何其他工具。如果您传递一个同名的工具
默认工具之一，该默认工具将被覆盖。


 使用 Azure OpenAI 生成代码的代理。请参阅
 [官方的
文档](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
 了解如何部署 openAI
Azure 上的模型


openAI 模型用于生成模式，因此即使对于
 `聊天()`
 API，最好使用这样的模型
 `“文本-达芬奇-003”`
 通过 chat-GPT 变体。对聊天 GPT 模型的适当支持将在下一个版本中提供。


例子：



```
from transformers import AzureOpenAiAgent

agent = AzureAiAgent(deployment_id="Davinci-003", api_key=xxx, resource_name=yyy)
agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
```




### 


 代理人



### 


班级
 

 变压器。
 

 代理人


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L196)



 (
 


 聊天\_提示\_模板
 
 = 无


运行\_提示\_模板
 
 = 无


额外\_工具
 
 = 无



 )
 


 参数


* **聊天\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `聊天`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `chat_prompt_template.txt`
 在本例中的此回购协议中。
* **运行\_提示\_模板**
 （
 `str`
 ,
 *选修的*
 )—
如果您想覆盖默认模板，请传递您自己的提示
 `运行`
 方法。可以是
实际提示模板或存储库 ID（在 Hugging Face Hub 上）。提示符应该位于名为
 `run_prompt_template.txt`
 在本例中的此回购协议中。
* **额外\_工具**
 （
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 、工具列表或带有工具值的字典，
 *选修的*
 )—
除了默认工具之外还包含任何其他工具。如果您传递一个同名的工具
默认工具之一，该默认工具将被覆盖。


 所有代理的基类，包含主要 API 方法。



#### 


聊天


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L268)



 (
 


 任务


 返回\_code
 
 = 假


偏僻的
 
 = 假


\*\*夸格




 )
 


 参数


* **任务**
 （
 `str`
 ) — 要执行的任务
* **返回\_code**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否只返回代码而不评估它。
* **偏僻的**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否使用远程工具（推理端点）而不是本地工具。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
评估代码时发送给代理的任何关键字参数。


 在聊天中向客服人员发送新请求。将使用其历史记录中的先前内容。


 例子：



```
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.chat("Draw me a picture of rivers and lakes")

agent.chat("Transform the picture so that there is a rock in there")
```


#### 


跑步


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L318)



 (
 


 任务


 返回\_code
 
 = 假


偏僻的
 
 = 假


\*\*夸格




 )
 


 参数


* **任务**
 （
 `str`
 ) — 要执行的任务
* **返回\_code**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否只返回代码而不评估它。
* **偏僻的**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否使用远程工具（推理端点）而不是本地工具。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
评估代码时发送给代理的任何关键字参数。


 向代理发送请求。


 例子：



```
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.run("Draw me a picture of rivers and lakes")
```


#### 


准备\_新\_聊天


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agents.py#L310)


（
 

 ）


清除之前调用的历史记录
 [聊天（）]（/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Agent.chat）
 。


## 工具



### 


 加载\_工具



#### 


变压器.load\_tool


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L643)



 (
 


 任务\_或\_repo\_id


 型号\_repo\_id
 
 = 无


偏僻的
 
 = 假


代币
 
 = 无


\*\*夸格




 )
 


 参数


* **任务\_或\_repo\_id**
 （
 `str`
 )—
要在 Hub 上加载工具或工具的存储库 ID 的任务。在 Transformers 中实现的任务
是：
 
+ `“文档问答”`
+ `“图像字幕”`
+ `“图像问答”`
+ `“图像分割”`
+ `“语音转文本”`
+ `“总结”`
+ `“文本分类”`
+ `“文字问答”`
+ `“文本转语音”`
+ `“翻译”`
* **模型\_repo\_id**
 （
 `str`
 ,
 *选修的*
 )—
使用此参数可为所选工具使用与默认模型不同的模型。
* **偏僻的**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否通过下载模型或（如果可用）通过推理端点来使用您的工具。
* **代币**
 （
 `str`
 ,
 *选修的*
 )—
在 hf.co 上识别您身份的令牌。如果未设置，将使用运行时生成的令牌
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
其他关键字参数将分为两部分：与 Hub 相关的所有参数（例如
 `缓存目录`
 ,
 `修订`
 ,
 `子文件夹`
 ) 将在下载您的工具和其他工具的文件时使用
将被传递给它的 init。


 主要功能是快速加载工具，无论是在 Hub 上还是在 Transformers 库中。



### 


 工具



### 


班级
 

 变压器。
 

 工具


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L81)



 (
 


 \*参数


 \*\*夸格




 )
 


代理使用的函数的基类。子类化并实现
 `__call__`
 方法以及
以下类属性：


* **描述**
 （
 `str`
 ) — 简短描述您的工具的功能、期望的输入和输出
将返回。例如“这是一个从某个地方下载文件的工具”
 `网址`
 。它需要
 `网址`
 作为输入，并且
返回文件中包含的文本”。
* **姓名**
 （
 `str`
 ) — 将在向代理提示中为您的工具使用的执行名称。例如
 `“文本分类器”`
 或者
 `“图像生成器”`
 。
* **输入**
 （
 `列表[str]`
 ) — 预期输入的模式列表（与调用中的顺序相同）。
方式应该是
 `“文本”`
 ,
 `“图像”`
 或者
 `“音频”`
 。这仅由以下人员使用
 `launch_gradio_demo`
 或做一个
距离你的工具有很大的空间。
* **输出**
 （
 `列表[str]`
 ) — 返回的模式列表，但工具（与返回的顺序相同）
调用方法）。方式应该是
 `“文本”`
 ,
 `“图像”`
 或者
 `“音频”`
 。这仅由以下人员使用
 `launch_gradio_demo`
 或者用你的工具创造一个漂亮的空间。


您还可以重写该方法
 [setup()](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool.setup)
 如果您的工具在使用之前执行起来是一项昂贵的操作
可用（例如加载模型）。
 [setup()](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool.setup)
 将在您第一次使用工具时被调用，但不会在
实例化。



#### 


来自\_gradio


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L330)



 (
 


 渐变\_工具




 )
 


创建一个
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 来自渐变工具。




#### 


来自\_hub


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L176)



 (
 


 仓库\_id
 
 : 字符串


型号\_repo\_id
 
 : 打字.可选[str] =无


代币
 
 : 打字.可选[str] =无


偏僻的
 
 ：布尔=假


\*\*夸格




 )
 


 参数


* **仓库\_id**
 （
 `str`
 )—
定义您的工具的 Hub 上的存储库的名称。
* **模型\_repo\_id**
 （
 `str`
 ,
 *选修的*
 )—
如果您的工具使用模型并且您想使用与默认模型不同的模型，则可以传递第二个模型
该参数的存储库 ID 或端点 url。
* **代币**
 （
 `str`
 ,
 *选修的*
 )—
在 hf.co 上识别您身份的令牌。如果未设置，将使用运行时生成的令牌
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **偏僻的**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否通过下载模型或（如果可用）通过推理端点来使用您的工具。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
其他关键字参数将分为两部分：与 Hub 相关的所有参数（例如
 `缓存目录`
 ,
 `修订`
 ,
 `子文件夹`
 ) 将在下载工具文件时使用，并且
其他的将被传递给它的 init。


 加载在 Hub 上定义的工具。




#### 


推送\_到\_集线器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L286)



 (
 


 仓库\_id
 
 : 字符串


提交\_消息
 
 : str = '上传工具'


私人的
 
 : 打字.Optional[bool] = None


代币
 
 : 打字.Union[str, bool, NoneType] = None


创建\_pr
 
 ：布尔=假



 )
 


 参数


* **仓库\_id**
 （
 `str`
 )—
您想要将工具推送到的存储库的名称。它应包含您的组织名称
推送到给定的组织。
* **提交\_消息**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“上传工具”`
 )—
推送时要提交的消息。
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
用作远程文件的 HTTP 承载授权的令牌。如果未设置，将使用生成的令牌
跑步时
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **创建\_pr**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否使用上传的文件创建 PR 或直接提交。


 将工具上传到 Hub。




#### 


节省


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L122)



 (
 


 输出\_dir




 )
 


 参数


* **输出\_dir**
 （
 `str`
 ) — 您要保存工具的文件夹。


 保存工具的相关代码文件，以便将其推送到集线器。这将复制您的代码
工具在
 `输出目录`
 以及自动生成：


* 一个名为的配置文件
 `tool_config.json`
* 一个
 `app.py`
 文件，以便您的工具可以转换为空格
* A
 `需求.txt`
 包含您的工具使用的模块的名称（在检查其模块时检测到）
代码）


您应该只使用此方法来保存在单独模块中定义的工具（而不是
 `__main__`
 ）。




#### 


设置


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L115)


（
 

 ）


对于任何昂贵且需要在开始使用之前执行的操作，请在此处覆盖此方法
你的工具。比如加载一个大模型。




### 


 管道工具



### 


班级
 

 变压器。
 

 管道工具


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L433)



 (
 


 模型
 
 = 无


预处理器
 
 = 无


后处理器
 
 = 无


设备
 
 = 无


设备\_map
 
 = 无


模型\_kwargs
 
 = 无


代币
 
 = 无


\*\*集线器\_kwargs




 )
 


 参数


* **模型**
 （
 `str`
 或者
 [PreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel)
 ,
 *选修的*
 )—
用于模型或实例化模型的检查点的名称。如果未设置，将默认为
类属性的值
 `默认检查点`
 。
* **预处理器**
 （
 `str`
 或者
 `任何`
 ,
 *选修的*
 )—
用于预处理器的检查点的名称，或实例化的预处理器（可以是
分词器、图像处理器、特征提取器或处理器）。将默认为以下值
 `模型`
 如果
未设置。
* **后\_处理器**
 （
 `str`
 或者
 `任何`
 ,
 *选修的*
 )—
用于后处理器或实例化预处理器的检查点的名称（可以是
分词器、图像处理器、特征提取器或处理器）。将默认为
 `预处理器`
 如果
未设置。
* **设备**
 （
 `int`
 ,
 `str`
 或者
 `火炬设备`
 ,
 *选修的*
 )—
执行模型的设备。将默认使用任何可用的加速器（GPU、MPS 等），
否则CPU。
* **设备\_map**
 （
 `str`
 或者
 `字典`
 ,
 *选修的*
 )—
如果传递，将用于实例化模型。
* **模型\_kwargs**
 （
 `字典`
 ,
 *选修的*
 )—
要发送到模型实例化的任何关键字参数。
* **代币**
 （
 `str`
 ,
 *选修的*
 )—
用作远程文件的 HTTP 承载授权的令牌。如果未设置，将使用时生成的令牌
跑步
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **中心\_kwargs**
 （附加关键字参数，
 *选修的*
 )—
发送到将从 Hub 加载数据的方法的任何附加关键字参数。


 A
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 针对 Transformer 模型量身定制。在基类的类属性之上
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 ， 你会
需要指定：


* **型号\_class**
 （
 `类型`
 ) — 用于在此工具中加载模型的类。
* **默认\_检查点**
 （
 `str`
 ) — 当用户未指定检查点时应使用的默认检查点。
* **预\_处理器\_类**
 （
 `类型`
 ,
 *选修的*
 ，默认为
 [自动处理器](/docs/transformers/v4.35.2/en/model_doc/auto#transformers.AutoProcessor)
 ) — 用于加载的类
预处理器
* **后\_处理器\_class**
 （
 `类型`
 ,
 *选修的*
 ，默认为
 [自动处理器](/docs/transformers/v4.35.2/en/model_doc/auto#transformers.AutoProcessor)
 ) — 用于加载的类
后处理器（当与预处理器不同时）。



#### 


解码


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L552)



 (
 


 输出




 )
 


使用
 `后处理器`
 解码模型输出。




#### 


编码


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L539)



 (
 


 原始输入




 )
 


使用
 `预处理器`
 准备输入
 `模型`
 。




#### 


向前


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L545)



 (
 


 输入




 )
 


通过发送输入
 `模型`
 。




#### 


设置


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L513)


（
 

 ）


实例化
 `预处理器`
 ,
 `模型`
 和
 `后处理器`
 如果需要的话。




### 


 远程工具



### 


班级
 

 变压器。
 

 远程工具


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L346)



 (
 


 端点\_url
 
 = 无


代币
 
 = 无


工具\_类
 
 = 无



 )
 


 参数


* **端点\_url**
 （
 `str`
 ,
 *选修的*
 )—
要使用的端点的 url。
* **代币**
 （
 `str`
 ,
 *选修的*
 )—
用作远程文件的 HTTP 承载授权的令牌。如果未设置，将使用时生成的令牌
跑步
 `huggingface-cli 登录`
 （存储在
 `~/.huggingface`
 ）。
* **工具\_class**
 （
 `类型`
 ,
 *选修的*
 )—
相应的
 `工具类`
 如果这是现有工具的远程版本。将有助于确定何时
输出应转换为另一种类型（如图像）。


 A
 [工具](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.Tool)
 它将向推理端点发出请求。



#### 


提取\_输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L409)



 (
 


 输出




 )
 


您可以在您的自定义类中重写此方法
 [RemoteTool](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.RemoteTool)
 应用一些自定义后处理
端点的输出。




#### 


准备\_输入


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L366)



 (
 


 \*参数


 \*\*夸格




 )
 


准备 HTTP 客户端向端点发送数据时收到的输入。位置参数将是
与签名者的签名相匹配
 `工具类`
 如果它是在实例时提供的。图像将被编码为
字节。


您可以在您的自定义类中重写此方法
 [RemoteTool](/docs/transformers/v4.35.2/en/main_classes/agent#transformers.RemoteTool)
 。




### 


 启动\_gradio\_demo



#### 


Transformers.launch\_gradio\_demo


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/base.py#L573)



 (
 


 工具\_类
 
 ： 工具



 )
 


 参数


* **工具\_class**
 （
 `类型`
 ) — 要启动演示的工具的类。


 启动工具的渐变演示。对应的工具类需要正确实现类属性
 `输入`
 和
 `输出`
 。


## 代理类型



代理可以处理工具之间的任何类型的对象；工具是完全多式联运的，可以接受并返回
文本、图像、音频、视频等类型。为了增加工具之间的兼容性，以及
在 ipython 中正确渲染这些返回（jupyter、colab、ipython 笔记本等），我们实现包装类
围绕这些类型。


包装的对象应该继续像最初一样运行；文本对象仍应表现为字符串、图像
对象仍应表现为
 `PIL.Image`
 。


这些类型具有三个特定目的：


* 呼叫
 `to_raw`
 类型上应该返回底层对象
* 呼叫
 `to_string`
 类型上应该以字符串形式返回对象：在以下情况下可以是字符串
 `代理文本`
 但将是其他实例中对象的序列化版本的路径
* 在 ipython 内核中显示它应该正确显示该对象


### 


 代理文本



### 


班级
 

 Transformers.tools.agent\_types。
 

 代理文本


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agent_types.py#L71)



 (
 


 价值




 )
 


代理返回的文本类型。表现为字符串。



### 


 代理图像



### 


班级
 

 Transformers.tools.agent\_types。
 

 代理图像


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agent_types.py#L83)



 (
 


 价值




 )
 


代理返回的图像类型。行为类似于 PIL.Image。



#### 


到\_raw


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agent_types.py#L115)


（
 

 ）


返回该对象的“原始”版本。对于 AgentImage，它是 PIL.Image。




#### 


到\_string


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tools/agent_types.py#L126)


（
 

 ）


返回该对象的字符串化版本。对于 AgentImage，它是序列化的路径
图像的版本。




### 


 代理音频



###