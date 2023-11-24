# 管道公用设施

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/internal/pipelines_utils>
>
> 原始地址：<https://huggingface.co/docs/transformers/internal/pipelines_utils>


此页面列出了该库为管道提供的所有实用函数。


其中大多数仅在您研究库中模型的代码时才有用。


## 参数处理




### 


班级
 

 变压器.管道。
 

 参数处理程序


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L425)


（
 

 ）


用于处理每个参数的基本接口
 [管道](/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.Pipeline)
 。




### 


班级
 

 变压器.管道。
 

 ZeroShotClassificationArgumentHandler


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/zero_shot_classification.py#L14)


（
 

 ）


通过将每个可能的标签转换为 NLI 来处理文本分类的零样本参数
前提/假设对。




### 


班级
 

 变压器.管道。
 

 问题回答参数处理程序


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/question_answering.py#L150)


（
 

 ）


QuestionAnsweringPipeline 要求用户提供要映射到的多个参数（即问题和上下文）
内部的
 `小队示例`
 。


QuestionAnsweringArgumentHandler 管理所有可能的创建
 `小队示例`
 从命令行
提供的参数。


## 数据格式




### 


班级
 

 变压器。
 

 管道数据格式


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L435)


（


 输出\_路径
 
 : 打字.可选[str]


输入\_路径
 
 : 打字.可选[str]


柱子
 
 : 打字.可选[str]


覆盖
 
 ：布尔=假


）


 参数


* **输出\_路径**
 （
 `str`
 ) — 保存传出数据的位置。
* **输入\_路径**
 （
 `str`
 ) — 在哪里查找输入数据。
* **柱子**
 （
 `str`
 ) — 要读取的列。
* **覆盖**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否覆盖
 `输出路径`
 。


 所有管道支持的读取和写入数据格式的基类。支持的数据格式
目前包括：


* JSON
* CSV
* 标准输入/标准输出（管道）


`管道数据格式`
 还包括一些用于处理多列的实用程序，例如从数据集列映射到
通过管道传递关键字参数
 `dataset_kwarg_1=dataset_column_1`
 格式。



#### 


来自\_str


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L512)


（


 格式
 
 : 字符串


输出\_路径
 
 : 打字.可选[str]


输入\_路径
 
 : 打字.可选[str]


柱子
 
 : 打字.可选[str]


覆盖
 
 = 假


）
 

 →


导出常量元数据='未定义';
 

[PipelineDataFormat](/docs/transformers/v4.35.2/en/internal/pipelines_utils#transformers.PipelineDataFormat)


 参数


* **格式**
 （
 `str`
 )—
所需管道的格式。可接受的值为
 `“json”`
 ,
 `“csv”`
 或者
 `“管道”`
 。
* **输出\_路径**
 （
 `str`
 ,
 *选修的*
 )—
保存传出数据的位置。
* **输入\_路径**
 （
 `str`
 ,
 *选修的*
 )—
在哪里查找输入数据。
* **柱子**
 （
 `str`
 ,
 *选修的*
 )—
要阅读的专栏。
* **覆盖**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否覆盖
 `输出路径`
 。


退货


导出常量元数据='未定义';
 

[PipelineDataFormat](/docs/transformers/v4.35.2/en/internal/pipelines_utils#transformers.PipelineDataFormat)


导出常量元数据='未定义';


正确的数据格式。


创建右子类的实例
 [PipelineDataFormat](/docs/transformers/v4.35.2/en/internal/pipelines_utils#transformers.PipelineDataFormat)
 根据
 `格式`
 。




#### 


节省


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L484)


（


 数据
 
 : 打字.Union[dict, 打字.List[dict]]


）


 参数


* **数据**
 （
 `字典`
 或列表
 `字典`
 ) — 要存储的数据。


 将提供的数据对象与当前的表示一起保存
 [PipelineDataFormat](/docs/transformers/v4.35.2/en/internal/pipelines_utils#transformers.PipelineDataFormat)
 。




#### 


保存\_binary


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L494)


（


 数据
 
 : 打字.Union[dict, 打字.List[dict]]


）
 

 →


导出常量元数据='未定义';
 

`str`


 参数


* **数据**
 （
 `字典`
 或列表
 `字典`
 ) — 要存储的数据。


退货


导出常量元数据='未定义';
 

`str`


导出常量元数据='未定义';


数据保存的路径。


将提供的数据对象作为 pickle 格式的二进制数据保存在磁盘上。


### 


班级
 

 变压器。
 

 CsvPipeline数据格式


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L548)


（


 输出\_路径
 
 : 打字.可选[str]


输入\_路径
 
 : 打字.可选[str]


柱子
 
 : 打字.可选[str]


覆盖
 
 = 假


）


 参数


* **输出\_路径**
 （
 `str`
 ) — 保存传出数据的位置。
* **输入\_路径**
 （
 `str`
 ) — 在哪里查找输入数据。
* **柱子**
 （
 `str`
 ) — 要读取的列。
* **覆盖**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否覆盖
 `输出路径`
 。


 支持使用 CSV 数据格式的管道。



#### 


节省


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L578)


（


 数据
 
 : 打字.List[字典]


）


 参数


* **数据**
 （
 `列表[字典]`
 ) — 要存储的数据。


 将提供的数据对象与当前的表示一起保存
 [PipelineDataFormat](/docs/transformers/v4.35.2/en/internal/pipelines_utils#transformers.PipelineDataFormat)
 。


### 


班级
 

 变压器。
 

 JsonPipeline数据格式


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L592)


（


 输出\_路径
 
 : 打字.可选[str]


输入\_路径
 
 : 打字.可选[str]


柱子
 
 : 打字.可选[str]


覆盖
 
 = 假


）


 参数


* **输出\_路径**
 （
 `str`
 ) — 保存传出数据的位置。
* **输入\_路径**
 （
 `str`
 ) — 在哪里查找输入数据。
* **柱子**
 （
 `str`
 ) — 要读取的列。
* **覆盖**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否覆盖
 `输出路径`
 。


 支持使用 JSON 文件格式的管道。



#### 


节省


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L623)


（


 数据
 
 : 字典


）


 参数


* **数据**
 （
 `字典`
 ) — 要存储的数据。


 将提供的数据对象保存在 json 文件中。


### 


班级
 

 变压器。
 

 PipedPipeline数据格式


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L634)


（


 输出\_路径
 
 : 打字.可选[str]


输入\_路径
 
 : 打字.可选[str]


柱子
 
 : 打字.可选[str]


覆盖
 
 ：布尔=假


）


 参数


* **输出\_路径**
 （
 `str`
 ) — 保存传出数据的位置。
* **输入\_路径**
 （
 `str`
 ) — 在哪里查找输入数据。
* **柱子**
 （
 `str`
 ) — 要读取的列。
* **覆盖**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否覆盖
 `输出路径`
 。


 从管道输入读取数据到 python 进程。对于多列数据，列应该用分隔符分隔


如果提供了列，则输出将是带有 {column\_x: value\_x} 的字典



#### 


节省


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L663)


（


 数据
 
 : 字典


）


 参数


* **数据**
 （
 `字典`
 ) — 要存储的数据。


 打印数据。


## 公用事业




### 


班级
 

 变压器.管道。
 

 管道异常


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pipelines/base.py#L408)


（


 任务
 
 : 字符串


模型
 
 : 字符串


原因
 
 : 字符串


）


 参数


* **任务**
 （
 `str`
 ) — 管道的任务。
* **模型**
 （
 `str`
 ) — 管道使用的模型。
* **原因**
 （
 `str`
 ) — 要显示的错误消息。


 由一个
 [管道](/docs/transformers/v4.35.2/en/main_classes/pipelines#transformers.Pipeline)
 处理时
 **称呼**
 。