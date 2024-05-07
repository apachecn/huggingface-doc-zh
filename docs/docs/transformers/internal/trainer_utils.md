# Trainer实用程序

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/internal/trainer_utils>
>
> 原始地址：<https://huggingface.co/docs/transformers/internal/trainer_utils>


此页面列出了使用的所有实用函数
 [Trainer](/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.Trainer)
 。


其中大多数仅当您在库中学习 Trainer 的代码时才有用。


## 公用事业




### 


班级
 

 变压器。
 

 评估预测


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_utils.py#L134)


（


 预测
 
 : 打字.Union[numpy.ndarray, 打字.Tuple[numpy.ndarray]]


标签\_ids
 
 : 打字.Union[numpy.ndarray, 打字.Tuple[numpy.ndarray]]


输入
 
 : 打字.Union[numpy.ndarray, 打字.Tuple[numpy.ndarray], NoneType] = 无


）


 参数


* **预测**
 （
 `np.ndarray`
 ) — 模型的预测。
* **标签\_ids**
 （
 `np.ndarray`
 ) — 要匹配的目标。
* **输入**
 （
 `np.ndarray`
 ,
 *选修的*
 )—


 评估输出（始终包含标签），用于计算指标。




### 


班级
 

 变压器。
 

 区间策略


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_utils.py#L208)


（


 价值


 名字
 
 = 无


模块
 
 = 无


限定名
 
 = 无


类型
 
 = 无


开始
 
 = 1


）


一个枚举。




#### 


Transformers.enable\_full\_决定论


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_utils.py#L58)


（


 种子
 
 ：整数


仅警告\_
 
 ：布尔=假


）


分布式训练期间可重现行为的辅助函数。看


* <https://pytorch.org/docs/stable/notes/randomness.html>
 对于火炬
* <https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism>
 对于张量流




#### 


变形金刚.set\_seed


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_utils.py#L85)


（


 种子
 
 ：整数


）


 参数


* **种子**
 （
 `int`
 ) — 要设置的种子。


 用于设置种子的可重现行为的辅助函数
 `随机`
 ,
 `numpy`
 ,
 `火炬`
 和/或
 `tf`
 （如果已安装）。




#### 


Transformers.torch\_distributed\_zero\_first


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_pt_utils.py#L242)


（


 本地\_等级
 
 ：整数


）


 参数


* **本地\_排名**
 （
 `int`
 ) — 本地进程的等级。


 装饰器使分布式训练中的所有进程等待每个本地\_master 执行某些操作。


## 回调内部结构




### 


班级
 

 Transformers.trainer\_callback。
 

 回调处理程序


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_callback.py#L300)


（


 回调


 模型


 分词器


 优化器


 lr\_调度程序


）


仅按顺序调用回调列表的内部类。


## 分布式评估




### 


班级
 

 Transformers.trainer\_pt\_utils。
 

 分布式张量收集器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_pt_utils.py#L371)


（


 世界规模


 样本数


 制作\_多个\_of
 
 = 无


填充\_index
 
 =-100


）


 参数


* **世界规模**
 （
 `int`
 )—
分布式训练中使用的进程数。
* **num\_samples**
 （
 `int`
 )—
我们数据集中的样本数量。
* **制作\_多个\_of**
 （
 `int`
 ,
 *选修的*
 )—
如果通过，该类假定传递给每个进程的数据集是该参数的倍数
（通过添加样本）。
* **填充\_index**
 （
 `int`
 ,
 *选修的*
 ，默认为-100) —
如果数组不具有相同的序列长度，则使用填充索引。


 负责在 CPU 上按块正确收集张量（或张量的嵌套列表/元组）的类。


如果我们的数据集有 16 个样本，批量大小为 2 对 3 个进程，并且我们每次收集数据然后在 CPU 上传输
步骤，我们的采样器将生成以下索引：


`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`


获得大小为 3 倍数的数据（以便每个进程获得相同的数据集长度）。然后处理0、1和
2将负责对以下样本进行预测：


* P0:
 `[0, 1, 2, 3, 4, 5]`
* P1：
 `[6,7,8,9,10,11]`
* P2：
 `[12, 13, 14, 15, 0, 1]`


每个工序处理的第一批产品将是


* P0:
 `[0, 1]`
* P1：
 `[6, 7]`
* P2：
 `[12, 13]`


因此，如果我们在第一批结束时收集，我们将得到一个对应于的张量（张量的嵌套列表/元组）
以下指数：


`[0, 1, 6, 7, 12, 13]`


如果我们直接连接我们的结果而不采取任何预防措施，用户将得到以下预测：
预测循环结束时按此顺序排列的索引：


`[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`


出于某种原因，这不会让他们的船翻滚。这个类就是为了解决这个问题而存在的。



#### 


添加\_数组


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_pt_utils.py#L431)


（


 数组


）


添加
 `数组`
 到内部存储，将在第一个数组传递时将存储初始化为完整大小
因此，如果我们一定会遇到 OOM，那么它会在一开始就发生。




#### 


敲定


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer_pt_utils.py#L467)


（
 

 ）


返回正确收集的数组并截断为样本数（因为采样器添加了一些额外的内容）
使每个进程获得相同长度的数据集）。


## 分布式评估




### 


班级
 

 变压器。
 

 Hf参数解析器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/hf_argparser.py#L109)


（


 数据类\_类型
 
 : 打字.Union[DataClassType, 打字.Iterable[DataClassType]]


\*\*夸格


）


这个子类
 `argparse.ArgumentParser`
 使用数据类上的类型提示来生成参数。


该类旨在与本机 argparse 很好地配合。特别是，您可以添加更多（非数据类支持）
初始化后将参数传递给解析器，解析后您将获得输出作为附加
命名空间。可选：要创建子参数组，请使用
 `_参数组_名称`
 数据类中的属性。



#### 


将\_args\_解析为\_dataclasses


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/hf_argparser.py#L265)


（


 参数
 
 = 无


返回\_剩余\_字符串
 
 = 假


查找\_for\_args\_file
 
 = 正确


args\_文件名
 
 = 无


args\_file\_flag
 
 = 无


）
 

 →


导出常量元数据='未定义';
 

 元组组成


 退货


导出常量元数据='未定义';
 

 元组组成


导出常量元数据='未定义';


* 数据类实例的顺序与传递给initializer.abspath的顺序相同
* 如果适用，将更多（非数据类支持的）参数的附加命名空间添加到解析器
初始化后。
* 剩余参数字符串的潜在列表。 （与 argparse.ArgumentParser.parse\_known\_args 相同）


将命令行参数解析为指定数据类类型的实例。


这依赖于argparse的
 `ArgumentParser.parse_known_args`
 。请参阅以下位置的文档：
docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse\_args




#### 


解析\_dict


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/hf_argparser.py#L351)


（


 参数
 
 : 打字.Dict[str, 打字.Any]


允许\_额外\_键
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

 元组组成


参数


* **参数**
 （
 `字典`
 )—
包含配置值的字典
* **允许\_额外\_键**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
默认为 False。如果为 False，如果字典包含未解析的键，将引发异常。


退货


导出常量元数据='未定义';
 

 元组组成


导出常量元数据='未定义';


* 数据类实例的顺序与传递给初始化器的顺序相同。


不使用的替代辅助方法
 `argparse`
 根本不用，而是使用字典并填充数据类
类型。




#### 


解析\_json\_file


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/hf_argparser.py#L379)


（


 json\_file
 
 : 字符串


允许\_额外\_键
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

 元组组成


参数


* **json\_file**
 （
 `str`
 或者
 `os.PathLike`
 )—
要解析的 json 文件的文件名
* **允许\_额外\_键**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
默认为 False。如果为 False，如果 json 文件包含不包含的键，则会引发异常
解析。


退货


导出常量元数据='未定义';
 

 元组组成


导出常量元数据='未定义';


* 数据类实例的顺序与传递给初始化器的顺序相同。


不使用的替代辅助方法
 `argparse`
 根本不用加载 json 文件并填充
数据类类型。




#### 


解析\_yaml\_file


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/hf_argparser.py#L401)


（


 yaml\_file
 
 : 字符串


允许\_额外\_键
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

 元组组成


参数


* **yaml\_file**
 （
 `str`
 或者
 `os.PathLike`
 )—
要解析的 yaml 文件的文件名
* **允许\_额外\_键**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
默认为 False。如果为 False，如果 json 文件包含不包含的键，则会引发异常
解析。


退货


导出常量元数据='未定义';
 

 元组组成


导出常量元数据='未定义';


* 数据类实例的顺序与传递给初始化器的顺序相同。


不使用的替代辅助方法
 `argparse`
 根本不用加载 yaml 文件并填充
数据类类型。


## 调试实用程序




### 


班级
 

 Transformers.debug\_utils。
 

 调试下溢溢出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/debug_utils.py#L27)


（


 模型


 最多保存帧数
 
 = 21


跟踪\_batch\_nums
 
 = []


中止\_after\_batch\_num
 
 = 无


）


 参数


* **模型**
 （
 `nn.模块`
 )—
要调试的模型。
* **最大\_帧数\_保存**
 （
 `int`
 ,
 *选修的*
 ，默认为 21) —
回记录多少帧
* **跟踪\_batch\_nums(
 `列表[整数]`
 ,**
*选修的*
 ，默认为
 `[]`
 )—
要追踪哪些批次号（关闭检测）
* **中止\_after\_batch\_num**
 （“整数”，
 *选修的*
 )—
某个批次完成后是否中止


 此调试类有助于检测和了解模型在哪里开始变得非常大或非常小，等等
重要的
 ‘南’
 或者
 `inf`
 重量和激活元素。


有2种工作模式：


1. 下溢/上溢检测（默认）
2. 无需检测的特定批次绝对最小/最大跟踪


模式1：下溢/上溢检测


 要激活下溢/溢出检测，请使用模型初始化对象：



```
debug_overflow = DebugUnderflowOverflow(model)
```


然后正常运行训练，如果
 ‘南’
 或者
 `inf`
 至少在权重、输入或输出之一中被检测到
该模块将抛出异常并打印的元素
 `要保存的最大帧数`
 导致此事件的帧，
每帧报告


1. 完全限定的模块名称加上其类名称
 ‘前进’
 被运行
2. 每个模块权重的所有元素的绝对最小值和最大值，以及输入和输出


例如，这是检测报告中的标题和最后几帧
 `谷歌/mt5-小`
 以 fp16 运行


 混合精度：



```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```


你可以在这里看到，
 `T5DenseGatedGeluDense.forward`
 导致输出激活，其绝对最大值为
62.7K左右，非常接近fp16的64K上限。在下一帧中我们有
 `辍学`
 哪个
在将一些元素归零后，重新规范化权重，这将绝对最大值推至超过
64K，我们得到了一个overlow。


正如您所看到的，当数字开始变得非常大时，我们需要研究之前的帧
fp16 数字。


跟踪是在前向钩子中完成的，该钩子在之后立即被调用
 ‘前进’
 已完成。


 默认情况下打印最后 21 帧。您可以更改默认值以根据您的需要进行调整。例如 ：



```
debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```


为了验证您是否已正确设置此调试功能，并且您打算在以下培训中使用它：
可能需要几个小时才能完成，首先在为几个批次之一启用正常跟踪的情况下运行它，如中所述
下一节。


模式2.特定批次绝对最小/最大跟踪，无需检测


第二种工作模式是每批次跟踪，下溢/上溢检测功能关闭。


假设您想查看每种成分的所有成分的绝对最小值和最大值
 ‘前进’
 一个的呼唤


 给定批次，并且仅对批次 1 和批次 3 执行此操作。然后将此类实例化为：



```
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```


现在，将使用与上述相同的格式来跟踪完整的批次 1 和批次 3。批次的索引为 0。


如果您知道程序在特定批次号后开始出现异常，这会很有帮助，因此您可以
向右快进到该区域。


提前停止：


 您还可以指定停止训练的批次号，方法是：



```
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```


此功能主要在跟踪模式下有用，但您可以将其用于任何模式。


**表现**
 :


由于该模块测量绝对值
 `分钟`
 /`
 `最大`
 每个前锋上模型的每个权重都会减慢训练速度
向下。因此，一旦满足调试需求，请记住将其关闭。