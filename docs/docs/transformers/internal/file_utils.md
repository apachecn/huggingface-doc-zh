# 通用公用事业

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/internal/file_utils>
>
> 原始地址：<https://huggingface.co/docs/transformers/internal/file_utils>


此页面列出了文件中找到的所有 Transformer 通用实用函数
 `utils.py`
 。


其中大多数仅在您学习库中的通用代码时才有用。


## 枚举和命名元组




### 


班级
 

 变压器.utils。
 

 显式枚举


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/generic.py#L448)


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


带有更明确的缺失值错误消息的枚举。




### 


班级
 

 变压器.utils。
 

 填充策略


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/generic.py#L460)


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


可能的值
 `填充`
 论证中
 [PreTrainedTokenizerBase.
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 。对于制表符补全很有用
IDE。




### 


班级
 

 变压器。
 

 张量类型


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/generic.py#L471)


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


可能的值
 `返回张量`
 论证中
 [PreTrainedTokenizerBase.
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 。对...有用
IDE 中的制表符补全。


## 特别装饰者




#### 


Transformers.add\_start\_docstrings


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/doc.py#L23)


（


 \* 文档字符串


）


#### 


Transformers.utils.add\_start\_docstrings\_to\_model\_forward


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/doc.py#L31)


（


 \* 文档字符串


）


#### 


Transformers.add\_end\_docstrings


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/doc.py#L53)


（


 \* 文档字符串


）


#### 


Transformers.utils.add\_code\_sample\_docstrings


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/doc.py#L1064)


（


 \* 文档字符串


 处理器\_class
 
 = 无


检查站
 
 = 无


输出\_类型
 
 = 无


配置\_class
 
 = 无


面具
 
 = '[掩码]'


qa\_target\_start\_index
 
 = 14


qa\_target\_end\_index
 
 = 15


型号\_cls
 
 = 无


情态
 
 = 无


预期\_输出
 
 = 无


预期损失
 
 = 无


真实\_检查点
 
 = 无


）


#### 


Transformers.utils.replace\_return\_docstrings


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/doc.py#L1152)


（


 输出\_类型
 
 = 无


配置\_class
 
 = 无


）


## 特殊属性




### 


班级
 

 变压器.utils。
 

 缓存\_属性


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/generic.py#L36)


（


 得到
 
 = 无


偏移量
 
 = 无


弗德尔
 
 = 无


文档
 
 = 无


）


模仿@property但将输出缓存在成员变量中的描述符。


来自张量流_数据集


Python 3.8 中内置 functools。


## 其他实用程序




### 


班级
 

 变压器.utils。
 

 \_LazyModule


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/utils/import_utils.py#L1304)


（


 姓名


 模块\_文件


 导入\_结构


 模块\_spec
 
 = 无


额外\_对象
 
 = 无


）


显示所有对象但仅在请求对象时执行关联导入的模块类。