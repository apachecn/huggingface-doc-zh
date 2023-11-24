# 自定义层和实用程序

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/internal/modeling_utils>
>
> 原始地址：<https://huggingface.co/docs/transformers/internal/modeling_utils>


此页面列出了该库使用的所有自定义层，以及它为建模提供的实用函数。


其中大多数仅在您研究库中模型的代码时才有用。


## Pytorch 自定义模块




### 


班级
 

 变压器。
 

 转换1D


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pytorch_utils.py#L87)


（


 NF


 恩克斯


）


 参数


* **nf**
 （
 `int`
 ) — 输出特征的数量。
* **nx**
 （
 `int`
 ) — 输入特征的数量。


 Radford 等人定义的一维卷积层。适用于 OpenAI GPT（也用于 GPT-2）。


基本上像线性层一样工作，但权重被转置了。




### 


班级
 

 Transformers.modeling\_utils。
 

 池启动日志


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4146)


（


 配置
 
 ：预训练配置


）


 参数


* **配置**
 （
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 )—
模型使用的配置将用于获取
 `隐藏大小`
 模型的。


 从序列隐藏状态计算 SQuAD 起始逻辑。



#### 


向前


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4159)


（


 隐藏\_状态
 
 : 浮点张量


p\_掩码
 
 : 打字.Optional[torch.FloatTensor] = None


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`


 参数


* **隐藏\_状态**
 （
 `torch.FloatTensor`
 形状的
 `（batch_size，seq_len，hidden_​​size）`
 )—
模型的最终隐藏状态。
* **p\_mask**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, seq_len)`
 ,
 *选修的*
 )—
无效位置标记的掩码，例如查询和特殊符号（PAD、SEP、CLS）。 1.0 表示代币
应该被屏蔽。


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`


导出常量元数据='未定义';


SQuAD 的开始日志。


### 


班级
 

 Transformers.modeling\_utils。
 

 池化结束日志


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4184)


（


 配置
 
 ：预训练配置


）


 参数


* **配置**
 （
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 )—
模型使用的配置将用于获取
 `隐藏大小`
 模型和
 `layer_norm_eps`
 使用。


 从序列隐藏状态计算 SQuAD 最终逻辑。



#### 


向前


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4201)


（


 隐藏\_状态
 
 : 浮点张量


开始\_状态
 
 : 打字.Optional[torch.FloatTensor] = None


开始\_位置
 
 : 打字.Optional[torch.LongTensor] = None


p\_掩码
 
 : 打字.Optional[torch.FloatTensor] = None


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`


 参数


* **隐藏\_状态**
 （
 `torch.FloatTensor`
 形状的
 `（batch_size，seq_len，hidden_​​size）`
 )—
模型的最终隐藏状态。
* **开始\_状态**
 （
 `torch.FloatTensor`
 形状的
 `（batch_size，seq_len，hidden_​​size）`
 ,
 *选修的*
 )—
标记范围的第一个标记的隐藏状态。
* **开始\_位置**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 )—
标记范围的第一个标记的位置。
* **p\_mask**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, seq_len)`
 ,
 *选修的*
 )—
无效位置标记的掩码，例如查询和特殊符号（PAD、SEP、CLS）。 1.0 表示代币
应该被屏蔽。


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`


导出常量元数据='未定义';


SQuAD 的最终日志。


之一
 `开始状态`
 或者
 `开始位置`
 不应该
 `无`
 。如果两者都设置了，
 `开始位置`
 覆盖
 `开始状态`
 。



### 


班级
 

 Transformers.modeling\_utils。
 

 普勒答案类


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4253)


（


 配置


）


 参数


* **配置**
 （
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 )—
模型使用的配置将用于获取
 `隐藏大小`
 模型的。


 根据分类计算 SQuAD 2.0 答案类别并开始标记隐藏状态。



#### 


向前


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4268)


（


 隐藏\_状态
 
 : 浮点张量


开始\_状态
 
 : 打字.Optional[torch.FloatTensor] = None


开始\_位置
 
 : 打字.Optional[torch.LongTensor] = None


cls\_index
 
 : 打字.Optional[torch.LongTensor] = None


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`


 参数


* **隐藏\_状态**
 （
 `torch.FloatTensor`
 形状的
 `（batch_size，seq_len，hidden_​​size）`
 )—
模型的最终隐藏状态。
* **开始\_状态**
 （
 `torch.FloatTensor`
 形状的
 `（batch_size，seq_len，hidden_​​size）`
 ,
 *选修的*
 )—
标记范围的第一个标记的隐藏状态。
* **开始\_位置**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 )—
标记范围的第一个标记的位置。
* **cls\_index**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 )—
批次中每个句子的 CLS 标记的位置。如果
 `无`
 ，取最后一个令牌。


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`


导出常量元数据='未定义';


SQuAD 2.0 答案课程。


之一
 `开始状态`
 或者
 `开始位置`
 不应该
 `无`
 。如果两者都设置了，
 `开始位置`
 覆盖
 `开始状态`
 。



### 


班级
 

 Transformers.modeling\_utils。
 

 班长输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4319)


（


 损失
 
 : 打字.Optional[torch.FloatTensor] = None


开始\_top\_log\_probs
 
 : 打字.Optional[torch.FloatTensor] = None


开始\_顶部\_索引
 
 : 打字.Optional[torch.LongTensor] = None


结束\_top\_log\_probs
 
 : 打字.Optional[torch.FloatTensor] = None


结束\_顶部\_索引
 
 : 打字.Optional[torch.LongTensor] = None


cls\_logits
 
 : 打字.Optional[torch.FloatTensor] = None


）


 参数


* **损失**
 （
 `torch.FloatTensor`
 形状的
 `(1,)`
 ,
 *选修的*
 ，如果两者都返回
 `开始位置`
 和
 `结束位置`
 提供）—
分类损失为开始标记、结束标记（如果提供，则不可能）分类的总和
损失。
* **开始\_top\_log\_probs**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.start_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供）—
记录顶部 config.start\_n\_top 启动标记可能性的概率（束搜索）。
* **开始\_top\_index**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, config.start_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供）—
顶级 config.start\_n\_top 启动标记可能性的索引（束搜索）。
* **结束\_top\_log\_probs**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.start_n_top * config.end_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供）—
顶部的对数概率
 `config.start_n_top * config.end_n_top`
 结束标记的可能性
（束搜索）。
* **结束\_top\_index**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, config.start_n_top * config.end_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供）—
顶部指数
 `config.start_n_top * config.end_n_top`
 结束令牌可能性（束搜索）。
* **cls\_logits**
 （
 `torch.FloatTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供）—
对数概率
 `是不可能的`
 答案的标签。


 使用问答模型输出的基类
 [SQuADHead](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.modeling_utils.SQuADHead)
 。




### 


班级
 

 Transformers.modeling\_utils。
 

 中队队长


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4349)


（


 配置


）


 参数


* **配置**
 （
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 )—
模型使用的配置将用于获取
 `隐藏大小`
 模型和
 `layer_norm_eps`
 使用。


 受 XLNet 启发的 SQuAD 负责人。



#### 


向前


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4368)


（


 隐藏\_状态
 
 : 浮点张量


开始\_位置
 
 : 打字.Optional[torch.LongTensor] = None


结束\_位置
 
 : 打字.Optional[torch.LongTensor] = None


cls\_index
 
 : 打字.Optional[torch.LongTensor] = None


是不可能的
 
 : 打字.Optional[torch.LongTensor] = None


p\_掩码
 
 : 打字.Optional[torch.FloatTensor] = None


返回\_dict
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

[transformers.modeling\_utils.SquadHeadOutput](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.modeling_utils.SquadHeadOutput)
 或者
 `元组（火炬.FloatTensor）`


 参数


* **隐藏\_状态**
 （
 `torch.FloatTensor`
 形状的
 `（batch_size，seq_len，hidden_​​size）`
 )—
序列标记上模型的最终隐藏状态。
* **开始\_位置**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 )—
标记范围的第一个标记的位置。
* **结束\_位置**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 )—
标记范围的最后一个标记的位置。
* **cls\_index**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 )—
批次中每个句子的 CLS 标记的位置。如果
 `无`
 ，取最后一个令牌。
* **是不可能的**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 )—
问题在段落中是否有可能的答案。
* **p\_mask**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, seq_len)`
 ,
 *选修的*
 )—
无效位置标记的掩码，例如查询和特殊符号（PAD、SEP、CLS）。 1.0 表示代币
应该被屏蔽。
* **返回\_dict**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否退货
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 而不是一个普通的元组。


退货


导出常量元数据='未定义';
 

[transformers.modeling\_utils.SquadHeadOutput](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.modeling_utils.SquadHeadOutput)
 或者
 `元组（火炬.FloatTensor）`


导出常量元数据='未定义';


A
 [transformers.modeling\_utils.SquadHeadOutput](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.modeling_utils.SquadHeadOutput)
 或一个元组
 `torch.FloatTensor`
 （如果
 `return_dict=False`
 通过或当
 `config.return_dict=False`
 ) 包括各种
元素取决于配置（
 `<class 'transformers.configuration_utils.PretrainedConfig'>`
 ）和输入。


* **损失**
 （
 `torch.FloatTensor`
 形状的
 `(1,)`
 ,
 *选修的*
 ，如果两者都返回
 `开始位置`
 和
 `结束位置`
 提供） - 分类损失作为开始标记、结束标记（如果提供，则为\_impossible）分类的总和
损失。
* **开始\_top\_log\_probs**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.start_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供） — 记录顶部 config.start\_n\_top 启动标记可能性的概率（束搜索）。
* **开始\_top\_index**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, config.start_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供） — 顶级 config.start\_n\_top 启动标记可能性的索引（束搜索）。
* **结束\_top\_log\_probs**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.start_n_top * config.end_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供） - 顶部的对数概率
 `config.start_n_top * config.end_n_top`
 结束标记的可能性
（束搜索）。
* **结束\_top\_index**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, config.start_n_top * config.end_n_top)`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供） - 顶部索引
 `config.start_n_top * config.end_n_top`
 结束令牌可能性（束搜索）。
* **cls\_logits**
 （
 `torch.FloatTensor`
 形状的
 `（批量大小，）`
 ,
 *选修的*
 ，如果返回
 `开始位置`
 或者
 `结束位置`
 未提供） - 对数概率
 `是不可能的`
 答案的标签。


### 


班级
 

 Transformers.modeling\_utils。
 

 序列摘要


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4466)


（


 配置
 
 ：预训练配置


）


 参数


* **配置**
 （
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 )—
模型使用的配置。模型的config类中的相关参数为（参考实际配置）
模型的配置类以获取其使用的默认值）：
 
+ **摘要\_类型**
（
`str`
) — 用于进行此总结的方法。接受的值为：



- `“最后”`
— 取最后一个 token 隐藏状态（如 XLNet）
- `“第一”`
— 取第一个 token 隐藏状态（如 Bert）
- `“意思”`
— 取所有 token 隐藏状态的平均值
- `“cls_index”`
— 提供分类标记位置的张量（GPT/GPT-2）
- `“收件人”`
- 现在没有实现，使用多头注意力
+ **摘要\_use\_proj**
（
`布尔`
) — 在矢量提取后添加投影。
+ **摘要\_proj\_to\_labels**
（
`布尔`
） - 如果
'真实'
，投影输出到
`config.num_labels`
类
（否则到
`config.hidden_​​size`
）。
+ **摘要\_激活**
（
`可选[str]`
） - 设置
`“正弦”`
向输出添加 tanh 激活，
另一个字符串或
`无`
将添加不激活。
+ **摘要\_first\_dropout**
（
`浮动`
) — 投影和激活之前的可选退出概率。
+ **摘要\_last\_dropout**
（
`浮动`
)— 投影和激活后的可选退出概率。


 计算序列隐藏状态的单个向量摘要。



#### 


向前


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_utils.py#L4521)


（


 隐藏\_状态
 
 : 浮点张量


cls\_index
 
 : 打字.Optional[torch.LongTensor] = None


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`


 参数


* **隐藏\_状态**
 （
 `torch.FloatTensor`
 形状的
 `[batch_size、seq_len、hidden_​​size]`
 )—
最后一层的隐藏状态。
* **cls\_index**
 （
 `torch.LongTensor`
 形状的
 `[批量大小]`
 或者
 `[批量大小，...]`
 其中…是可选的主尺寸
 `隐藏状态`
 ,
 *选修的*
 )—
使用如果
 `summary_type == "cls_index"`
 并将序列的最后一个标记作为分类标记。


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`


导出常量元数据='未定义';


序列隐藏状态的摘要。


计算序列隐藏状态的单个向量摘要。


## PyTorch 辅助函数




#### 


Transformers.apply\_chunking\_to\_forward


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pytorch_utils.py#L169)


（


 向前\_fn
 
 : 打字.Callable[..., torch.Tensor]


块\_大小
 
 ：整数


块\_dim
 
 ：整数


\*输入\_张量


）
 

 →


导出常量元数据='未定义';
 

`火炬.张量`


 参数


* **向前\_fn**
 （
 `可调用[..., torch.Tensor]`
 )—
模型的前向函数。
* **块\_size**
 （
 `int`
 )—
分块张量的块大小：
 `num_chunks = len(input_tensors[0]) /chunk_size`
 。
* **块\_dim**
 （
 `int`
 )—
维度
 `输入张量`
 应该分块。
* **输入\_张量**
 （
 `元组[火炬.张量]`
 )—
输入张量为
 `forward_fn`
 它将被分块


退货


导出常量元数据='未定义';
 

`火炬.张量`


导出常量元数据='未定义';


与形状相同的张量
 `forward_fn`
 如果应用的话就会给出`。


该函数将分块
 `输入张量`
 分成更小的输入张量部分
 `块大小`
 超过维度
 `chunk_dim`
 。然后应用一层
 `forward_fn`
 独立地分配给每个块以节省内存。


如果
 `forward_fn`
 是独立的
 `chunk_dim`
 该函数将产生与直接函数相同的结果
申请
 `forward_fn`
 到
 `输入张量`
 。


 例子：



```
# rename the usual forward() fn to forward\_chunk()
def forward\_chunk(self, hidden\_states):
    hidden_states = self.decoder(hidden_states)
    return hidden_states


# implement a chunked forward function
def forward(self, hidden\_states):
    return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
```


#### 


Transformers.pytorch\_utils.find\_pruneable\_heads\_and\_indices


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pytorch_utils.py#L244)


（


 头
 
 : 打字.List[int]


n\_头
 
 ：整数


头\_尺寸
 
 ：整数


已经\_修剪\_头
 
 : 打字.Set[int]


）
 

 →


导出常量元数据='未定义';
 

`元组[Set[int], torch.LongTensor]`


 参数


* **头**
 （
 `列表[整数]`
 ) — 要修剪的头索引列表。
* **n\_heads**
 （
 `int`
 ) — 模型中的头部数量。
* **头\_尺寸**
 （
 `int`
 ) — 每个头的大小。
* **已经\_修剪\_heads**
 （
 `设置[整数]`
 ) — 一组已经修剪过的头。


退货


导出常量元数据='未定义';
 

`元组[Set[int], torch.LongTensor]`


导出常量元数据='未定义';


带有要修剪的头索引的元组
 `已经修剪的头`
 考虑到要保留在层权重中的行/列的索引。


查找头部及其索引
 `已经修剪的头`
 考虑到。




#### 


Transformers.prune\_layer


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pytorch_utils.py#L145)


（


 层
 
 : 打字.Union[torch.nn.modules.linear.Linear, Transformers.pytorch\_utils.Conv1D]


指数
 
 : 长张量


暗淡
 
 : 打字.Optional[int] = None


）
 

 →


导出常量元数据='未定义';
 

`torch.nn.Linear`
 或者
 [Conv1D](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.Conv1D)


 参数


* **层**
 （
 `Union[torch.nn.Linear, Conv1D]`
 ) — 要修剪的图层。
* **指数**
 （
 `torch.LongTensor`
 ) — 要保留在层中的索引。
* **暗淡**
 （
 `int`
 ,
 *选修的*
 ) — 保留索引的维度。


退货


导出常量元数据='未定义';
 

`torch.nn.Linear`
 或者
 [Conv1D](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.Conv1D)


导出常量元数据='未定义';


修剪后的层作为新层
 `requires_grad=True`
 。


修剪 Conv1D 或线性层以仅保留索引中的条目。


用于去除头部。




#### 


Transformers.pytorch\_utils.prune\_conv1d\_layer


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pytorch_utils.py#L112)


（


 层
 
 ：Conv1D


指数
 
 : 长张量


暗淡
 
 ：整数= 1


）
 

 →


导出常量元数据='未定义';
 

[Conv1D](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.Conv1D)


 参数


* **层**
 （
 [Conv1D](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.Conv1D)
 ) — 要修剪的图层。
* **指数**
 （
 `torch.LongTensor`
 ) — 要保留在层中的索引。
* **暗淡**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) — 保留索引的维度。


退货


导出常量元数据='未定义';
 

[Conv1D](/docs/transformers/v4.35.2/en/internal/modeling_utils#transformers.Conv1D)


导出常量元数据='未定义';


修剪后的层作为新层
 `requires_grad=True`
 。


修剪 Conv1D 层以仅保留索引中的条目。 Conv1D 作为线性层工作（参见 BERT 等），但权重
被转置。


用于去除头部。




#### 


Transformers.pytorch\_utils.prune\_线性\_layer


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/pytorch_utils.py#L53)


（


 层
 
 ：线性


指数
 
 : 长张量


暗淡
 
 ：整数=0


）
 

 →


导出常量元数据='未定义';
 

`torch.nn.Linear`


 参数


* **层**
 （
 `torch.nn.Linear`
 ) — 要修剪的图层。
* **指数**
 （
 `torch.LongTensor`
 ) — 要保留在层中的索引。
* **暗淡**
 （
 `int`
 ,
 *选修的*
 ，默认为 0) — 保留索引的维度。


退货


导出常量元数据='未定义';
 

`torch.nn.Linear`


导出常量元数据='未定义';


修剪后的层作为新层
 `requires_grad=True`
 。


修剪线性层以仅保留索引中的条目。


用于去除头部。


## TensorFlow 自定义层




### 


班级
 

 Transformers.modeling\_tf\_utils。
 

 TFConv1D


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L3187)


（


 \*参数


 \*\*夸格


）


 参数


* **nf**
 （
 `int`
 )—
输出特征的数量。
* **nx**
 （
 `int`
 )—
输入特征的数量。
* **初始化器\_range**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.02) —
用于初始化权重的标准差。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
传递给的附加关键字参数
 `__init__`
 的
 `tf.keras.layers.Layer`
 。


 Radford 等人定义的一维卷积层。适用于 OpenAI GPT（也用于 GPT-2）。


基本上像线性层一样工作，但权重被转置了。




### 


班级
 

 变压器。
 

 TF序列摘要


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L3330)


（


 \*参数


 \*\*夸格


）


 参数


* **配置**
 （
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 )—
模型使用的配置。模型的config类中的相关参数为（参考实际配置）
模型的配置类以获取其使用的默认值）：
 
+ **摘要\_类型**
（
`str`
) — 用于进行此总结的方法。接受的值为：



- `“最后”`
— 取最后一个 token 隐藏状态（如 XLNet）
- `“第一”`
— 取第一个 token 隐藏状态（如 Bert）
- `“意思”`
— 取所有 token 隐藏状态的平均值
- `“cls_index”`
— 提供分类标记位置的张量（GPT/GPT-2）
- `“收件人”`
- 现在没有实现，使用多头注意力
+ **摘要\_use\_proj**
（
`布尔`
) — 在矢量提取后添加投影。
+ **摘要\_proj\_to\_labels**
（
`布尔`
） - 如果
'真实'
，投影输出到
`config.num_labels`
类
（否则到
`config.hidden_​​size`
）。
+ **摘要\_激活**
（
`可选[str]`
） - 设置
`“正弦”`
向输出添加 tanh 激活，
另一个字符串或
`无`
将添加不激活。
+ **摘要\_first\_dropout**
（
`浮动`
) — 投影和激活之前的可选退出概率。
+ **摘要\_last\_dropout**
（
`浮动`
)— 投影和激活后的可选退出概率。
* **初始化器\_range**
 （
 `浮动`
 ，默认为 0.02) — 用于初始化权重的标准差。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
传递给的附加关键字参数
 `__init__`
 的
 `tf.keras.layers.Layer`
 。


 计算序列隐藏状态的单个向量摘要。


## TensorFlow 损失函数




### 


班级
 

 Transformers.modeling\_tf\_utils。
 

 TF因果语言建模损失


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L192)


（
 

 ）


损失函数适合因果语言建模（CLM），即猜测下一个标记的任务。


在损失计算中，任何 -100 的标签都将被忽略（以及相应的 logits）。


### 


班级
 

 Transformers.modeling\_tf\_utils。
 

 TFMaskedLanguageModelingLoss


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L311)


（
 

 ）


损失函数适用于屏蔽语言建模（MLM），即猜测屏蔽标记的任务。


在损失计算中，任何 -100 的标签都将被忽略（以及相应的 logits）。


### 


班级
 

 Transformers.modeling\_tf\_utils。
 

 TF多项选择损失


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L301)


（
 

 ）


适合多项选择任务的损失函数。




### 


班级
 

 Transformers.modeling\_tf\_utils。
 

 TF问题回答损失


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L223)


（
 

 ）


适合问答的损失函数。




### 


班级
 

 Transformers.modeling\_tf\_utils。
 

 TF序列分类损失


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L282)


（
 

 ）


适合序列分类的损失函数。




### 


班级
 

 Transformers.modeling\_tf\_utils。
 

 TFToken分类损失


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L238)


（
 

 ）


适合token分类的损失函数。


在损失计算中，任何 -100 的标签都将被忽略（以及相应的 logits）。


## TensorFlow 辅助函数




#### 


Transformers.modeling\_tf\_utils.get\_initializer


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L3446)


（


 初始化器\_range
 
 ：浮动= 0.02


）
 

 →


导出常量元数据='未定义';
 

`tf.keras.initializers.TruncatedNormal`


 参数


* **初始化器\_range**
 （
 *漂浮*
 ，默认为 0.02) — 初始值设定项范围的标准偏差。


退货


导出常量元数据='未定义';
 

`tf.keras.initializers.TruncatedNormal`


导出常量元数据='未定义';


被截断的普通初始值设定项。


创建一个
 `tf.keras.initializers.TruncatedNormal`
 与给定的范围。




#### 


Transformers.modeling\_tf\_utils.keras\_serialized


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/modeling_tf_utils.py#L127)


（
 

 ）


 参数


* **CLS**
 （A
 `tf.keras.layers.Layers 子类`
 )—
通常是一个
 `TF.MainLayer`
 在这个项目中的类，一般必须接受一个
 `配置`
 对其论证
初始化程序。


 装饰 Keras Layer 类以支持 Keras 序列化。


这是通过以下方式完成的：


1.添加一个
 `transformers_config`
 dict 到 Keras 配置字典中
 `获取配置`
 （由 Keras 调用
序列化时间。
2. 包裹
 `__init__`
 接受那个
 `transformers_config`
 dict（在反序列化时由 Keras 传递）和
将其转换为实际层初始值设定项的配置对象。
3、在Keras中将类注册为自定义对象（如果Tensorflow版本支持的话），这样就不会
需要提供在
 `自定义对象`
 在通话中
 `tf.keras.models.load_model`
 。




#### 


变压器.shape\_list


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tf_utils.py#L26)


（


 张量
 
 : 打字.Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray]


）
 

 →


导出常量元数据='未定义';
 

`列表[整数]`


 参数


* **张量**
 （
 `tf.张量`
 或者
 `np.ndarray`
 ) — 我们想要形状的张量。


退货


导出常量元数据='未定义';
 

`列表[整数]`


导出常量元数据='未定义';


张量的形状为列表。


干净地处理张量流中的动态形状。