# 分词器

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/main_classes/tokenizer>
>
> 原始地址：<https://huggingface.co/docs/transformers/main_classes/tokenizer>


分词器负责准备模型的输入。该库包含所有模型的标记器。最多
分词器有两种形式：完整的 python 实现和基于
Rust 库
 [🤗 标记器](https://github.com/huggingface/tokenizers)
 。 “快速”实施允许：


1. 显着的加速，特别是在进行批量标记化时
2. 在原始字符串（字符和单词）和标记空间之间进行映射的其他方法（例如获取
包含给定字符的标记的索引或与给定标记相对应的字符范围）。


基类
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 和
 [PreTrainedTokenizerFast](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
 实现在模型输入中编码字符串输入的常用方法（见下文）以及实例化/保存 python 和
来自本地文件或目录或来自库提供的预训练标记生成器的“快速”标记生成器
（从 HuggingFace 的 AWS S3 存储库下载）。他们俩都依靠
 [PreTrainedTokenizerBase](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
 包含常用方法，以及
 [SpecialTokensMixin](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.SpecialTokensMixin)
 。


[PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 和
 [PreTrainedTokenizerFast](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
 从而实现主要
使用所有分词器的方法：


* 标记化（在子词标记字符串中分割字符串），将标记字符串转换为 id 并返回，以及
编码/解码（即标记化并转换为整数）。
* 以独立于底层结构（BPE、SentencePiece...）的方式向词汇表添加新标记。
* 管理特殊标记（如掩码、句子开头等）：添加它们，将它们分配给属性中
分词器可轻松访问并确保它们在分词过程中不会被分割。


[BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)
 保存的输出
 [PreTrainedTokenizerBase](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
 的编码方法（
 `__call__`
 ,
 `编码加`
 和
 `batch_encode_plus`
 ）并且源自 Python 字典。当分词器是纯Python时
tokenizer，这个类的行为就像一个标准的 python 字典，并保存由以下方法计算的各种模型输入
这些方法（
 `输入ID`
 ,
 `注意力掩码`
 ……）。当分词器是“快速”分词器时（即由
抱脸
 [tokenizers 库](https://github.com/huggingface/tokenizers)
 ），这个类还提供了
几种高级对齐方法，可用于在原始字符串（字符和单词）和字符串之间进行映射
令牌空间（例如，获取包含给定字符的令牌的索引或对应的字符范围
到给定的令牌）。


## 预训练分词器




### 


班级
 

 变压器。
 

 预训练分词器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils.py#L336)



 (
 


 \*\*夸格




 )
 


 参数


* **模型\_最大\_长度**
 （
 `int`
 ,
 *选修的*
 )—
变压器模型输入的最大长度（以令牌数量计）。当分词器是
满载
 [来自\_pretrained()](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained)
 ，这将被设置为
为关联模型存储的值
 `最大模型输入大小`
 （往上看）。如果没有提供值，将
默认为非常大整数（
 `int(1e30)`
 ）。
* **填充\_side**
 （
 `str`
 ,
 *选修的*
 )—
模型应应用填充的一侧。应在[‘右’、‘左’]之间选择。
默认值是从同名的类属性中选取的。
* **截断\_side**
 （
 `str`
 ,
 *选修的*
 )—
模型应应用截断的一侧。应在[‘右’、‘左’]之间选择。
默认值是从同名的类属性中选取的。
* **聊天\_模板**
 （
 `str`
 ,
 *选修的*
 )—
Jinja 模板字符串，将用于格式化聊天消息列表。看
 <https://huggingface.co/docs/transformers/chat_templated>
 以获得完整的描述。
* **模型\_输入\_名称**
 （
 `列表[字符串]`
 ,
 *选修的*
 )—
模型前向传递接受的输入列表（例如
 `“token_type_ids”`
 或者
 `“注意掩码”`
 ）。默认值是从同名的类属性中选取的。
* **bos\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表句子开头的特殊标记。将关联到
 `self.bos_token`
 和
 `self.bos_token_id`
 。
* **eos\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表句子结尾的特殊标记。将关联到
 `self.eos_token`
 和
 `self.eos_token_id`
 。
* **unk\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表词汇表外标记的特殊标记。将关联到
 `self.unk_token`
 和
 `self.unk_token_id`
 。
* **sep\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
在同一输入中分隔两个不同句子的特殊标记（例如 BERT 使用）。将
关联到
 `self.sep_token`
 和
 `self.sep_token_id`
 。
* **填充\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
一种特殊的令牌，用于使令牌数组具有相同的大小以进行批处理。然后将被忽略
注意机制或损失计算。将关联到
 `self.pad_token`
 和
 `self.pad_token_id`
 。
* **cls\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
表示输入类别的特殊标记（例如 BERT 使用）。将关联到
 `self.cls_token`
 和
 `self.cls_token_id`
 。
* **掩码\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表掩码标记的特殊标记（由掩码语言建模预训练目标使用，例如
伯特）。将关联到
 `self.mask_token`
 和
 `self.mask_token_id`
 。
* **额外\_特殊\_令牌**
 （元组或列表
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
附加特殊标记的元组或列表。将它们添加到此处以确保在解码时跳过它们
 `skip_special_tokens`
 设置为 True。如果它们不是词汇表的一部分，则会在末尾添加
的词汇。
* **清理\_up\_标记化\_spaces**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
模型是否应该清除在分割输入文本时添加的空格
标记化过程。
* **分割\_特殊\_代币**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
在标记化过程中是否应分割特殊标记。默认行为是
不分割特殊令牌。这意味着如果
 `<s>`
 是个
 `bos_token`
 ， 然后
 `tokenizer.tokenize("<s>") = ['<s>`
 ]。否则，如果
 `split_special_tokens=True`
 ， 然后
 `tokenizer.tokenize("<s>")`
 将给予
 `['<', 's', '>']`
 。该论点仅支持
 ‘慢’
 暂时使用标记器。


 所有慢标记器的基类。


继承自
 [PreTrainedTokenizerBase](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
 。


处理所有用于标记化和特殊标记的共享方法以及下载/缓存/加载方法
预训练标记器以及将标记添加到词汇表中。


该类还以统一的方式包含在所有标记器之上添加的标记，因此我们不必处理
各种底层词典结构（BPE、sentpiece…）的特定词汇增强方法。


类属性（被派生类覆盖）


* **词汇\_文件\_名称**
 （
 `字典[str, str]`
 ) — 一本字典，其中的键为
 `__init__`
 每个的关键字名称
模型所需的词汇文件，以及关联值，用于保存关联文件的文件名
（细绳）。
* **预训练\_vocab\_files\_map**
 （
 `字典[str，字典[str，str]]`
 ) — 字典中的字典，其中
高级按键是
 `__init__`
 模型所需的每个词汇文件的关键字名称，
低级是
 `快捷名称`
 预训练模型的相关值是
 `网址`
 到
关联的预训练词汇文件。
* **最大\_模型\_输入\_尺寸**
 （
 `字典[str，可选[int]]`
 ) — 一本字典，其中的键为
 `快捷名称`
 预训练模型的数量，以及该模型的序列输入的最大长度作为关联值，
或者
 `无`
 如果模型没有最大输入大小。
* **预训练\_init\_configuration**
 （
 `字典[str，字典[str，任意]]`
 ) — 一本字典，其中的键为
 `快捷名称`
 预训练模型的数据，以及作为关联值的特定参数的字典
传递到
 `__init__`
 加载分词器时此预训练模型的分词器类的方法
与
 [来自\_pretrained()](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained)
 方法。
* **模型\_输入\_名称**
 （
 `列表[str]`
 ) — 模型前向传递中预期的输入列表。
* **填充\_side**
 （
 `str`
 ) — 模型应应用填充的一侧的默认值。
应该
 ``对'``
 或者
 ``左'`
 。
* **截断\_side**
 （
 `str`
 ) — 模型应截断一侧的默认值
应用。应该
 ``对'``
 或者
 ``左'`
 。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L2724)



 (
 


 文本
 
 : 打字.Union[str, 打字.List[str], 打字.List[打字.List[str]]] = 无


文本\_对
 
 : 打字.Union[str, 打字.List[str], 打字.List[打字.List[str]], NoneType] = 无


文本\_目标
 
 : 打字.Union[str, 打字.List[str], 打字.List[打字.List[str]]] = 无


文本\_对\_目标
 
 : 打字.Union[str, 打字.List[str], 打字.List[打字.List[str]], NoneType] = 无


添加\_特殊\_令牌
 
 ：布尔=真


填充
 
 ：打字.Union[bool，str，transformers.utils.generic.PaddingStrategy] = False


截断
 
 ：typing.Union[bool，str，transformers.tokenization\_utils\_base.TruncationStrategy] = None


最长长度
 
 : 打字.Optional[int] = None


跨步
 
 ：整数=0


被\_分割\_成\_单词
 
 ：布尔=假


填充到多个
 
 : 打字.Optional[int] = None


返回\_张量
 
 ：打字.Union[str，transformers.utils.generic.TensorType，NoneType] =无


返回\_token\_type\_ids
 
 : 打字.Optional[bool] = None


返回\_attention\_mask
 
 : 打字.Optional[bool] = None


返回\_overflowing\_tokens
 
 ：布尔=假


返回\_特殊\_令牌\_掩码
 
 ：布尔=假


返回\_offsets\_mapping
 
 ：布尔=假


返回\_length
 
 ：布尔=假


冗长的
 
 ：布尔=真


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

[BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)


 参数




* **text** 
 (
 `str` 
 ,
 `List[str]` 
 ,
 `List[List[str]]` 
 ,
 *optional* 
 ) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
 `is_split_into_words=True` 
 (to lift the ambiguity with a batch of sequences).
* **text\_pair** 
 (
 `str` 
 ,
 `List[str]` 
 ,
 `List[List[str]]` 
 ,
 *optional* 
 ) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
 `is_split_into_words=True` 
 (to lift the ambiguity with a batch of sequences).
* **text\_target** 
 (
 `str` 
 ,
 `List[str]` 
 ,
 `List[List[str]]` 
 ,
 *optional* 
 ) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set
 `is_split_into_words=True` 
 (to lift the ambiguity with a batch of sequences).
* **text\_pair\_target** 
 (
 `str` 
 ,
 `List[str]` 
 ,
 `List[List[str]]` 
 ,
 *optional* 
 ) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set
 `is_split_into_words=True` 
 (to lift the ambiguity with a batch of sequences).
* **add\_special\_tokens** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to add special tokens when encoding the sequences. This will use the underlying
 `PretrainedTokenizerBase.build_inputs_with_special_tokens` 
 function, which defines which tokens are
automatically added to the input ids. This is usefull if you want to add
 `bos` 
 or
 `eos` 
 tokens
automatically.
* **padding** 
 (
 `bool` 
 ,
 `str` 
 or
 [PaddingStrategy](/docs/transformers/v4.35.2/en/internal/file_utils#transformers.utils.PaddingStrategy) 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Activates and controls padding. Accepts the following values:
 
	+ `True` 
	 or
	 `'longest'` 
	 : Pad to the longest sequence in the batch (or no padding if only a single
	sequence if provided).
	+ `'max_length'` 
	 : Pad to a maximum length specified with the argument
	 `max_length` 
	 or to the maximum
	acceptable input length for the model if that argument is not provided.
	+ `False` 
	 or
	 `'do_not_pad'` 
	 (default): No padding (i.e., can output a batch with sequences of different
	lengths).
* **truncation** 
 (
 `bool` 
 ,
 `str` 
 or
 [TruncationStrategy](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy) 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Activates and controls truncation. Accepts the following values:
 
	+ `True` 
	 or
	 `'longest_first'` 
	 : Truncate to a maximum length specified with the argument
	 `max_length` 
	 or
	to the maximum acceptable input length for the model if that argument is not provided. This will
	truncate token by token, removing a token from the longest sequence in the pair if a pair of
	sequences (or a batch of pairs) is provided.
	+ `'only_first'` 
	 : Truncate to a maximum length specified with the argument
	 `max_length` 
	 or to the
	maximum acceptable input length for the model if that argument is not provided. This will only
	truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
	+ `'only_second'` 
	 : Truncate to a maximum length specified with the argument
	 `max_length` 
	 or to the
	maximum acceptable input length for the model if that argument is not provided. This will only
	truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
	+ `False` 
	 or
	 `'do_not_truncate'` 
	 (default): No truncation (i.e., can output batch with sequence lengths
	greater than the model maximum admissible input size).
* **max\_length** 
 (
 `int` 
 ,
 *optional* 
 ) —
Controls the maximum length to use by one of the truncation/padding parameters.
 
 If left unset or set to
 `None` 
 , this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **stride** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) —
If set to a number along with
 `max_length` 
 , the overflowing tokens returned when
 `return_overflowing_tokens=True` 
 will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
* **is\_split\_into\_words** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to
 `True` 
 , the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** 
 (
 `int` 
 ,
 *optional* 
 ) —
If set will pad the sequence to a multiple of the provided value. Requires
 `padding` 
 to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
 `>= 7.5` 
 (Volta).
* **return\_tensors** 
 (
 `str` 
 or
 [TensorType](/docs/transformers/v4.35.2/en/internal/file_utils#transformers.TensorType) 
 ,
 *optional* 
 ) —
If set, will return tensors instead of list of python integers. Acceptable values are:
 
	+ `'tf'` 
	 : Return TensorFlow
	 `tf.constant` 
	 objects.
	+ `'pt'` 
	 : Return PyTorch
	 `torch.Tensor` 
	 objects.
	+ `'np'` 
	 : Return Numpy
	 `np.ndarray` 
	 objects.
* **return\_token\_type\_ids** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer’s default, defined by the
 `return_outputs` 
 attribute.
 
[What are token type IDs?](../glossary#token-type-ids)
* **return\_attention\_mask** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer’s default, defined by the
 `return_outputs` 
 attribute.
 
[What are attention masks?](../glossary#attention-mask)
* **return\_overflowing\_tokens** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with
 `truncation_strategy = longest_first` 
 or
 `True` 
 , an error is raised instead
of returning overflowing tokens.
* **return\_special\_tokens\_mask** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to return special tokens mask information.
* **return\_offsets\_mapping** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to return
 `(char_start, char_end)` 
 for each token.
 
 This is only available on fast tokenizers inheriting from
 [PreTrainedTokenizerFast](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) 
 , if using
Python’s tokenizer, this method will raise
 `NotImplementedError` 
.
* **return\_length** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether or not to return the lengths of the encoded inputs.
* **verbose** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to print more information and warnings.
\*\*kwargs — passed to the
 `self.tokenize()` 
 method


退货


导出常量元数据='未定义';
 

[BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)


导出常量元数据='未定义';


A
 [BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)
 包含以下字段：


* **输入\_ids**
 — 要馈送到模型的令牌 ID 列表。


[什么是输入 ID？](../glossary#input-ids)
* **令牌\_type\_ids**
 — 要馈送到模型的令牌类型 ID 列表（当
 `return_token_type_ids=True`
 或者
如果
 *“令牌\_type\_ids”*
 是在
 `self.model_input_names`
 ）。


[什么是令牌类型 ID？](../glossary#token-type-ids)
* **注意\_mask**
 — 指定模型应关注哪些标记的索引列表（当
 `return_attention_mask=True`
 或者如果
 *“注意\_mask”*
 是在
 `self.model_input_names`
 ）。


[什么是注意力蒙版？](../glossary#attention-mask)
* **溢出\_tokens**
 — 溢出标记序列列表（当
 `最大长度`
 被指定并且
 `return_overflowing_tokens=True`
 ）。
* **num\_truncated\_tokens**
 — 被截断的令牌数量（当
 `最大长度`
 被指定并且
 `return_overflowing_tokens=True`
 ）。
* **特殊\_令牌\_掩码**
 — 0 和 1 的列表，其中 1 指定添加的特殊标记，0 指定
常规序列标记（当
 `add_special_tokens=True`
 和
 `return_special_tokens_mask=True`
 ）。
* **长度**
 — 输入的长度（当
 `return_length=True`
 ）


对一个或多个序列或一对或多对序列进行标记和准备模型的主要方法
序列。




#### 


应用\_聊天\_模板


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L1679)



 (
 


 对话
 
 : Typing.Union[typing.List[typing.Dict[str, str]], ForwardRef('对话')]


聊天\_模板
 
 : 打字.可选[str] =无


添加\_代\_提示
 
 ：布尔=假