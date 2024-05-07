# 分词器实用程序

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/internal/tokenization_utils>
>
> 原始地址：<https://huggingface.co/docs/transformers/internal/tokenization_utils>


本页列出了分词器使用的所有实用函数，主要是类
 [PreTrainedTokenizerBase](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
 实现之间的通用方法
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 和
 [PreTrainedTokenizerFast](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
 和混合
 [SpecialTokensMixin](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.SpecialTokensMixin)
 。


其中大多数仅当您正在研究库中标记器的代码时才有用。


## 预训练分词器库




### 


班级
 

 变压器。
 

 预训练分词器库


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L1543)


（


 \*\*夸格


）


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


 基类为
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 和
 [PreTrainedTokenizerFast](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
 。


处理这两个类的共享（主要是样板）方法。


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


（


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


（


 对话
 
 : Typing.Union[typing.List[typing.Dict[str, str]], ForwardRef('对话')]


聊天\_模板
 
 : 打字.可选[str] =无


添加\_代\_提示
 
 ：布尔=假


标记化
 
 ：布尔=真


填充
 
 ：布尔=假


截断
 
 ：布尔=假


最长长度
 
 : 打字.Optional[int] = None


返回\_张量
 
 ：打字.Union[str，transformers.utils.generic.TensorType，NoneType] =无


\*\*分词器\_kwargs


）
 

 →


导出常量元数据='未定义';
 

`列表[整数]`


 参数


* **对话**
 (Union[List[Dict[str, str]], “Conversation”]) — Conversation 对象或字典列表
“角色”和“内容”键，代表到目前为止的聊天历史记录。
* **聊天\_模板**
 （str，
 *选修的*
 ) — 用于此转换的 Jinja 模板。如果
如果不通过，将使用模型的默认聊天模板。
* **添加\_生成\_提示**
 （布尔，
 *选修的*
 ) — 是否用指示的标记结束提示
助理消息的开始。当您想要从模型生成响应时，这非常有用。
请注意，此参数将传递到聊天模板，因此必须在
使该参数产生任何效果的模板。
* **标记化**
 （
 `布尔`
 ，默认为
 '真实'
 )—
是否对输出进行标记。如果
 ‘假’
 ，输出将是一个字符串。
* **填充**
 （
 `布尔`
 ，默认为
 ‘假’
 )—
是否将序列填充到最大长度。如果 tokenize 为 则无效
 ‘假’
 。
* **截断**
 （
 `布尔`
 ，默认为
 ‘假’
 )—
是否以最大长度截断序列。如果 tokenize 为 则无效
 ‘假’
 。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 )—
用于填充或截断的最大长度（以标记为单位）。如果 tokenize 为 则无效
 ‘假’
 。如果
未指定，标记器的
 `最大长度`
 属性将用作默认值。
* **返回\_张量**
 （
 `str`
 或者
 [TensorType](/docs/transformers/v4.35.2/en/internal/file_utils#transformers.TensorType)
 ,
 *选修的*
 )—
如果设置，将返回特定框架的张量。如果 tokenize 为 则无效
 ‘假’
 。可以接受
值为：
 
+ `'tf'`
：返回TensorFlow
`tf.张量`
对象。
+ `'点'`
：返回PyTorch
`火炬.张量`
对象。
+ `'np'`
：返回 NumPy
`np.ndarray`
对象。
+ `'贾克斯'`
：返回JAX
`jnp.ndarray`
对象。
\*\*tokenizer\_kwargs — 传递给 tokenizer 的附加 kwargs。


退货


导出常量元数据='未定义';
 

`列表[整数]`


导出常量元数据='未定义';


代表到目前为止标记化聊天的令牌 ID 列表，包括控制令牌。这
输出已准备好直接或通过类似方法传递给模型
 `生成()`
 。


转换 Conversation 对象或字典列表
 `“角色”`
 和
 `“内容”`
 令牌列表的键
id。此方法旨在与聊天模型一起使用，并将读取分词器的 chat\_template 属性
确定转换时要使用的格式和控制令牌。当chat\_template为None时，会回退
到班级级别指定的默认聊天模板。




#### 


作为\_target\_tokenizer


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3850)


（
 

 ）


临时设置用于对目标进行编码的分词器。对于关联到的分词器很有用
序列到序列模型需要对标签进行稍微不同的处理。




#### 


批量解码


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3682)


（


 序列
 
 ：typing.Union[typing.List[int]、typing.List[typing.List[int]]、ForwardRef('np.ndarray')、ForwardRef('torch.Tensor')、ForwardRef('tf.Tensor') ]


跳过\_特殊\_令牌
 
 ：布尔=假


清理\_up\_标记化\_空间
 
 : 布尔 = 无


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`列表[str]`


 参数


* **序列**
 （
 `Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`
 )—
标记化输入 ID 列表。可以使用以下方式获得
 `__call__`
 方法。
* **跳过\_特殊\_令牌**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否在解码中删除特殊标记。
* **清理\_up\_标记化\_spaces**
 （
 `布尔`
 ,
 *选修的*
 )—
是否清理标记化空间。如果
 `无`
 ，将默认为
 `self.clean_up_tokenization_spaces`
 。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
将传递给底层模型特定的解码方法。


退货


导出常量元数据='未定义';
 

`列表[str]`


导出常量元数据='未定义';


已解码句子的列表。


通过调用decode将令牌ID列表的列表转换为字符串列表。




#### 


批\_编码\_plus


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3021)


（


 批次\_文本\_或\_文本\_对
 
 ：typing.Union[typing.List[str]、typing.List[typing.Tuple[str, str]]、typing.List[typing.List[str]]、typing.List[typing.Tuple[typing.List[ str]、typing.List[str]]]、typing.List[typing.List[int]]、typing.List[typing.Tuple[typing.List[int]、typing.List[int]]]]


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




* **batch\_text\_or\_text\_pairs** 
 (
 `List[str]` 
 ,
 `List[Tuple[str, str]]` 
 ,
 `List[List[str]]` 
 ,
 `List[Tuple[List[str], List[str]]]` 
 , and for not-fast tokenizers, also
 `List[List[int]]` 
 ,
 `List[Tuple[List[int], List[int]]]` 
 ) —
Batch of sequences or pair of sequences to be encoded. This can be a list of
string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
details in
 `encode_plus` 
 ).
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


对模型进行分词并准备序列列表或序列对列表。


此方法已被弃用，
 `__call__`
 应该使用。


#### 


使用特殊令牌构建输入


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3317)


（


 令牌\_ids\_0
 
 : 打字.List[int]


令牌\_ids\_1
 
 : 打字.可选[打字.列表[int]] =无


）
 

 →


导出常量元数据='未定义';
 

`列表[整数]`


 参数


* **令牌\_ids\_0**
 （
 `列表[整数]`
 ) — 第一个标记化序列。
* **令牌\_ids\_1**
 （
 `列表[整数]`
 ,
 *选修的*
 ) — 第二个标记化序列。


退货


导出常量元数据='未定义';
 

`列表[整数]`


导出常量元数据='未定义';


带有特殊标记的模型输入。


通过连接和构建序列分类任务的序列或一对序列的模型输入
添加特殊标记。


此实现不添加特殊标记，并且应在子类中重写此方法。




#### 


清理\_up\_标记化


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3793)


（


 输出\_字符串
 
 : 字符串


）
 

 →


导出常量元数据='未定义';
 

`str`


 参数


* **输出\_string**
 （
 `str`
 ) — 要清理的文本。


退货


导出常量元数据='未定义';
 

`str`


导出常量元数据='未定义';


清理干净的字符串。


清理简单的英语标记化工件列表，例如标点符号和缩写形式之前的空格。




#### 


将\_tokens\_转换为\_string


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3669)


（


 代币
 
 : 打字.List[str]


）
 

 →


导出常量元数据='未定义';
 

`str`


 参数


* **代币**
 （
 `列表[str]`
 ) — 要加入字符串的标记。


退货


导出常量元数据='未定义';
 

`str`


导出常量元数据='未定义';


加入的令牌。


将一系列标记转换为单个字符串。最简单的方法是
 `" ".join(令牌)`
 但我们
通常希望同时删除子词标记化伪影。




#### 


创建\_token\_type\_ids\_from\_sequences


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3297)


（


 令牌\_ids\_0
 
 : 打字.List[int]


令牌\_ids\_1
 
 : 打字.可选[打字.列表[int]] =无


）
 

 →


导出常量元数据='未定义';
 

`列表[整数]`


 参数


* **令牌\_ids\_0**
 （
 `列表[整数]`
 ) — 第一个标记化序列。
* **令牌\_ids\_1**
 （
 `列表[整数]`
 ,
 *选修的*
 ) — 第二个标记化序列。


退货


导出常量元数据='未定义';
 

`列表[整数]`


导出常量元数据='未定义';


令牌类型 ID。


创建与传递的序列相对应的令牌类型 ID。
 [什么是代币类型
ID？](../glossary#token-type-ids)


如果模型有特殊的构建方法，则应在子类中重写。




#### 


解码


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3716)


（


 令牌\_ids
 
 ：typing.Union[int，typing.List[int]，ForwardRef('np.ndarray')，ForwardRef('torch.Tensor')，ForwardRef('tf.Tensor')]


跳过\_特殊\_令牌
 
 ：布尔=假


清理\_up\_标记化\_空间
 
 : 布尔 = 无


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`str`


 参数


* **令牌\_ids**
 （
 `Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`
 )—
标记化输入 ID 列表。可以使用以下方式获得
 `__call__`
 方法。
* **跳过\_特殊\_令牌**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否在解码中删除特殊标记。
* **清理\_up\_标记化\_spaces**
 （
 `布尔`
 ,
 *选修的*
 )—
是否清理标记化空间。如果
 `无`
 ，将默认为
 `self.clean_up_tokenization_spaces`
 。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
将传递给底层模型特定的解码方法。


退货


导出常量元数据='未定义';
 

`str`


导出常量元数据='未定义';


解码后的句子。


使用分词器和词汇表以及删除特殊选项的选项来转换字符串中的 id 序列
令牌并清理令牌化空间。


类似于做
 `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`
 。




#### 


编码


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L2532)


（


 文本
 
 : 打字.Union[str, 打字.List[str], 打字.List[int]]


文本\_对
 
 : 打字.Union[str, 打字.List[str], 打字.List[int], NoneType] = 无


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


返回\_张量
 
 ：打字.Union[str，transformers.utils.generic.TensorType，NoneType] =无


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`列表[整数]`
 ,
 `火炬.张量`
 ,
 `tf.张量`
 或者
 `np.ndarray`


 参数




* **text** 
 (
 `str` 
 ,
 `List[str]` 
 or
 `List[int]` 
 ) —
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
 `tokenize` 
 method) or a list of integers (tokenized string ids using the
 `convert_tokens_to_ids` 
 method).
* **text\_pair** 
 (
 `str` 
 ,
 `List[str]` 
 or
 `List[int]` 
 ,
 *optional* 
 ) —
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the
 `tokenize` 
 method) or a list of integers (tokenized string ids using the
 `convert_tokens_to_ids` 
 method).
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

 \*\*kwargs — Passed along to the
 `.tokenize()` 
 method.


退货


导出常量元数据='未定义';
 

`列表[整数]`
 ,
 `火炬.张量`
 ,
 `tf.张量`
 或者
 `np.ndarray`


导出常量元数据='未定义';


文本的标记化 ID。


使用分词器和词汇表将字符串转换为 id（整数）序列。


和做一样
 `self.convert_tokens_to_ids(self.tokenize(text))`
 。




#### 


编码\_plus


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L2925)


（


 文本
 
 : 打字.Union[str, 打字.List[str], 打字.List[int]]


文本\_对
 
 : 打字.Union[str, 打字.List[str], 打字.List[int], NoneType] = 无


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
 or
 `List[int]` 
 (the latter only for not-fast tokenizers)) —
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
 `tokenize` 
 method) or a list of integers (tokenized string ids using the
 `convert_tokens_to_ids` 
 method).
* **text\_pair** 
 (
 `str` 
 ,
 `List[str]` 
 or
 `List[int]` 
 ,
 *optional* 
 ) —
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the
 `tokenize` 
 method) or a list of integers (tokenized string ids using the
 `convert_tokens_to_ids` 
 method).
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


对一个序列或一对序列进行标记并为模型准备。


此方法已被弃用，
 `__call__`
 应该使用。


#### 


来自\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L1798)


（


 预训练\_model\_name\_or\_path
 
 : 打字.Union[str, os.PathLike]


\*初始化\_输入


 缓存\_dir
 
 : 打字.Union[str, os.PathLike, NoneType] = None


强制\_下载
 
 ：布尔=假


仅本地\_文件\_
 
 ：布尔=假


代币
 
 : 打字.Union[str, bool, NoneType] = None


修订
 
 : str = '主'


\*\*夸格


）


 参数


* **预训练\_model\_name\_or\_path**
 （
 `str`
 或者
 `os.PathLike`
 )—
可以是：
 
+ 一个字符串，
*型号编号*
托管在 Huggingface.co 上的模型存储库内的预定义标记生成器。
有效的模型 ID 可以位于根级别，例如
`bert-base-uncased`
，或命名空间下
用户或组织名称，例如
`dbmdz/bert-base-德语-大小写`
。
+ 通往a的路径
*目录*
包含分词器所需的词汇文件，例如保存的
使用
[save\_pretrained()](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained)
方法，例如
`./my_model_directory/`
。
+ (
**已弃用**
，不适用于所有派生类）单个已保存词汇表的路径或 url
文件（当且仅当分词器仅需要单个词汇文件（如 Bert 或 XLNet）时），例如
`./my_model_directory/vocab.txt`
。
* **缓存\_dir**
 （
 `str`
 或者
 `os.PathLike`
 ,
 *选修的*
 )—
如果是，则应缓存下载的预定义分词器词汇文件的目录路径
不应使用标准缓存。
* **强制\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否强制（重新）下载词汇文件并覆盖缓存版本（如果它们）
存在。
* **继续\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否删除接收不完整的文件。如果存在此类文件，请尝试恢复下载
存在。
* **代理**
 （
 `字典[str, str]`
 ,
 *选修的*
 )—
由协议或端点使用的代理服务器字典，例如，
 `{'http': 'foo.bar:3128', 'http://主机名': 'foo.bar:4012'}`
 。每个请求都会使用代理。
* **代币**
 （
 `str`
 或者
 *布尔*
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
 ）。
* **本地\_文件\_only**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否仅依赖本地文件而不尝试下载任何文件。
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
* **子文件夹**
 （
 `str`
 ,
 *选修的*
 )—
如果相关文件位于 Huggingface.co 上模型存储库的子文件夹内（例如，
facebook/rag-token-base)，请在此处指定。
* **输入**
 （额外的位置参数，
 *选修的*
 )—
将被传递到 Tokenizer
 `__init__`
 方法。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
将被传递到 Tokenizer
 `__init__`
 方法。可用于设置特殊标记，例如
 `bos_token`
 ,
 `eos_token`
 ,
 `unk_token`
 ,
 `sep_token`
 ,
 `pad_token`
 ,
 `cls_token`
 ,
 `掩码令牌`
 ,
 `additional_special_tokens`
 。请参阅中的参数
 `__init__`
 更多细节。


 实例化一个
 [PreTrainedTokenizerBase](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
 （或派生类）来自预定义的
分词器。


通过
 `令牌=真`
 当您想使用私有模型时需要。


例子：



```
# We can't instantiate directly the base class \*PreTrainedTokenizerBase\* so let's show our examples on a derived class: BertTokenizer
# Download vocabulary from huggingface.co and cache.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Download vocabulary from huggingface.co (user-uploaded) and cache.
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

# If vocabulary files are in a directory (e.g. tokenizer was saved using \*save\_pretrained('./test/saved\_model/')\*)
tokenizer = BertTokenizer.from_pretrained("./test/saved\_model/")

# If the tokenizer uses a single vocabulary file, you can point directly to this file
tokenizer = BertTokenizer.from_pretrained("./test/saved\_model/my\_vocab.txt")

# You can link tokens to special vocabulary when instantiating
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", unk_token="<unk>")
# You should be sure '<unk>' is in the vocabulary when doing that.
# Otherwise use tokenizer.add\_special\_tokens({'unk\_token': '<unk>'}) instead)
assert tokenizer.unk_token == "<unk>"
```


#### 


获取\_特殊\_令牌\_掩码


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3762)


（


 令牌\_ids\_0
 
 : 打字.List[int]


令牌\_ids\_1
 
 : 打字.可选[打字.列表[int]] =无


已经\_有\_特殊\_令牌
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

 [0, 1] 范围内的整数列表


参数


* **令牌\_ids\_0**
 （
 `列表[整数]`
 )—
第一个序列的 id 列表。
* **令牌\_ids\_1**
 （
 `列表[整数]`
 ,
 *选修的*
 )—
第二个序列的 id 列表。
* **已经\_有\_特殊\_令牌**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
标记列表是否已使用模型的特殊标记进行格式化。


退货


导出常量元数据='未定义';
 

 [0, 1] 范围内的整数列表


导出常量元数据='未定义';


1 表示特殊标记，0 表示序列标记。


从未添加特殊标记的标记列表中检索序列 ID。添加时会调用该方法
使用标记器的特殊标记
 `准备模型`
 或者
 `编码加`
 方法。




#### 


获取\_vocab


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L1667)


（
 

 ）
 

 →


导出常量元数据='未定义';
 

`字典[str, int]`


退货


导出常量元数据='未定义';
 

`字典[str, int]`


导出常量元数据='未定义';


词汇。


将词汇表作为标记字典返回到索引。


`tokenizer.get_vocab()[令牌]`
 相当于
 `tokenizer.convert_tokens_to_ids(token)`
 什么时候
 `令牌`
 在里面
词汇。




#### 


软垫


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3124)


（


 编码\_输入
 
 ：typing.Union[transformers.tokenization\_utils\_base.BatchEncoding，typing.List[transformers.tokenization\_utils\_base.BatchEncoding]，typing.Dict[str，typing.List[int]]，typing.Dict[str，打字.List[打字.List[int]]]，打字.List[打字.Dict[str，打字.List[int]]]]


填充
 
 ：打字.Union[bool，str，transformers.utils.generic.PaddingStrategy] = True


最长长度
 
 : 打字.Optional[int] = None


填充到多个
 
 : 打字.Optional[int] = None


返回\_attention\_mask
 
 : 打字.Optional[bool] = None


返回\_张量
 
 ：打字.Union[str，transformers.utils.generic.TensorType，NoneType] =无


冗长的
 
 ：布尔=真


）


 参数


* **编码\_输入**
 （
 [BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)
 ， 列表
 [BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)
 ,
 `字典[str，列表[int]]`
 ,
 `字典[str，列表[列表[int]]`
 或者
 `列表[字典[str，列表[int]]]`
 )—
标记化输入。可以代表一个输入（
 [BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)
 或者
 `字典[str，列表[int]]`
 ）或一批
标记化输入（列表
 [BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)
 ,
 *字典[str，列表[列表[int]]]*
 或者
 *列表[字典[str,
列表[整数]]]*
 ），因此您可以在预处理期间以及 PyTorch 数据加载器中使用此方法
整理功能。
 
 代替
 `列表[整数]`
 你可以有张量（numpy 数组、PyTorch 张量或 TensorFlow 张量），请参阅
上面关于返回类型的注释。
* **填充**
 （
 `布尔`
 ,
 `str`
 或者
 [PaddingStrategy](/docs/transformers/v4.35.2/en/internal/file_utils#transformers.utils.PaddingStrategy)
 ,
 *选修的*
 ，默认为
 '真实'
 )—
选择一种策略来填充返回的序列（根据模型的填充边和填充
索引）其中：
 
+ `真实`
或者
「最长」
：填充到批次中最长的序列（如果只有一个，则不填充
顺序（如果提供）。
+ `'最大长度'`
：填充到参数指定的最大长度
`最大长度`
或最大
如果未提供该参数，则模型可接受的输入长度。
+ `假`
或者
`'不要垫'`
（默认）：无填充（即，可以输出具有不同序列的批次
长度）。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 )—
返回列表的最大长度和可选的填充长度（见上文）。
* **填充\_到\_多个\_的**
 （
 `int`
 ,
 *选修的*
 )—
如果设置，会将序列填充为提供值的倍数。
 
 这对于在具有计算能力的 NVIDIA 硬件上使用 Tensor Core 特别有用
 `>= 7.5`
 （沃尔特）。
* **返回\_attention\_mask**
 （
 `布尔`
 ,
 *选修的*
 )—
是否返回注意力掩码。如果保留默认值，将根据
特定分词器的默认值，由
 `返回输出`
 属性。
 
[什么是注意力蒙版？](../glossary#attention-mask)
* **返回\_张量**
 （
 `str`
 或者
 [TensorType](/docs/transformers/v4.35.2/en/internal/file_utils#transformers.TensorType)
 ,
 *选修的*
 )—
如果设置，将返回张量而不是 python 整数列表。可接受的值为：
 
+ `'tf'`
：返回TensorFlow
`tf.常量`
对象。
+ `'点'`
：返回PyTorch
`火炬.张量`
对象。
+ `'np'`
：返回Numpy
`np.ndarray`
对象。
* **详细**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
是否打印更多信息和警告。


 将单个编码输入或一批​​编码输入填充到预定义长度或最大序列长度
在批次中。


填充侧（左/右）填充标记 ID 在标记器级别定义（使用
 `self.padding_side`
 ,
 `self.pad_token_id`
 和
 `self.pad_token_type_id`
 ）。


请注意，对于快速分词器，使用
 `__call__`
 方法比使用方法编码更快
文本后跟一个电话
 `垫`
 获取填充编码的方法。


如果
 `编码输入`
 传递的是 numpy 数组、PyTorch 张量或 TensorFlow 张量的字典，
结果将使用相同的类型，除非您提供不同的张量类型
 `返回张量`
 。如果是
PyTorch 张量，但是您将丢失张量的特定设备。


#### 


准备模型


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3337)


（


 id
 
 : 打字.List[int]


对\_ids
 
 : 打字.可选[打字.列表[int]] =无


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


前置\_batch\_axis
 
 ：布尔=假


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

[BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)


 参数




* **ids** 
 (
 `List[int]` 
 ) —
Tokenized input ids of the first sequence. Can be obtained from a string by chaining the
 `tokenize` 
 and
 `convert_tokens_to_ids` 
 methods.
* **pair\_ids** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Tokenized input ids of the second sequence. Can be obtained from a string by chaining the
 `tokenize` 
 and
 `convert_tokens_to_ids` 
 methods.
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


准备一个输入 id 序列或一对输入 id 序列，以便模型可以使用它。它
添加特殊标记，如果溢出则截断序列，同时考虑特殊标记和
管理溢出令牌的移动窗口（具有用户定义的步幅）。请注意，对于
 *对\_id*
 不同于
 `无`
 和
 *截断\_策略 = 最长\_优先*
 或者
 '真实'
 , 无法返回
溢出的代币。这样的参数组合会引发错误。




#### 


准备\_seq2seq\_batch


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3893)


（


 源\_文本
 
 : 打字.List[str]


tgt\_文本
 
 : 打字.可选[打字.列表[str]] =无


最长长度
 
 : 打字.Optional[int] = None


最大目标长度
 
 : 打字.Optional[int] = None


填充
 
 : str = '最长'


返回\_张量
 
 ：str=无


截断
 
 ：布尔=真


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

[BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)


 参数


* **src\_texts**
 （
 `列表[str]`
 )—
要总结的文档列表或源语言文本。
* **tgt\_文本**
 （
 `列表`
 ,
 *选修的*
 )—
摘要或目标语言文本列表。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 )—
控制编码器输入的最大长度（要摘要的文档或源语言文本）如果
未设置或设置为
 `无`
 ，如果最大长度为，这将使用预定义的模型最大长度
截断/填充参数之一所需。如果模型没有特定的最大输入长度
（如 XLNet）截断/填充到最大长度将被停用。
* **最大\_目标\_长度**
 （
 `int`
 ,
 *选修的*
 )—
控制解码器输入的最大长度（目标语言文本或摘要）如果未设置或已设置
到
 `无`
 ，这将使用 max\_length 值。
* **填充**
 （
 `布尔`
 ,
 `str`
 或者
 [PaddingStrategy](/docs/transformers/v4.35.2/en/internal/file_utils#transformers.utils.PaddingStrategy)
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
激活并控制填充。接受以下值：
 
+ `真实`
或者
「最长」
：填充到批次中最长的序列（如果只有一个，则不填充
顺序（如果提供）。
+ `'最大长度'`
：填充到参数指定的最大长度
`最大长度`
或最大
如果未提供该参数，则模型可接受的输入长度。
+ `假`
或者
`'不要垫'`
（默认）：无填充（即，可以输出具有不同序列的批次
长度）。
* **返回\_张量**
 （
 `str`
 或者
 [TensorType](/docs/transformers/v4.35.2/en/internal/file_utils#transformers.TensorType)
 ,
 *选修的*
 )—
如果设置，将返回张量而不是 python 整数列表。可接受的值为：
 
+ `'tf'`
：返回TensorFlow
`tf.常量`
对象。
+ `'点'`
：返回PyTorch
`火炬.张量`
对象。
+ `'np'`
：返回Numpy
`np.ndarray`
对象。
* **截断**
 （
 `布尔`
 ,
 `str`
 或者
 [TruncationStrategy](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy)
 ,
 *选修的*
 ，默认为
 '真实'
 )—
激活并控制截断。接受以下值：
 
+ `真实`
或者
`'最长优先'`
：截断到参数指定的最大长度
`最大长度`
或者
如果未提供该参数，则为模型可接受的最大输入长度。这会
逐个截断标记，如果一对，则从该对中最长的序列中删除一个标记
提供序列（或一批对）。
+ `'仅_第一个'`
：截断到参数指定的最大长度
`最大长度`
或到
如果未提供该参数，则模型可接受的最大输入长度。这只会
如果提供了一对序列（或一批序列对），则截断一对序列中的第一个序列。
+ `'仅_第二'`
：截断到参数指定的最大长度
`最大长度`
或到
如果未提供该参数，则模型可接受的最大输入长度。这只会
如果提供了一对序列（或一批序列对），则截断一对序列中的第二个序列。
+ `假`
或者
`'不截断'`
（默认）：无截断（即，可以输出具有序列长度的批次
大于模型最大允许输入大小）。
\*\*kwargs —
附加关键字参数传递给
`self.__call__`
。


退货


导出常量元数据='未定义';
 

[BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)


导出常量元数据='未定义';


A
 [BatchEncoding](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.BatchEncoding)
 包含以下字段：


* **输入\_ids**
 — 要馈送到编码器的令牌 ID 列表。
* **注意\_mask**
 — 指定模型应关注哪些标记的索引列表。
* **标签**
 — tgt\_texts 的令牌 ID 列表。


全套按键
 `[input_ids、attention_mask、标签]`
 ，仅当 tgt\_texts 被传递时才会返回。
否则，input\_ids、attention\_mask 将是唯一的键。


准备用于翻译的模型输入。为了获得最佳效果，请一次翻译一个句子。




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
您想要将标记生成器推送到的存储库的名称。它应包含您的组织名称
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
 `“上传分词器”`
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


 将标记生成器文件上传到 🤗 模型中心。


 例子：



```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Push the tokenizer to your namespace with the name "my-finetuned-bert".
tokenizer.push_to_hub("my-finetuned-bert")

# Push the tokenizer to an organization with the name "my-finetuned-bert".
tokenizer.push_to_hub("huggingface/my-finetuned-bert")
```


#### 


注册\_for\_auto\_class


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3867)



 (
 


 汽车\_类
 
 = '自动标记器'



 )
 


 参数


* **汽车\_class**
 （
 `str`
 或者
 `类型`
 ,
 *选修的*
 ，默认为
 `“自动标记器”`
 )—
用于注册这个新分词器的自动类。


 使用给定的自动类注册此类。这应该只用于自定义分词器，如
库已经映射了
 `自动标记器`
 。


此 API 是实验性的，在下一版本中可能会有一些轻微的重大更改。


#### 


保存\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L2297)



 (
 


 保存\_目录
 
 : 打字.Union[str, os.PathLike]


旧版\_格式
 
 : 打字.Optional[bool] = None


文件名\_前缀
 
 : 打字.可选[str] =无


推送\_到\_集线器
 
 ：布尔=假


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

 一个元组
 `str`


 参数


* **保存\_目录**
 （
 `str`
 或者
 `os.PathLike`
 ) — 将保存标记生成器的目录的路径。
* **旧版\_格式**
 （
 `布尔`
 ,
 *选修的*
 )—
仅适用于快速分词器。如果未设置（默认），会将标记生成器保存在统一的 JSON 中
格式以及遗留格式（如果存在），即具有分词器特定词汇和单独的
添加\_tokens 文件。
 
 如果
 ‘假’
 ，只会以统一的 JSON 格式保存 tokenizer。此格式不兼容
“慢”分词器（不由
 *分词器*
 库），因此分词器将无法
加载到相应的“慢”分词器中。


如果
 '真实'
 ，将以旧格式保存标记生成器。如果“慢”分词器不存在，则一个值
出现错误。
* **文件名\_前缀**
 （
 `str`
 ,
 *选修的*
 )—
添加到标记生成器保存的文件名称的前缀。
* **推送\_到\_hub**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
保存模型后是否将其推送到 Hugging Face 模型中心。您可以指定
您想要推送到的存储库
 `repo_id`
 （将默认为名称
 `保存目录`
 在你的
命名空间）。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
传递给的附加关键字参数
 [push\_to\_hub()](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig.push_to_hub)
 方法。


退货


导出常量元数据='未定义';
 

 一个元组
 `str`


导出常量元数据='未定义';


文件已保存。


保存完整的标记器状态。


此方法确保可以使用以下命令重新加载完整的分词器
 `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`
 类方法..


警告，无这不会保存您在实例化后可能应用到分词器的修改（例如
实例，修改
 `tokenizer.do_lower_case`
 创建后）。




#### 


保存\_词汇


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L2494)



 (
 


 保存\_目录
 
 : 字符串


文件名\_前缀
 
 : 打字.可选[str] =无


）
 

 →


导出常量元数据='未定义';
 

`元组(str)`


 参数


* **保存\_目录**
 （
 `str`
 )—
保存词汇的目录。
* **文件名\_前缀**
 （
 `str`
 ,
 *选修的*
 )—
添加到已保存文件的名称中的可选前缀。


退货


导出常量元数据='未定义';
 

`元组(str)`


导出常量元数据='未定义';


保存的文件的路径。


仅保存分词器的词汇表（词汇+添加的标记）。


此方法不会保存分词器的配置和特殊标记映射。使用
 `_save_pretrained()`
 保存标记器的整个状态。




#### 


标记化


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L2512)



 (
 


 文本
 
 : 字符串


一对
 
 : 打字.可选[str] =无


添加\_特殊\_令牌
 
 ：布尔=假


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`列表[str]`


 参数


* **文本**
 （
 `str`
 )—
要编码的序列。
* **一对**
 （
 `str`
 ,
 *选修的*
 )—
将与第一个序列一起编码的第二个序列。
* **添加\_特殊\_令牌**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否添加与对应模型关联的特殊标记。
* **夸格斯**
 （附加关键字参数，
 *选修的*
 )—
将传递给底层模型特定的编码方法。详情请参阅
 [**称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)


退货


导出常量元数据='未定义';
 

`列表[str]`


导出常量元数据='未定义';


代币列表。


将字符串转换为标记序列，将未知标记替换为
 `unk_token`
 。




#### 


截断\_序列


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L3473)



 (
 


 id
 
 : 打字.List[int]


对\_ids
 
 : 打字.可选[打字.列表[int]] =无


要删除的 num\_tokens\_
 
 ：整数=0


截断\_策略
 
 : Typing.Union[str, Transformers.tokenization\_utils\_base.TruncationStrategy] = '最长\_first'


跨步
 
 ：整数=0


）
 

 →


导出常量元数据='未定义';
 

`元组[列表[int]，列表[int]，列表[int]]`


 参数


* **ID**
 （
 `列表[整数]`
 )—
第一个序列的标记化输入 ID。可以通过链接从字符串中获得
 `标记化`
 和
 `convert_tokens_to_ids`
 方法。
* **对\_id**
 （
 `列表[整数]`
 ,
 *选修的*
 )—
第二个序列的标记化输入 ID。可以通过链接从字符串中获得
 `标记化`
 和
 `convert_tokens_to_ids`
 方法。
* **num\_tokens\_to\_remove**
 （
 `int`
 ,
 *选修的*
 ，默认为 0) —
使用截断策略删除的标记数。
* **截断\_策略**
 （
 `str`
 或者
 [TruncationStrategy](/docs/transformers/v4.35.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy)
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
截断遵循的策略。可：
 
+ `'最长优先'`
：截断到参数指定的最大长度
`最大长度`
或到
如果未提供该参数，则模型可接受的最大输入长度。这将截断
逐个标记，如果一对序列（或一个
提供成对的批次）。
+ `'仅_第一个'`
：截断到参数指定的最大长度
`最大长度`
或到
如果未提供该参数，则模型可接受的最大输入长度。这只会
如果提供了一对序列（或一批序列对），则截断一对序列中的第一个序列。
+ `'仅_第二'`
：截断到参数指定的最大长度
`最大长度`
或到
如果未提供该参数，则模型可接受的最大输入长度。这只会
如果提供了一对序列（或一批序列对），则截断一对序列中的第二个序列。
+ `'不截断'`
（默认）：无截断（即，可以输出序列长度更大的批次
比模型最大允许输入大小）。
* **大步**
 （
 `int`
 ,
 *选修的*
 ，默认为 0) —
如果设置为正数，则返回的溢出令牌将包含来自主程序的一些令牌
返回序列。该参数的值定义附加标记的数量。


退货


导出常量元数据='未定义';
 

`元组[列表[int]，列表[int]，列表[int]]`


导出常量元数据='未定义';


被截断的
 `id`
 ，截断的
 `pair_id`
 和名单
溢出的代币。注：
 *最长\_第一个*
 如果一对，则策略返回溢出令牌的空列表
提供序列（或一批对）。


按照策略就地截断序列对。


## SpecialTokensMixin




### 


班级
 

 变压器。
 

 SpecialTokensMixin


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L795)


（


 冗长的
 
 = 假


\*\*夸格




 )
 


 参数


* **bos\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表句子开头的特殊标记。
* **eos\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表句子结尾的特殊标记。
* **unk\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表词汇表外标记的特殊标记。
* **sep\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
在同一输入中分隔两个不同句子的特殊标记（例如 BERT 使用）。
* **填充\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
一种特殊的令牌，用于使令牌数组具有相同的大小以进行批处理。然后将被忽略
注意机制或损失计算。
* **cls\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
表示输入类别的特殊标记（例如 BERT 使用）。
* **掩码\_token**
 （
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
代表掩码标记的特殊标记（由掩码语言建模预训练目标使用，例如
伯特）。
* **额外\_特殊\_令牌**
 （元组或列表
 `str`
 或者
 `tokenizers.AddedToken`
 ,
 *选修的*
 )—
一个元组或附加标记的列表，将被标记为
 ‘特别’
 ，这意味着他们将
解码时跳过 if
 `skip_special_tokens`
 被设定为
 '真实'
 。


 派生出的 mixin
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 和
 [PreTrainedTokenizerFast](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)
 处理与相关的特定行为
特殊令牌。特别是，该类包含可用于直接访问这些特殊的属性
以独立于模型的方式处理令牌，并允许设置和更新特殊令牌。



#### 


添加\_特殊\_令牌


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L873)



 (
 


 特殊\_tokens\_dict
 
 : Typing.Dict[str, Typing.Union[str, tokenizers.AddedToken]]


替换\_额外\_特殊\_令牌
 
 = 正确


）
 

 →


导出常量元数据='未定义';
 

`int`


 参数


* **特殊\_tokens\_dict**
 （字典
 *字符串*
 到
 *字符串*
 或者
 `tokenizers.AddedToken`
 )—
键应该位于预定义的特殊属性列表中：[
 `bos_token`
 ,
 `eos_token`
 ,
 `unk_token`
 ,
 `sep_token`
 ,
 `pad_token`
 ,
 `cls_token`
 ,
 `掩码令牌`
 ,
 `additional_special_tokens`
 ]。
 
 仅当标记尚未出现在词汇表中时才会添加标记（通过检查标记生成器是否
分配索引
 `unk_token`
 给他们）。
* **替换\_附加\_特殊\_令牌**
 （
 `布尔`
 ,
 *选修的*
 ,, 默认为
 '真实'
 )—
如果
 '真实'
 ，现有的附加特殊令牌列表将被替换为中提供的列表
 `special_tokens_dict`
 。否则，
 `self._additional_special_tokens`
 只是延长了。在前者中
在这种情况下，标记不会从标记器的完整词汇表中删除 - 它们只是被标记
作为非特殊标记。请记住，这仅影响解码期间跳过的标记，而不影响
 `added_tokens_encoder`
 和
 `added_tokens_decoder`
 。这意味着之前的
 `additional_special_tokens`
 仍然是添加的 token，不会被模型分割。


退货


导出常量元数据='未定义';
 

`int`


导出常量元数据='未定义';


添加到词汇表中的标记数量。


将特殊标记（eos、pad、cls 等）的字典添加到编码器并将它们链接到类属性。如果
特殊标记不在词汇表中，它们被添加到词汇表中（从最后一个索引开始索引）
当前词汇）。


当向词汇表中添加新标记时，您应该确保还调整了词汇表的标记嵌入矩阵的大小
模型，使其嵌入矩阵与分词器匹配。


为此，请使用
 [resize\_token\_embeddings()](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings)
 方法。


使用
 `add_special_tokens`
 将确保您的特殊代币可以通过多种方式使用：


* 使用解码时可以跳过特殊标记
 `skip_special_tokens = True`
 。
* 特殊标记由分词器仔细处理（它们永远不会被分割），类似于
 `添加的令牌`
 。
* 您可以使用 tokenizer 类属性轻松引用特殊标记，例如
 `tokenizer.cls_token`
 。这
可以轻松开发与模型无关的训练和微调脚本。


如果可能，已经为提供的预训练模型注册了特殊令牌（例如
 [BertTokenizer](/docs/transformers/v4.35.2/en/model_doc/bert#transformers.BertTokenizer)
`cls_token`
 已注册为 :obj
 *'[CLS]'*
 XLM 的也注册为
 `'</s>'`
 ）。


 例子：



```
# Let's see how to add a new classification token to GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

special_tokens_dict = {"cls\_token": "<CLS>"}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")
# Notice: resize\_token\_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))

assert tokenizer.cls_token == "<CLS>"
```


#### 


添加\_令牌


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L974)



 (
 


 新\_代币
 
 ：typing.Union[str，tokenizers.AddedToken，typing.List[typing.Union[str，tokenizers.AddedToken]]]


特殊\_令牌
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

`int`


 参数


* **新\_代币**
 （
 `str`
 ,
 `tokenizers.AddedToken`
 或列表
 *字符串*
 或者
 `tokenizers.AddedToken`
 )—
仅当词汇表中尚不存在时，才会添加标记。
 `tokenizers.AddedToken`
 包裹一个字符串
令牌让您个性化其行为：此令牌是否只应与单个单词匹配，
该标记是否应去除左侧所有潜在的空白，该标记是否应
去除右侧所有潜在的空格等。
* **特殊\_令牌**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
可用于指定令牌是否是特殊令牌。这主要改变了规范化行为
（例如，像 CLS 或 [MASK] 这样的特殊标记通常不是小写的）。
 
 查看详细信息
 `tokenizers.AddedToken`
 在 HuggingFace 分词器库中。


退货


导出常量元数据='未定义';
 

`int`


导出常量元数据='未定义';


添加到词汇表中的标记数量。


将新标记列表添加到标记生成器类中。如果新的标记不在词汇表中，则将它们添加到
它的索引从当前词汇的长度开始，并且将在标记化之前被隔离
应用算法。因此，添加的标记和来自标记化算法词汇表的标记是
不以同样的方式对待。


请注意，在向词汇表添加新标记时，您应该确保还调整标记嵌入矩阵的大小
模型的嵌入矩阵与分词器匹配。


为此，请使用
 [resize\_token\_embeddings()](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings)
 方法。


 例子：



```
# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

num_added_toks = tokenizer.add_tokens(["new\_tok1", "my\_new-tok2"])
print("We have added", num_added_toks, "tokens")
# Notice: resize\_token\_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))
```


#### 


清理\_特殊\_令牌


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L865)


（
 

 ）


这
 `清理特殊令牌`
 现已弃用，保留向后兼容性，并将在
变形金刚 v5.


## 枚举和命名元组




### 


班级
 

 Transformers.tokenization\_utils\_base。
 

 截断策略


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L138)



 (
 


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



 )
 


可能的值
 `截断`
 论证中
 [PreTrainedTokenizerBase.
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 。对于制表符补全很有用
一个IDE。




### 


班级
 

 变压器。
 

 字符跨度


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L150)



 (
 


 开始
 
 ：整数


结尾
 
 ：整数



 )
 


 参数


* **开始**
 （
 `int`
 ) — 原始字符串中第一个字符的索引。
* **结尾**
 （
 `int`
 ) — 原始字符串中最后一个字符后面的字符的索引。


 原始字符串中的字符跨度。




### 


班级
 

 变压器。
 

 令牌跨度


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/tokenization_utils_base.py#L163)



 (
 


 开始
 
 ：整数


结尾
 
 ：整数



 )
 


 参数


* **开始**
 （
 `int`
 ) — 范围中第一个标记的索引。
* **结尾**
 （
 `int`
 ) — 范围中最后一个标记之后的标记索引。


 编码字符串（标记列表）中的标记范围。