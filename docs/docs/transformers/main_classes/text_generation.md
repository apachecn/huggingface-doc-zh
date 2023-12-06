# 一代

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/main_classes/text_generation>
>
> 原始地址：<https://huggingface.co/docs/transformers/main_classes/text_generation>


每个框架都有一个在各自的框架中实现的文本生成方法
 `GenerationMixin`
 班级：


* 火炬
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.generate）
 实施于
 [GenerationMixin](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin)
 。
* TensorFlow
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.TFGenerationMixin.generate）
 实施于
 [TFGenerationMixin](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.TFGenerationMixin)
 。
* 亚麻/JAX
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.FlaxGenerationMixin.generate）
 实施于
 [FlaxGenerationMixin](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.FlaxGenerationMixin)
 。


无论您选择哪种框架，您都可以使用以下参数参数化生成方法
 [GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)
 类实例。请参阅此类以获取控制行为的生成参数的完整列表
的生成方法。


要了解如何检查模型的生成配置、默认值是什么、如何临时更改参数，
以及如何创建和保存自定义生成配置，请参阅
 [文本生成策略指南](../Generation_strategies)
 。该指南还解释了如何使用相关功能，
就像令牌流一样。


## 生成配置




### 


班级
 

 变压器。
 

 生成配置


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/configuration_utils.py#L40)



 (
 


 \*\*夸格




 )
 


 控制输出长度的参数


* **最长长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 20) —
生成的令牌可以具有的最大长度。对应输入提示的长度+
 `max_new_tokens`
 。它的效果被覆盖
 `max_new_tokens`
 ，如果也设置了。
* **最大\_新\_令牌**
 （
 `int`
 ,
 *选修的*
 )—
要生成的最大令牌数，忽略提示中的令牌数。
* **最小\_长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 0) —
要生成的序列的最小长度。对应输入提示的长度+
 `min_new_tokens`
 。它的效果被覆盖
 `min_new_tokens`
 ，如果也设置了。
* **min\_new\_tokens**
 （
 `int`
 ,
 *选修的*
 )—
要生成的最小令牌数，忽略提示中的令牌数。
* **提前\_停止**
 （
 `布尔`
 或者
 `str`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
控制基于波束的方法（例如波束搜索）的停止条件。它接受以下值：
 '真实'
 ，一旦出现，生成就会停止
 `num_beams`
 完整的候选人；
 ‘假’
 ，其中
应用启发式，当不太可能找到更好的候选者时，生成停止；
 `“从来没有”`
 ，其中波束搜索过程仅在无法找到更好的候选者时停止（规范
波束搜索算法）。
* **最大\_时间(
 `浮动`
 ,**
*选修的*
 )—
允许计算运行的最长时间（以秒为单位）。一代仍将结束
经过分配的时间后的当前通道。


控制所使用的生成策略的参数


* **做\_样本**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否采用抽样；否则使用贪婪解码。
* **num\_beams**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
用于波束搜索的波束数量。 1 表示不进行波束搜索。
* **num\_beam\_groups**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
划分的组数
 `num_beams`
 以确保不同组波束之间的多样性。
 [本文](https://arxiv.org/pdf/1610.02424.pdf)
 更多细节。
* **处罚\_alpha**
 （
 `浮动`
 ,
 *选修的*
 )—
这些值平衡了对比搜索解码中的模型置信度和退化惩罚。
* **使用\_cache**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
模型是否应该使用过去的最后一个键/值注意力（如果适用于模型）来
加快解码速度。


用于操作模型输出 logits 的参数




* **temperature** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
The value used to modulate the next token probabilities.
* **top\_k** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of highest probability vocabulary tokens to keep for top-k-filtering.
* **top\_p** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
 `top_p` 
 or higher are kept for generation.
* **typical\_p** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
Local typicality measures how similar the conditional probability of predicting a target token next is to
the expected conditional probability of predicting a random token next, given the partial text already
generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
add up to
 `typical_p` 
 or higher are kept for generation. See
 [this
paper](https://arxiv.org/pdf/2202.00666.pdf) 
 for more details.
* **epsilon\_cutoff** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
 `epsilon_cutoff` 
 will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
size of the model. See
 [Truncation Sampling as Language Model
Desmoothing](https://arxiv.org/abs/2210.15191) 
 for more details.
* **eta\_cutoff** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
0 and 1, a token is only considered if it is greater than either
 `eta_cutoff` 
 or
 `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))` 
. The latter term is intuitively the expected next token
probability, scaled by
 `sqrt(eta_cutoff)` 
. In the paper, suggested values range from 3e-4 to 2e-3,
depending on the size of the model. See
 [Truncation Sampling as Language Model
Desmoothing](https://arxiv.org/abs/2210.15191) 
 for more details.
* **diversity\_penalty** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a
particular time. Note that
 `diversity_penalty` 
 is only effective if
 `group beam search` 
 is enabled.
* **repetition\_penalty** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
The parameter for repetition penalty. 1.0 means no penalty. See
 [this
paper](https://arxiv.org/pdf/1909.05858.pdf) 
 for more details.
* **encoder\_repetition\_penalty** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
The paramater for encoder\_repetition\_penalty. An exponential penalty on sequences that are not in the
original input. 1.0 means no penalty.
* **length\_penalty** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
likelihood of the sequence (i.e. negative),
 `length_penalty` 
 > 0.0 promotes longer sequences, while
 `length_penalty` 
 < 0.0 encourages shorter sequences.
* **no\_repeat\_ngram\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) —
If set to int > 0, all ngrams of that size can only occur once.
* **bad\_words\_ids(
 `List[List[int]]` 
 ,** 
*optional* 
 ) —
List of list of token ids that are not allowed to be generated. Check
 [NoBadWordsLogitsProcessor](/docs/transformers/v4.35.2/en/internal/generation_utils#transformers.NoBadWordsLogitsProcessor) 
 for further documentation and examples.
* **force\_words\_ids(
 `List[List[int]]`**
 or
 `List[List[List[int]]]` 
 ,
 *optional* 
 ) —
List of token ids that must be generated. If given a
 `List[List[int]]` 
 , this is treated as a simple list of
words that must be included, the opposite to
 `bad_words_ids` 
. If given
 `List[List[List[int]]]` 
 , this
triggers a
 [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081) 
 , where one
can allow different forms of each word.
* **renormalize\_logits** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to renormalize the logits after applying all the logits processors or warpers (including the custom
ones). It’s highly recommended to set this flag to
 `True` 
 as the search algorithms suppose the score logits
are normalized but some logit processors or warpers break the normalization.
* **constraints** 
 (
 `List[Constraint]` 
 ,
 *optional* 
 ) —
Custom constraints that can be added to the generation to ensure that the output will contain the use of
certain tokens as defined by
 `Constraint` 
 objects, in the most sensible way possible.
* **forced\_bos\_token\_id** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `model.config.forced_bos_token_id` 
 ) —
The id of the token to force as the first generated token after the
 `decoder_start_token_id` 
. Useful for
multilingual models like
 [mBART](../model_doc/mbart) 
 where the first generated token needs to be the target
language token.
* **forced\_eos\_token\_id** 
 (
 `Union[int, List[int]]` 
 ,
 *optional* 
 , defaults to
 `model.config.forced_eos_token_id` 
 ) —
The id of the token to force as the last generated token when
 `max_length` 
 is reached. Optionally, use a
list to set multiple
 *end-of-sequence* 
 tokens.
* **remove\_invalid\_values** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `model.config.remove_invalid_values` 
 ) —
Whether to remove possible
 *nan* 
 and
 *inf* 
 outputs of the model to prevent the generation method to crash.
Note that using
 `remove_invalid_values` 
 can slow down generation.
* **exponential\_decay\_length\_penalty** 
 (
 `tuple(int, float)` 
 ,
 *optional* 
 ) —
This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
generated. The tuple shall consist of:
 `(start_index, decay_factor)` 
 where
 `start_index` 
 indicates where
penalty starts and
 `decay_factor` 
 represents the factor of exponential decay
* **suppress\_tokens** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
A list of tokens that will be suppressed at generation. The
 `SupressTokens` 
 logit processor will set their
log probs to
 `-inf` 
 so that they are not sampled.
* **begin\_suppress\_tokens** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
A list of tokens that will be suppressed at the beginning of the generation. The
 `SupressBeginTokens` 
 logit
processor will set their log probs to
 `-inf` 
 so that they are not sampled.
* **forced\_decoder\_ids** 
 (
 `List[List[int]]` 
 ,
 *optional* 
 ) —
A list of pairs of integers which indicates a mapping from generation indices to token indices that will be
forced before sampling. For example,
 `[[1, 123]]` 
 means the second generated token will always be a token
of index 123.
* **sequence\_bias** 
 (
 `Dict[Tuple[int], float]` 
 ,
 *optional* 
 )) —
Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
sequence being selected, while negative biases do the opposite. Check
 [SequenceBiasLogitsProcessor](/docs/transformers/v4.35.2/en/internal/generation_utils#transformers.SequenceBiasLogitsProcessor) 
 for further documentation and examples.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 ) —
The guidance scale for classifier free guidance (CFG). CFG is enabled by setting
 `guidance_scale > 1` 
.
Higher guidance scale encourages the model to generate samples that are more closely linked to the input
prompt, usually at the expense of poorer quality.
* **low\_memory** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Switch to sequential topk for contrastive search to reduce peak memory. Used with contrastive search.


定义“generate”输出变量的参数


* **num\_return\_sequences(
 `int`
 ,**
*选修的*
 ，默认为 1) —
批次中每个元素独立计算的返回序列的数量。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 退货不足
张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回的张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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


可以在生成时使用的特殊令牌


* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **bos\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *序列开始*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。


编码器-解码器模型独有的生成参数


* **编码器\_no\_repeat\_ngram\_size**
 （
 `int`
 ,
 *选修的*
 ，默认为 0) —
如果设置为 int > 0，则出现在该大小的所有 ngram
 `encoder_input_ids`
 不能发生在
 `解码器输入ID`
 。
* **解码器\_start\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
如果编码器-解码器模型使用与以下不同的标记开始解码
 *老板*
 ，该令牌的 id。


[辅助生成](https


* **num\_assistant\_tokens**
 （
 `int`
 ,
 *选修的*
 ，默认为 5) —
定义数量
 *投机代币*
 之前应由辅助模型生成
在每次迭代时由目标模型检查。更高的值
 `num_assistant_tokens`
 使一代人
更多的
 *推测*
 ：如果辅助模型性能良好，则可以达到更大的加速，如果辅助模型
模型需要大量修正，达到较低的加速比。
* **num\_assistant\_tokens\_schedule**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `“启发式”`
 )—
定义在推理过程中更改最大辅助令牌的时间表。
 
+ `"_启发式_`
: 当所有
*推测*
token正确，增加
`num_assistant_tokens`
由 2 个其他
减少 1
+ `“常数”`
:
`num_assistant_tokens`
在生成期间保持不变


外卡


保存生成任务配置的类。 A
 `生成`
 call支持以下生成方式
对于文本解码器、文本到文本、语音到文本和视觉到文本模型：


* *贪心解码*
 通过致电
 [greedy\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.greedy_search)
 如果
 `num_beams=1`
 和
 `do_sample=False`
* *对比搜索*
 通过致电
 [contrastive\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.contrastive_search)
 如果
 `penalty_alpha>0。`
 和
 `top_k>1`
* *多项抽样*
 通过致电
 [样本（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.sample）
 如果
 `num_beams=1`
 和
 `do_sample=True`
* *波束搜索解码*
 通过致电
 [beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_search)
 如果
 `num_beams>1`
 和
 `do_sample=False`
* *波束搜索多项式采样*
 通过致电
 [beam\_sample()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_sample)
 如果
 `num_beams>1`
 和
 `do_sample=True`
* *多样化波束搜索解码*
 通过致电
 [group\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.group_beam_search)
 ， 如果
 `num_beams>1`
 和
 `num_beam_groups>1`
* *约束波束搜索解码*
 通过致电
 [constrained\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.constrained_beam_search)
 ， 如果
 `约束！=无`
 或者
 `force_words_ids！=无`
**辅助解码*
 通过致电
 `辅助解码()`
 ， 如果
 `助理模型`
 被传递给
 `.generate()`


您不需要直接调用上述任何方法。将自定义参数值传递给“.generate()”。学习
有关解码策略的更多信息，请参阅
 [文本生成策略指南](../Generation_strategies)
 。



#### 


来自\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/configuration_utils.py#L575)



 (
 


 预训练\_model\_name
 
 : 打字.Union[str, os.PathLike]


配置\_文件\_名称
 
 : 打字.Union[str, os.PathLike, NoneType] = None


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
 

 →


导出常量元数据='未定义';
 

[GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)


 参数


* **预训练\_model\_name**
 （
 `str`
 或者
 `os.PathLike`
 )—
这可以是：
 
+ 一个字符串，
*型号编号*
托管在模型存储库内的预训练模型配置
拥抱脸.co。有效的模型 ID 可以位于根级别，例如
`bert-base-uncased`
， 或者
以用户或组织名称命名，例如
`dbmdz/bert-base-德语-大小写`
。
+ 一条通往 a 的路径
*目录*
包含使用保存的配置文件
save\_pretrained()
方法，例如
`./my_model_directory/`
。
* **配置\_文件\_名称**
 （
 `str`
 或者
 `os.PathLike`
 ,
 *选修的*
 ，默认为
 `“ Generation_config.json”`
 )—
要从中加载的生成配置 JSON 文件的名称
 `预训练模型名称`
 。
* **缓存\_dir**
 （
 `str`
 或者
 `os.PathLike`
 ,
 *选修的*
 )—
如果是，则应缓存下载的预训练模型配置的目录路径
不应使用标准缓存。
* **强制\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
如果出现以下情况，是否强制（重新）下载配置文件并覆盖缓存的版本
它们存在。
* **继续\_下载**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否删除未完整接收的文件。如果存在此类文件，则尝试恢复下载
存在。
* **代理**
 （
 `字典[str, str]`
 ,
 *选修的*
 )—
由协议或端点使用的代理服务器字典，例如，
 `{'http': 'foo.bar:3128', 'http://主机名': 'foo.bar:4012'}。`
 每个请求都会使用代理。
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
* **返回\_未使用\_kwargs**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
如果
 ‘假’
 ，那么这个函数只返回最终的配置对象。
 
 如果
 '真实'
 ，那么这个函数返回一个
 `元组（配置，未使用的_kwargs）`
 在哪里
 *未使用\_kwargs*
 是一个
由键/值对组成的字典，其键不是配置属性：即
部分
 `夸格斯`
 尚未用于更新
 `配置`
 否则会被忽略。
* **子文件夹**
 （
 `str`
 ,
 *选修的*
 ，默认为
 `````
 )—
如果相关文件位于 Huggingface.co 上模型存储库的子文件夹内，您可以
在此指定文件夹名称。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
作为配置属性的任何键的 kwargs 中的值将用于覆盖加载的
价值观。有关键/值对的行为，其键为
 *不是*
 配置属性受控制
由
 `return_unused_kwargs`
 关键字参数。


退货


导出常量元数据='未定义';
 

[GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)


导出常量元数据='未定义';


从该预训练模型实例化的配置对象。


实例化一个
 [GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)
 来自生成配置文件。


 例子：



```
>>> from transformers import GenerationConfig

>>> # Download configuration from huggingface.co and cache.
>>> generation_config = GenerationConfig.from_pretrained("gpt2")

>>> # E.g. config was saved using \*save\_pretrained('./test/saved\_model/')\*
>>> generation_config.save_pretrained("./test/saved\_model/")
>>> generation_config = GenerationConfig.from_pretrained("./test/saved\_model/")

>>> # You can also specify configuration names to your generation configuration file
>>> generation_config.save_pretrained("./test/saved\_model/", config_file_name="my\_configuration.json")
>>> generation_config = GenerationConfig.from_pretrained("./test/saved\_model/", "my\_configuration.json")

>>> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
>>> # arguments to `.from\_pretrained()`. Be mindful that typos and unused arguments will be ignored
>>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
...     "gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
... )
>>> generation_config.top_k
1

>>> unused_kwargs
{'foo': False}
```


#### 


来自\_model\_config


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/configuration_utils.py#L892)



 (
 


 模型\_config
 
 ：预训练配置


）
 

 →


导出常量元数据='未定义';
 

[GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)


 参数


* **模型\_config**
 （
 `预训练配置`
 )—
将用于实例化生成配置的模型配置。


退货


导出常量元数据='未定义';
 

[GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)


导出常量元数据='未定义';


从这些参数实例化的配置对象。


实例化一个
 [GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)
 从一个
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 。此功能对于转换旧版很有用
 [PretrainedConfig](/docs/transformers/v4.35.2/en/main_classes/configuration#transformers.PretrainedConfig)
 对象，可能包含生成参数，到一个独立的
 [GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)
 。




#### 


保存\_预训练


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/configuration_utils.py#L496)



 (
 


 保存\_目录
 
 : 打字.Union[str, os.PathLike]


配置\_文件\_名称
 
 : 打字.Union[str, os.PathLike, NoneType] = None


推送\_到\_集线器
 
 ：布尔=假


\*\*夸格




 )
 


 参数


* **保存\_目录**
 （
 `str`
 或者
 `os.PathLike`
 )—
配置 JSON 文件的保存目录（如果不存在则创建）。
* **配置\_文件\_名称**
 （
 `str`
 或者
 `os.PathLike`
 ,
 *选修的*
 ，默认为
 `“ Generation_config.json”`
 )—
要保存的生成配置 JSON 文件的名称
 `保存目录`
 。
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


 将生成配置对象保存到目录中
 `保存目录`
 ，以便可以使用重新加载
 [来自\_pretrained()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig.from_pretrained)
 类方法。


## 世代混音




### 


班级
 

 变压器。
 

 世代混音


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L491)


（
 

 ）


包含自动回归文本生成的所有函数的类，用作 mixin
 [PreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.PreTrainedModel)
 。


班级暴露
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.generate）
 ，可用于：


* *贪心解码*
 通过致电
 [greedy\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.greedy_search)
 如果
 `num_beams=1`
 和
 `do_sample=False`
* *对比搜索*
 通过致电
 [contrastive\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.contrastive_search)
 如果
 `penalty_alpha>0`
 和
 `top_k>1`
* *多项抽样*
 通过致电
 [样本（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.sample）
 如果
 `num_beams=1`
 和
 `do_sample=True`
* *波束搜索解码*
 通过致电
 [beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_search)
 如果
 `num_beams>1`
 和
 `do_sample=False`
* *波束搜索多项式采样*
 通过致电
 [beam\_sample()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_sample)
 如果
 `num_beams>1`
 和
 `do_sample=True`
* *多样化波束搜索解码*
 通过致电
 [group\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.group_beam_search)
 ， 如果
 `num_beams>1`
 和
 `num_beam_groups>1`
* *约束波束搜索解码*
 通过致电
 [constrained\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.constrained_beam_search)
 ， 如果
 `约束！=无`
 或者
 `force_words_ids！=无`


您不需要直接调用上述任何方法。将自定义参数值传递给“generate”。到
了解有关解码策略的更多信息，请参阅
 [文本生成策略指南](../Generation_strategies)
 。



#### 


产生


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L1350)



 (
 


 输入
 
 : 打字.可选[火炬.张量] =无


生成\_config
 
 : 打字.可选[transformers. Generation.configuration\_utils.GenerationConfig] = None


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


前缀\_允许\_令牌\_fn
 
 : 打字.Union[打字.Callable[[int, torch.Tensor],打字.List[int]], NoneType] = None


已同步\_gpus
 
 : 打字.Optional[bool] = None


助理\_模特
 
 : 打字.Optional[ForwardRef('PreTrainedModel')] = None


流光
 
 : 打字.Optional[ForwardRef('BaseStreamer')] = None


否定\_prompt\_ids
 
 : 打字.可选[火炬.张量] =无


负面\_提示\_注意\_掩码
 
 : 打字.可选[火炬.张量] =无


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

[模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 或者
 `torch.LongTensor`


 参数


* **输入**
 （
 `火炬.张量`
 根据模态的不同形状也不同，
 *选修的*
 )—
用作生成提示或编码器的模型输入的序列。如果
 `无`
 这
方法将其初始化为
 `bos_token_id`
 批量大小为 1。对于仅解码器模型
 `输入`
 应该采用以下格式
 `输入ID`
 。对于编码器-解码器模型
 *输入*
 可以代表任何一个
 `输入ID`
 ,
 `输入值`
 ,
 `输入特征`
 ， 或者
 `像素值`
 。
* **生成\_config**
 （
 `~ Generation.GenerationConfig`
 ,
 *选修的*
 )—
用作生成调用的基本参数化的生成配置。
 `**夸格斯`
 传递生成匹配的属性
 `一代配置`
 将覆盖它们。如果
 `一代配置`
 未提供，将使用默认值，其中有以下加载
优先级：1）从
 ` Generation_config.json`
 模型文件（如果存在）； 2）从模型
配置。请注意，未指定的参数将继承
 [GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)
 的
默认值，应检查其文档以参数化生成。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
自定义 logits 处理器，补充由参数构建的默认 logits 处理器
生成配置。如果传递的 logit 处理器已经使用参数或
生成配置抛出错误。此功能适用于高级用户。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
自定义停止标准，补充由参数和a构建的默认停止标准
生成配置。如果通过了已使用参数或
生成配置抛出错误。如果您的停止标准取决于
 `分数`
 输入，制作
确保你通过
 `return_dict_in_generate=True，output_scores=True`
 到
 `生成`
 。这个功能是
适合高级用户。
* **前缀\_允许\_令牌\_fn**
 （
 `Callable[[int, torch.Tensor], List[int]]`
 ,
 *选修的*
 )—
如果提供，此函数将波束搜索限制为仅在每一步允许的标记。如果不
前提是不施加任何限制。该函数有 2 个参数：批次 ID
 `batch_id`
 和
 `输入ID`
 。它必须返回一个列表，其中包含条件允许的下一代步骤的令牌
在批次 ID 上
 `batch_id`
 以及之前生成的token
 `inputs_ids`
 。这个论点很有用
对于以前缀为条件的约束生成，如中所述
 [自回归实体
检索](https://arxiv.org/abs/2010.00904)
 。
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 )—
是否继续运行 while 循环直到 max\_length。除非被覆盖，否则该标志将被设置为
 '真实'
 在 DeepSpeed ZeRO Stage 3 多 GPU 环境下，以避免在一个 GPU 完成时挂起
在其他 GPU 之前生成。否则它会被设置为
 ‘假’
 。
* **助理\_模特**
 （
 `预训练模型`
 ,
 *选修的*
 )—
可用于加速生成的辅助模型。助理模型必须具有准确的
相同的分词器。使用辅助模型预测候选标记时实现了加速
比使用您调用生成的模型运行生成要快得多。因此，
助理模型应该小得多。
* **流光**
 （
 `BaseStreamer`
 ,
 *选修的*
 )—
Streamer 对象将用于流式传输生成的序列。生成的token被传递
通过
 `streamer.put(token_ids)`
 流媒体负责任何进一步的处理。
* **否定\_prompt\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 ,
 *选修的*
 )—
某些处理器（例如 CFG）需要负面提示。批次大小必须与输入批次匹配
尺寸。这是一项实验性功能，未来版本中可能会发生重大 API 更改。
* **负面\_prompt\_attention\_mask**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 ,
 *选修的*
 )—
注意\_mask
 `否定提示 ID`
 。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
临时参数化
 `生成配置`
 和/或其他特定于模型的 kwargs
转发至
 ‘前进’
 模型的功能。如果模型是编码器-解码器模型，则编码器
特定的 kwargs 不应加前缀，解码器特定的 kwargs 应加前缀
 *解码器\_*
 。


退货


导出常量元数据='未定义';
 

[模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 或者
 `torch.LongTensor`


导出常量元数据='未定义';


A
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 （如果
 `return_dict_in_generate=True`
 或当
 `config.return_dict_in_generate=True`
 ）或一个
 `torch.FloatTensor`
 。


如果模型是
 *不是*
 编码器-解码器模型（
 `model.config.is_encoder_decoder=False`
 ），可能的
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 类型有：


* [GreedySearchDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.GreedySearchDecoderOnlyOutput)
 ,
* [SampleDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleDecoderOnlyOutput)
 ,
* [BeamSearchDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.BeamSearchDecoderOnlyOutput)
 ,
* [BeamSampleDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.BeamSampleDecoderOnlyOutput)


如果模型是编码器-解码器模型（
 `model.config.is_encoder_decoder=True`
 ），可能的
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 类型有：


* [GreedySearchEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.GreedySearchEncoderDecoderOutput)
 ,
* [SampleEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleEncoderDecoderOutput)
 ,
* [BeamSearchEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.BeamSearchEncoderDecoderOutput)
 ,
* [BeamSampleEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.BeamSampleEncoderDecoderOutput)


为具有语言建模头的模型生成令牌 ID 序列。


大多数发电控制参数设置在
 `一代配置`
 如果没有通过，将被设置为
模型的默认生成配置。您可以覆盖任何
 `一代配置`
 通过传递相应的
生成（）的参数，例如
 `.generate(输入，num_beams=4, do_sample=True)`
 。


有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


#### 


计算\_transition\_scores


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L1067)



 (
 


 序列
 
 ：张量


分数
 
 : 打字.元组[火炬.张量]


梁\_索引
 
 : 打字.可选[火炬.张量] =无


标准化\_logits
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

`火炬.张量`


 参数


* **序列**
 （
 `torch.LongTensor`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或者
如果所有批次提前完成，则时间会更短
 `eos_token_id`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
 )—
每个生成步骤中每个词汇标记的转换分数。梁跃迁分数包括
以先前生成的标记的 log softmax 为条件的标记的对数概率的元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），其中
每个形状张量
 `(batch_size*num_beams, config.vocab_size)`
 。
* **梁\_索引**
 （
 `torch.LongTensor`
 ,
 *选修的*
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。仅当出现以下情况时才需要
 `num_beams>1`
 在
生成时间。
* **标准化\_logits**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否标准化 logits（由于遗留原因，可能未标准化）。


退货


导出常量元数据='未定义';
 

`火炬.张量`


导出常量元数据='未定义';


A
 `火炬.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 含有
转换分数 (logits)


计算给定生成分数（和波束索引，如果波束搜索是
用过的）。这是一种在生成时快速获取所选令牌分数的便捷方法。


 例子：



```
>>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
>>> import numpy as np

>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer.pad_token_id = tokenizer.eos_token_id
>>> inputs = tokenizer(["Today is"], return_tensors="pt")

>>> # Example 1: Print the scores for each token generated with Greedy Search
>>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, normalize_logits=True
... )
>>> # input\_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
>>> # encoder-decoder models, like BART or T5.
>>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
>>> generated_tokens = outputs.sequences[:, input_length:]
>>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
...     # | token | token string | logits | probability
...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
|   262 |  the     | -1.414 | 24.33%
|  1110 |  day     | -2.609 | 7.36%
|   618 |  when    | -2.010 | 13.40%
|   356 |  we      | -1.859 | 15.58%
|   460 |  can     | -2.508 | 8.14%

>>> # Example 2: Reconstruct the sequence scores from Beam Search
>>> outputs = model.generate(
...     **inputs,
...     max_new_tokens=5,
...     num_beams=4,
...     num_return_sequences=4,
...     return_dict_in_generate=True,
...     output_scores=True,
... )
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
... )
>>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
>>> # Tip: recomputing the scores is only guaranteed to match with `normalize\_logits=False`. Depending on the
>>> # use case, you might want to recompute it with `normalize\_logits=True`.
>>> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
>>> length_penalty = model.generation_config.length_penalty
>>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
>>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
True
```


#### 


贪婪\_搜索


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L2353)



 (
 


 输入\_ids
 
 : 长张量


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


最长长度
 
 : 打字.Optional[int] = None


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


输出\_attentions
 
 : 打字.Optional[bool] = None


输出\_隐藏\_状态
 
 : 打字.Optional[bool] = None


输出\_分数
 
 : 打字.Optional[bool] = None


返回\_dict\_in\_generate
 
 : 打字.Optional[bool] = None


已同步\_gpus
 
 ：布尔=假


流光
 
 : 打字.Optional[ForwardRef('BaseStreamer')] = None


\*\*模型\_kwargs




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 用于修改每个生成步骤中应用的语言建模头的预测分数。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
一个实例
 [StoppingCriteriaList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteriaList)
 。派生类的实例列表
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 用于判断生成循环是否应该停止。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 20) —
 **已弃用**
 。使用
 `logits_processor`
 或者
 `停止标准`
 直接限制生成的数量
代币。要生成的序列的最大长度。
* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 在下面
返回张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否继续运行 while 循环直到 max\_length （ZeRO 阶段 3 需要）
* **流光**
 （
 `BaseStreamer`
 ,
 *选修的*
 )—
Streamer 对象将用于流式传输生成的序列。生成的token被传递
通过
 `streamer.put(token_ids)`
 流媒体负责任何进一步的处理。
模型\_kwargs —
其他特定于模型的关键字参数将转发到
 ‘前进’
 模型的功能。
如果模型是编码器-解码器模型，则 kwargs 应包括
 `编码器输出`
 。


 使用语言建模头为模型生成令牌 ID 序列
 **贪心解码**
 并且可以是
用于文本解码器、文本到文本、语音到文本和视觉到文本模型。


大多数情况下，您不需要调用
 [greedy\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.greedy_search)
 直接地。使用生成（）
反而。有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


例子：



```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForCausalLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     StoppingCriteriaList,
...     MaxLengthCriteria,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> # set pad\_token\_id to eos\_token\_id because GPT2 does not have a PAD token
>>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

>>> input_prompt = "It might be possible to"
>>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

>>> # instantiate logits processors
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
...     ]
... )
>>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

>>> outputs = model.greedy_search(
...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
["It might be possible to get a better understanding of the nature of the problem, but it's not"]
```


#### 


样本


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L2612)



 (
 


 输入\_ids
 
 : 长张量


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


逻辑\_warper
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


最长长度
 
 : 打字.Optional[int] = None


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


输出\_attentions
 
 : 打字.Optional[bool] = None


输出\_隐藏\_状态
 
 : 打字.Optional[bool] = None


输出\_分数
 
 : 打字.Optional[bool] = None


返回\_dict\_in\_generate
 
 : 打字.Optional[bool] = None


已同步\_gpus
 
 ：布尔=假


流光
 
 : 打字.Optional[ForwardRef('BaseStreamer')] = None


\*\*模型\_kwargs


）
 

 →


导出常量元数据='未定义';
 

[SampleDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleDecoderOnlyOutput)
 ,
 [SampleEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleEncoderDecoderOutput)
 或者
 `torch.LongTensor`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 用于修改每个生成步骤中应用的语言建模头的预测分数。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
一个实例
 [StoppingCriteriaList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteriaList)
 。派生类的实例列表
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 用于判断生成循环是否应该停止。
* **logits\_warper**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 用过的
扭曲多项式之前应用的语言建模头的预测分数分布
在每个生成步骤进行采样。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 20) —
 **已弃用**
 。使用
 `logits_processor`
 或者
 `停止标准`
 直接限制生成的数量
代币。要生成的序列的最大长度。
* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 在下面
返回张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否继续运行 while 循环直到 max\_length （ZeRO 阶段 3 需要）
* **流光**
 （
 `BaseStreamer`
 ,
 *选修的*
 )—
Streamer 对象将用于流式传输生成的序列。生成的token被传递
通过
 `streamer.put(token_ids)`
 流媒体负责任何进一步的处理。
模型\_kwargs —
其他特定于模型的 kwargs 将转发给
 ‘前进’
 模型的功能。如果型号是
kwargs 应包含的编码器-解码器模型
 `编码器输出`
 。


退货


导出常量元数据='未定义';
 

[SampleDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleDecoderOnlyOutput)
 ,
 [SampleEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleEncoderDecoderOutput)
 或者
 `torch.LongTensor`


导出常量元数据='未定义';


A
 `torch.LongTensor`
 包含生成的令牌（默认行为）或
 [SampleDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleDecoderOnlyOutput)
 如果
 `model.config.is_encoder_decoder=False`
 和
 `return_dict_in_generate=True`
 或一个
 [SampleEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.SampleEncoderDecoderOutput)
 如果
 `model.config.is_encoder_decoder=True`
 。


使用语言建模头为模型生成令牌 ID 序列
 **多项抽样**
 和
可用于文本解码器、文本到文本、语音到文本和视觉到文本模型。


大多数情况下，您不需要调用
 [样本（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.sample）
 直接地。请改用generate()。
有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


例子：



```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForCausalLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     TopKLogitsWarper,
...     TemperatureLogitsWarper,
...     StoppingCriteriaList,
...     MaxLengthCriteria,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")

>>> # set pad\_token\_id to eos\_token\_id because GPT2 does not have a EOS token
>>> model.config.pad_token_id = model.config.eos_token_id
>>> model.generation_config.pad_token_id = model.config.eos_token_id

>>> input_prompt = "Today is a beautiful day, and"
>>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

>>> # instantiate logits processors
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
...     ]
... )
>>> # instantiate logits processors
>>> logits_warper = LogitsProcessorList(
...     [
...         TopKLogitsWarper(50),
...         TemperatureLogitsWarper(0.7),
...     ]
... )

>>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

>>> torch.manual_seed(0)
>>> outputs = model.sample(
...     input_ids,
...     logits_processor=logits_processor,
...     logits_warper=logits_warper,
...     stopping_criteria=stopping_criteria,
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
```


#### 


光束\_搜索


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L2894)



 (
 


 输入\_ids
 
 : 长张量


光束\_scorer
 
 : 光束记录仪


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


最长长度
 
 : 打字.Optional[int] = None


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


输出\_attentions
 
 : 打字.Optional[bool] = None


输出\_隐藏\_状态
 
 : 打字.Optional[bool] = None


输出\_分数
 
 : 打字.Optional[bool] = None


返回\_dict\_in\_generate
 
 : 打字.Optional[bool] = None


已同步\_gpus
 
 ：布尔=假


\*\*模型\_kwargs




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **光束\_scorer**
 （
 `光束评分器`
 )—
的派生实例
 [BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 定义了梁假设如何构建、存储和
在生成期间排序。有关更多信息，请参阅文档
 [BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 应该阅读。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 用于修改每个生成步骤中应用的语言建模头的预测分数。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
一个实例
 [StoppingCriteriaList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteriaList)
 。派生类的实例列表
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 用于判断生成循环是否应该停止。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 20) —
 **已弃用**
 。使用
 `logits_processor`
 或者
 `停止标准`
 直接限制生成的数量
代币。要生成的序列的最大长度。
* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 在下面
返回张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否继续运行 while 循环直到 max\_length （ZeRO 阶段 3 需要）
模型\_kwargs —
其他特定于模型的 kwargs 将转发给
 ‘前进’
 模型的功能。如果型号是
kwargs 应包含的编码器-解码器模型
 `编码器输出`
 。


 使用语言建模头为模型生成令牌 ID 序列
 **波束搜索解码**
 和
可用于文本解码器、文本到文本、语音到文本和视觉到文本模型。


大多数情况下，您不需要调用
 [beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_search)
 直接地。使用生成（）
反而。有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


例子：



```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     BeamSearchScorer,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


>>> # lets run beam search using 3 beams
>>> num_beams = 3
>>> # define decoder start token ids
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> # add encoder\_outputs to model keyword arguments
>>> model_kwargs = {
...     "encoder\_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }

>>> # instantiate beam scorer
>>> beam_scorer = BeamSearchScorer(
...     batch_size=1,
...     num_beams=num_beams,
...     device=model.device,
... )

>>> # instantiate logits processors
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
...     ]
... )

>>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt bist du?']
```


#### 


光束\_样本


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L3217)



 (
 


 输入\_ids
 
 : 长张量


光束\_scorer
 
 : 光束记录仪


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


逻辑\_warper
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


最长长度
 
 : 打字.Optional[int] = None


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


输出\_attentions
 
 : 打字.Optional[bool] = None


输出\_隐藏\_状态
 
 : 打字.Optional[bool] = None


输出\_分数
 
 : 打字.Optional[bool] = None


返回\_dict\_in\_generate
 
 : 打字.Optional[bool] = None


已同步\_gpus
 
 ：布尔=假


\*\*模型\_kwargs




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **光束\_scorer**
 （
 `光束评分器`
 )—
的派生实例
 [BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 定义了梁假设如何构建、存储和
在生成期间排序。有关更多信息，请参阅文档
 [BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 应该阅读。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 用于修改每个生成步骤中应用的语言建模头的预测分数。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
一个实例
 [StoppingCriteriaList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteriaList)
 。派生类的实例列表
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 用于判断生成循环是否应该停止。
* **logits\_warper**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 用过的
扭曲多项式之前应用的语言建模头的预测分数分布
在每个生成步骤进行采样。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 20) —
 **已弃用**
 。使用
 `logits_processor`
 或者
 `停止标准`
 直接限制生成的数量
代币。要生成的序列的最大长度。
* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 在下面
返回张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否继续运行 while 循环直到 max\_length （ZeRO 阶段 3 需要）
模型\_kwargs —
其他特定于模型的 kwargs 将转发给
 ‘前进’
 模型的功能。如果型号是
kwargs 应包含的编码器-解码器模型
 `编码器输出`
 。


 使用语言建模头为模型生成令牌 ID 序列
 **束搜索多项式
采样**
 可用于文本解码器、文本到文本、语音到文本和视觉到文本模型。


大多数情况下，您不需要调用
 [beam\_sample()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_sample)
 直接地。使用生成（）
反而。有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


例子：



```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     TopKLogitsWarper,
...     TemperatureLogitsWarper,
...     BeamSearchScorer,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

>>> # lets run beam search using 3 beams
>>> num_beams = 3
>>> # define decoder start token ids
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> # add encoder\_outputs to model keyword arguments
>>> model_kwargs = {
...     "encoder\_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }

>>> # instantiate beam scorer
>>> beam_scorer = BeamSearchScorer(
...     batch_size=1,
...     max_length=model.config.max_length,
...     num_beams=num_beams,
...     device=model.device,
... )

>>> # instantiate logits processors
>>> logits_processor = LogitsProcessorList(
...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
... )
>>> # instantiate logits processors
>>> logits_warper = LogitsProcessorList(
...     [
...         TopKLogitsWarper(50),
...         TemperatureLogitsWarper(0.7),
...     ]
... )

>>> outputs = model.beam_sample(
...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt bist du?']
```


#### 


对比搜索


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L1909)



 (
 


 输入\_ids
 
 : 长张量


顶部\_k
 
 : 打字.Optional[int] = 1


惩罚\_alpha
 
 : 打字。可选[浮点] = 0


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


逻辑\_warper
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


输出\_attentions
 
 : 打字.Optional[bool] = None


输出\_隐藏\_状态
 
 : 打字.Optional[bool] = None


输出\_分数
 
 : 打字.Optional[bool] = None


返回\_dict\_in\_generate
 
 : 打字.Optional[bool] = None


已同步\_gpus
 
 ：布尔=假


流光
 
 : 打字.Optional[ForwardRef('BaseStreamer')] = None


顺序的
 
 : 打字.Optional[bool] = None


\*\*模型\_kwargs




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **顶部\_k**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
用于重新排序以进行对比搜索的候选集的大小
* **处罚\_alpha**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0) —
对比搜索的退化惩罚；大于0时激活
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 用于修改每个生成步骤中应用的语言建模头的预测分数。
* **logits\_warper**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 用过的
扭曲多项式之前应用的语言建模头的预测分数分布
在每个生成步骤进行采样。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
一个实例
 [StoppingCriteriaList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteriaList)
 。派生类的实例列表
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 用于判断生成循环是否应该停止。
* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 在下面
返回张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否继续运行 while 循环直到 max\_length （ZeRO 阶段 3 需要）
* **流光**
 （
 `BaseStreamer`
 ,
 *选修的*
 )—
Streamer 对象将用于流式传输生成的序列。生成的token被传递
通过
 `streamer.put(token_ids)`
 流媒体负责任何进一步的处理。
* **顺序**
 （
 `布尔`
 ,
 *选修的*
 )—
如果为 True，则将 topk 隐藏状态计算从并行切换为顺序以减少内存。
模型\_kwargs —
其他特定于模型的关键字参数将转发到
 ‘前进’
 模型的功能。
如果模型是编码器-解码器模型，则 kwargs 应包括
 `编码器输出`
 。


 使用语言建模头为模型生成令牌 ID 序列
 **对比搜索**
 并且可以
可用于文本解码器、文本到文本、语音到文本和视觉到文本模型。


大多数情况下，您不需要调用
 [contrastive\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.contrastive_search)
 直接地。使用
改为生成（）。有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


例子：



```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForCausalLM,
...     StoppingCriteriaList,
...     MaxLengthCriteria,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
>>> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> # set pad\_token\_id to eos\_token\_id because OPT does not have a PAD token
>>> model.config.pad_token_id = model.config.eos_token_id
>>> input_prompt = "DeepMind Company is"
>>> input_ids = tokenizer(input_prompt, return_tensors="pt")
>>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
>>> outputs = model.contrastive_search(
...     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
... )
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['DeepMind Company is a company that focuses on the development and commercialization of artificial intelligence (AI). DeepMind’s mission is to help people understand and solve problems that are difficult to solve in the world today.

In this post, we talk about the benefits of deep learning in business and how it']
```


#### 


组\_beam\_search


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L3546)



 (
 


 输入\_ids
 
 : 长张量


光束\_scorer
 
 : 光束记录仪


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


最长长度
 
 : 打字.Optional[int] = None


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


输出\_attentions
 
 : 打字.Optional[bool] = None


输出\_隐藏\_状态
 
 : 打字.Optional[bool] = None


输出\_分数
 
 : 打字.Optional[bool] = None


返回\_dict\_in\_generate
 
 : 打字.Optional[bool] = None


已同步\_gpus
 
 ：布尔=假


\*\*模型\_kwargs




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **光束\_scorer**
 （
 `光束评分器`
 )—
的派生实例
 [BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 定义了梁假设如何构建、存储和
在生成期间排序。有关更多信息，请参阅文档
 [BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 应该阅读。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 用于修改每个生成步骤中应用的语言建模头的预测分数。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
一个实例
 [StoppingCriteriaList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteriaList)
 。派生类的实例列表
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 用于判断生成循环是否应该停止。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 20) —
 **已弃用**
 。使用
 `logits_processor`
 或者
 `停止标准`
 直接限制生成的数量
代币。要生成的序列的最大长度。
* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 在下面
返回张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否继续运行 while 循环直到 max\_length （ZeRO 阶段 3 需要）
 
 模型\_kwargs —
其他特定于模型的 kwargs 将转发给
 ‘前进’
 模型的功能。如果
model 是 kwargs 应该包含的编码器-解码器模型
 `编码器输出`
 。


 使用语言建模头为模型生成令牌 ID 序列
 **多样化波束搜索
解码**
 可用于文本解码器、文本到文本、语音到文本和视觉到文本模型。


大多数情况下，您不需要调用
 [group\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.group_beam_search)
 直接地。使用
改为生成（）。有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


例子：



```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     HammingDiversityLogitsProcessor,
...     BeamSearchScorer,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


>>> # lets run diverse beam search using 6 beams
>>> num_beams = 6
>>> # define decoder start token ids
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> # add encoder\_outputs to model keyword arguments
>>> model_kwargs = {
...     "encoder\_outputs": model.get_encoder()(
...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...     )
... }

>>> # instantiate beam scorer
>>> beam_scorer = BeamSearchScorer(
...     batch_size=1,
...     max_length=model.config.max_length,
...     num_beams=num_beams,
...     device=model.device,
...     num_beam_groups=3,
... )

>>> # instantiate logits processors
>>> logits_processor = LogitsProcessorList(
...     [
...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
...     ]
... )

>>> outputs = model.group_beam_search(
...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt bist du?']
```


#### 


约束\_beam\_search


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L3925)



 (
 


 输入\_ids
 
 : 长张量


约束\_beam\_scorer
 
 ：ConstrainedBeamSearchScorer


logits\_processor
 
 : 打字.可选[transformers. Generation.logits\_process.LogitsProcessorList] = None


停止\_标准
 
 : 打字.可选[transformers. Generation.stopping\_criteria.StoppingCriteriaList] = None


最长长度
 
 : 打字.Optional[int] = None


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


输出\_attentions
 
 : 打字.Optional[bool] = None


输出\_隐藏\_状态
 
 : 打字.Optional[bool] = None


输出\_分数
 
 : 打字.Optional[bool] = None


返回\_dict\_in\_generate
 
 : 打字.Optional[bool] = None


已同步\_gpus
 
 : 打字.Optional[bool] = None


\*\*模型\_kwargs




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **约束\_beam\_scorer**
 （
 `ConstrainedBeamSearchScorer`
 )—
的派生实例
 [BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 定义了梁假设如何构建、存储和
在生成过程中进行排序，同时满足积极约束列表。欲了解更多信息，请
的文档
 [ConstrainedBeamSearchScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.ConstrainedBeamSearchScorer)
 应该阅读。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 用于修改每个生成步骤中应用的语言建模头的预测分数。
* **停止\_criteria**
 （
 `停止标准列表`
 ,
 *选修的*
 )—
一个实例
 [StoppingCriteriaList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteriaList)
 。派生类的实例列表
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 用于判断生成循环是否应该停止。
* **logits\_warper**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
一个实例
 [LogitsProcessorList](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessorList)
 。派生类的实例列表
 [LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 用过的
扭曲多项式之前应用的语言建模头的预测分数分布
在每个生成步骤进行采样。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 ，默认为 20) —
 **已弃用**
 。使用
 `logits_processor`
 或者
 `停止标准`
 直接限制生成的数量
代币。要生成的序列的最大长度。
* **pad\_token\_id**
 （
 `int`
 ,
 *选修的*
 )—
的 ID
 *填充*
 令牌。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 ,
 *选修的*
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输出\_注意**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有注意力层的注意力张量。看
 `注意`
 在下面
返回张量以获取更多详细信息。
* **输出\_隐藏\_状态**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回所有层的隐藏状态。看
 `隐藏状态`
 在返回张量下
更多细节。
* **输出\_分数**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否返回预测分数。看
 `分数`
 在返回的张量下了解更多详细信息。
* **返回\_dict\_in\_generate**
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
* **同步\_gpus**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否继续运行 while 循环直到 max\_length （ZeRO 阶段 3 需要）
模型\_kwargs —
其他特定于模型的 kwargs 将转发给
 ‘前进’
 模型的功能。如果型号是
kwargs 应包含的编码器-解码器模型
 `编码器输出`
 。


 使用语言建模头为模型生成令牌 ID 序列
 **约束波束搜索
解码**
 可用于文本解码器、文本到文本、语音到文本和视觉到文本模型。


大多数情况下，您不需要调用
 [constrained\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.constrained_beam_search)
 直接地。使用
改为生成（）。有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


例子：



```
>>> from transformers import (
...     AutoTokenizer,
...     AutoModelForSeq2SeqLM,
...     LogitsProcessorList,
...     MinLengthLogitsProcessor,
...     ConstrainedBeamSearchScorer,
...     PhrasalConstraint,
... )
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> encoder_input_str = "translate English to German: How old are you?"
>>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


>>> # 让我们使用 3 个波束运行波束搜索
>>> num_beams = 3
>>> # 定义解码器启动令牌 id
>>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>>> input_ids = input_ids * model.config.decoder_start_token_id

>>> # 将编码器\_outputs 添加到模型关键字参数
>>> model_kwargs = {
...“编码器\_输出”：model.get_encoder()(
...encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
...）
... }

>>>约束_str =“谢”
>>>constraint_token_ids = tokenizer.encode(constraint_str)[:-1] # 用于删除 eos 令牌的切片
>>> 约束 = [PhrasalConstraint(token_ids=constraint_token_ids)]


>>> # instantiate beam scorer
>>> beam_scorer = ConstrainedBeamSearchScorer(
...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
... )

>>> # instantiate logits processors
>>> logits_processor = LogitsProcessorList(
...     [
...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
...     ]
... )

>>> outputs = model.constrained_beam_search(
...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
... )

>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Wie alt sind Sie?']
```


## TFGenerationMixin




### 


班级
 

 变压器。
 

 TFGenerationMixin


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L444)


（
 

 ）


包含所有支持生成的函数的类，可用作 mixin
 [TFPreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.TFPreTrainedModel)
 。


班级暴露
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.TFGenerationMixin.generate）
 ，可用于：


* *贪心解码*
 通过致电
 `贪婪搜索()`
 如果
 `num_beams=1`
 和
 `do_sample=False`
* *对比搜索*
 通过致电
 `contrastive_search()`
 如果
 `penalty_alpha>0`
 和
 `top_k>1`
* *多项抽样*
 通过致电
 `样本()`
 如果
 `num_beams=1`
 和
 `do_sample=True`
* *波束搜索解码*
 通过致电
 `beam_search()`
 如果
 `num_beams>1`


您不需要直接调用上述任何方法。将自定义参数值传递给“generate”。到
了解有关解码策略的更多信息，请参阅
 [文本生成策略指南](../Generation_strategies)
 。



#### 


产生


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L645)



 (
 


 输入
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


生成\_config
 
 : 打字.可选[transformers. Generation.configuration\_utils.GenerationConfig] = None


logits\_processor
 
 : 打字.可选[transformers. Generation.tf\_logits\_process.TFLogitsProcessorList] = None


种子
 
 = 无


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

[模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 或者
 `tf.张量`


 参数


* **输入**
 （
 `tf.张量`
 根据模态的不同形状也不同，
 *选修的*
 )—
用作生成提示或编码器的模型输入的序列。如果
 `无`
 这
方法将其初始化为
 `bos_token_id`
 批量大小为 1。对于仅解码器模型
 `输入`
 应该采用以下格式
 `输入ID`
 。对于编码器-解码器模型
 *输入*
 可以代表任何一个
 `输入ID`
 ,
 `输入值`
 ,
 `输入特征`
 ， 或者
 `像素值`
 。
* **生成\_config**
 （
 `~ Generation.GenerationConfig`
 ,
 *选修的*
 )—
用作生成调用的基本参数化的生成配置。
 `**夸格斯`
 传递生成匹配的属性
 `一代配置`
 将覆盖它们。如果
 `一代配置`
 未提供，将使用默认值，其中有以下加载
优先级：1）从
 ` Generation_config.json`
 模型文件（如果存在）； 2）从模型
配置。请注意，未指定的参数将继承
 [GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)
 的
默认值，应检查其文档以参数化生成。
* **logits\_processor**
 （
 `LogitsProcessorList`
 ,
 *选修的*
 )—
自定义 logits 处理器，补充由参数构建的默认 logits 处理器
生成配置。如果传递的 logit 处理器已经使用参数或
生成配置抛出错误。此功能适用于高级用户。
* **种子**
 （
 `列表[整数]`
 ,
 *选修的*
 )—
控制采样的随机种子，包含两个整数，使用时
 `做样本`
 是
 '真实'
 。请参阅
 `种子`
 来自无状态函数的参数
 `tf.随机`
 。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
临时参数化
 `生成配置`
 和/或其他特定于模型的 kwargs
转发至
 ‘前进’
 模型的功能。如果模型是编码器-解码器模型，则编码器
特定的 kwargs 不应加前缀，解码器特定的 kwargs 应加前缀
 *解码器\_*
 。


退货


导出常量元数据='未定义';
 

[模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 或者
 `tf.张量`


导出常量元数据='未定义';


A
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 （如果
 `return_dict_in_generate=True`
 或当
 `config.return_dict_in_generate=True`
 ）或一个
 `tf.张量`
 。


如果模型是
 *不是*
 编码器-解码器模型（
 `model.config.is_encoder_decoder=False`
 ），可能的
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 类型有：


* [TFGreedySearchDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFGreedySearchDecoderOnlyOutput)
 ,
* [TFSampleDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFSampleDecoderOnlyOutput)
 ,
* [TFBeamSearchDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFBeamSearchDecoderOnlyOutput)
 ,
* [TFBeamSampleDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFBeamSampleDecoderOnlyOutput)


如果模型是编码器-解码器模型（
 `model.config.is_encoder_decoder=True`
 ），可能的
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 类型有：


* [TFGreedySearchEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFGreedySearchEncoderDecoderOutput)
 ,
* [TFSampleEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFSampleEncoderDecoderOutput)
 ,
* [TFBeamSearchEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFBeamSearchEncoderDecoderOutput)
 ,
* [TFBeamSampleEncoderDecoderOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.TFBeamSampleEncoderDecoderOutput)


为具有语言建模头的模型生成令牌 ID 序列。


大多数发电控制参数设置在
 `一代配置`
 如果没有通过，将被设置为
模型的默认生成配置。您可以覆盖任何
 `一代配置`
 通过传递相应的
要生成的参数，例如
 `.generate(输入，num_beams=4, do_sample=True)`
 。


有关生成策略和代码示例的概述，请查看
 [下列的
指南](../Generation_strategies)
 。


#### 


计算\_transition\_scores


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L477)



 (
 


 序列
 
 ：张量


分数
 
 ：打字.元组[tensorflow.python.framework.ops.Tensor]


梁\_索引
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


标准化\_logits
 
 ：布尔=假


）
 

 →


导出常量元数据='未定义';
 

`tf.张量`


 参数


* **序列**
 （
 `tf.张量`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或者
如果所有批次提前完成，则时间会更短
 `eos_token_id`
 。
* **分数**
 （
 `元组(tf.张量)`
 )—
每个生成步骤中每个词汇标记的转换分数。梁跃迁分数包括
以先前生成的标记的 log softmax 为条件的标记的对数概率的元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个生成的标记一个元素），每个元素
形状张量
 `(batch_size*num_beams, config.vocab_size)`
 。
* **梁\_索引**
 （
 `tf.张量`
 ,
 *选修的*
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。仅当出现以下情况时才需要
 `num_beams>1`
 在
生成时间。
* **标准化\_logits**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否标准化 logits（由于遗留原因，可能未标准化）。


退货


导出常量元数据='未定义';
 

`tf.张量`


导出常量元数据='未定义';


A
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 含有
转换分数 (logits)


计算给定生成分数（和波束索引，如果波束搜索是
用过的）。这是一种在生成时快速获取所选令牌分数的便捷方法。


 例子：



```
>>> from transformers import GPT2Tokenizer, TFAutoModelForCausalLM
>>> import numpy as np

>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> model = TFAutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer.pad_token_id = tokenizer.eos_token_id
>>> inputs = tokenizer(["Today is"], return_tensors="tf")

>>> # Example 1: Print the scores for each token generated with Greedy Search
>>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, normalize_logits=True
... )
>>> # input\_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
>>> # encoder-decoder models, like BART or T5.
>>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
>>> generated_tokens = outputs.sequences[:, input_length:]
>>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
...     # | token | token string | logits | probability
...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
|   262 |  the     | -1.413 | 24.33%
|  1110 |  day     | -2.609 | 7.36%
|   618 |  when    | -2.009 | 13.41%
|   356 |  we      | -1.859 | 15.58%
|   460 |  can     | -2.508 | 8.14%

>>> # Example 2: Reconstruct the sequence scores from Beam Search
>>> outputs = model.generate(
...     **inputs,
...     max_new_tokens=5,
...     num_beams=4,
...     num_return_sequences=4,
...     return_dict_in_generate=True,
...     output_scores=True,
... )
>>> transition_scores = model.compute_transition_scores(
...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
... )
>>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
>>> # Tip: recomputing the scores is only guaranteed to match with `normalize\_logits=False`. Depending on the
>>> # use case, you might want to recompute it with `normalize\_logits=True`.
>>> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
>>> length_penalty = model.generation_config.length_penalty
>>> reconstructed_scores = np.sum(transition_scores, axis=1) / (output_length**length_penalty)
>>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
True
```


## FlaxGenerationMixin




### 


班级
 

 变压器。
 

 FlaxGenerationMixin


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_utils.py#L129)


（
 

 ）


包含自动回归文本生成的所有函数的类，用作 mixin
 [FlaxPreTrainedModel](/docs/transformers/v4.35.2/en/main_classes/model#transformers.FlaxPreTrainedModel)
 。


班级暴露
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.FlaxGenerationMixin.generate）
 ，可用于：


* *贪心解码*
 通过致电
 `_greedy_search()`
 如果
 `num_beams=1`
 和
 `do_sample=False`
* *多项抽样*
 通过致电
 `_sample()`
 如果
 `num_beams=1`
 和
 `do_sample=True`
* *波束搜索解码*
 通过致电
 `_beam_search()`
 如果
 `num_beams>1`
 和
 `do_sample=False`


您不需要直接调用上述任何方法。将自定义参数值传递给“generate”。到
了解有关解码策略的更多信息，请参阅
 [文本生成策略指南](../Generation_strategies)
 。



#### 


产生


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_utils.py#L267)



 (
 


 输入\_ids
 
 ： 大批


生成\_config
 
 : 打字.可选[transformers. Generation.configuration\_utils.GenerationConfig] = None


prng\_key
 
 : 打字.可选[jax.Array] =无


痕迹
 
 ：布尔=真


参数
 
 : Typing.Union[typing.Dict[str, jax.Array], NoneType] = None


logits\_processor
 
 : 打字.可选[transformers. Generation.flax\_logits\_process.FlaxLogitsProcessorList] = None


\*\*夸格




 )
 


 参数


* **输入\_ids**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，序列长度）`
 )—
用作生成提示的序列。
* **生成\_config**
 （
 `~ Generation.GenerationConfig`
 ,
 *选修的*
 )—
用作生成调用的基本参数化的生成配置。
 `**夸格斯`
 传递生成匹配的属性
 `一代配置`
 将覆盖它们。如果
 `一代配置`
 未提供，将使用默认值，其中有以下加载
优先级：1）从
 ` Generation_config.json`
 模型文件（如果存在）； 2）从模型
配置。请注意，未指定的参数将继承
 [GenerationConfig](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationConfig)
 的
默认值，应检查其文档以参数化生成。
* **痕迹**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
是否追溯生成。环境
 `跟踪=假`
 应该只用于调试，并且会导致
运行时间相当慢。
* **参数**
 （
 `字典[str, jnp.ndarray]`
 ,
 *选修的*
 )—
可以选择传递模型参数。对于并行生成很有用。
* **logits\_processor**
 （
 `FlaxLogitsProcessorList`
 ,
 *选修的*
 )—
自定义 logits 处理器，补充由参数构建的默认 logits 处理器
生成配置。如果传递的 logit 处理器已经使用参数或
生成配置抛出错误。此功能适用于高级用户。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
临时参数化
 `生成配置`
 和/或其他特定于模型的 kwargs
转发至
 ‘前进’
 模型的功能。如果模型是编码器-解码器模型，则编码器
特定的 kwargs 不应加前缀，解码器特定的 kwargs 应加前缀
 *解码器\_*
 。


 为具有语言建模头的模型生成令牌 ID 序列。