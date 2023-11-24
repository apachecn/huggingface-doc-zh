# 发电公用事业

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/internal/generation_utils>
>
> 原始地址：<https://huggingface.co/docs/transformers/internal/generation_utils>


此页面列出了使用的所有实用函数
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.generate）
 ,
 [greedy\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.greedy_search)
 ,
 [contrastive\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.contrastive_search)
 ,
 [样本（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.sample）
 ,
 [beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_search)
 ,
 [beam\_sample()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_sample)
 ,
 [group\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.group_beam_search)
 ， 和
 [constrained\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.constrained_beam_search)
 。


其中大多数仅在您研究库中生成方法的代码时才有用。


## 生成输出



的输出
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.generate）
 是一个子类的实例
 [模型输出](/docs/transformers/v4.35.2/en/main_classes/output#transformers.utils.ModelOutput)
 。该输出是一个包含所有返回信息的数据结构
经过
 [生成（）]（/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.generate）
 ，但也可以用作元组或字典。


这是一个例子：



```
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```


这
 `一代输出`
 对象是一个
 [GreedySearchDecoderOnlyOutput](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers. Generation.GreedySearchDecoderOnlyOutput)
 ，我们可以
请参阅下面该类的文档，这意味着它具有以下属性：


* `序列`
 ：生成的标记序列
*`分数`
 （可选）：每个生成步骤的语言建模头的预测分数
* `隐藏状态`
 （可选）：每个生成步骤的模型的隐藏状态
*`注意`
 （可选）：模型的每个生成步骤的注意力权重


在这里我们有
 `分数`
 自从我们过去
 `output_scores=True`
 ，但我们没有
 `隐藏状态`
 和
 `注意`
 因为我们没有通过
 `output_hidden_​​states=True`
 或者
 `output_attentions=True`
 。


您可以像平常一样访问每个属性，如果模型尚未返回该属性，您可以
将会得到
 `无`
 。这里例如
 ` Generation_output.scores`
 是所有生成的预测分数
语言建模头，以及
 ` Generation_output.attentions`
 是
 `无`
 。


当使用我们的
 `一代输出`
 对象作为元组，它只保留没有的属性
 `无`
 价值观。
例如，这里有两个元素，
 `损失`
 然后
 `逻辑`
 ， 所以



```
generation_output[:2]
```


将返回元组
 `( Generation_output.sequences, Generation_output.scores)`
 例如。


当使用我们的
 `一代输出`
 对象作为字典，它只保留没有的属性
 `无`
 价值观。例如，这里有两个键：
 `序列`
 和
 `分数`
 。


我们在这里记录所有输出类型。


### 


 火炬



### 


班级
 

 变形金刚.一代.
 

 贪婪搜索编码器解码器输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L184)


（


 序列
 
 : 长张量 = 无


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（一个元素用于
每个生成的标记），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **编码器\_注意**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `torch.FloatTensor`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `torch.FloatTensor`
 （一个用于嵌入的输出+一个用于每一层的输出）
形状
 `（批量大小，序列长度，隐藏大小）`
 。
* **解码器\_attentions**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用贪婪搜索的编码器-解码器生成模型的输出的基类。隐藏状态和注意力
解码器（分别是编码器）的权重可以通过编码器\_attentions和
编码器\_hidden\_states属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 贪婪搜索解码器仅输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L88)


（


 序列
 
 : 长张量 = 无


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


注意事项
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（一个元素用于
每个生成的标记），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **注意**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用贪婪搜索的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 样本编码器解码器输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L255)


（


 序列
 
 : 长张量 = 无


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（一个元素用于
每个生成的标记），每个张量的形状
 `(batch_size*num_return_sequences, config.vocab_size)`
 。
* **编码器\_注意**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `torch.FloatTensor`
 （解码器的每一层都有一个）形状
 `（batch_size * num_return_sequences，num_heads，sequence_length，sequence_length）`
 。
* **编码器\_隐藏\_状态**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `torch.FloatTensor`
 （一个用于嵌入的输出+一个用于每一层的输出）
形状
 `（batch_size * num_return_sequences，sequence_length，hidden_​​size）`
 。
* **解码器\_attentions**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size * num_return_sequences，num_heads，generate_length，sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `(batch_size*num_return_sequences, generated_length,hidden_​​size)`
 。


 使用采样的编码器-解码器生成模型的输出的基类。的隐藏状态和注意力权重
解码器（分别是编码器）可以通过编码器\_attentions和编码器\_hidden\_states访问
属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 仅采样解码器输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L226)


（


 序列
 
 : 长张量 = 无


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


注意事项
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（一个元素用于
每个生成的标记），每个张量的形状
 `(batch_size*num_return_sequences, config.vocab_size)`
 。
* **注意**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `(num_return_sequences*batch_size, num_heads, generated_length, sequence_length)`
 。
* **隐藏\_状态**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `(num_return_sequences*batch_size, generated_length, hide_size)`
 。


 使用采样的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 BeamSearchEncoderDecoderOutput


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L333)


（


 序列
 
 : 长张量 = 无


序列\_scores
 
 : 打字.Optional[torch.FloatTensor] = None


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None


编码器\_注意力
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size*num_return_sequences)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的波束转换分数。梁跃迁分数包括
令牌的对数概率以该波束中先前生成的令牌的 log softmax 为条件。
元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），
每个形状张量
 `(batch_size*num_beams, config.vocab_size)`
 。
* **梁\_索引**
 （
 `torch.LongTensor`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **编码器\_注意**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `torch.FloatTensor`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `torch.FloatTensor`
 （一个用于嵌入的输出+一个用于每一层的输出）
形状
 `（batch_size * num_beams * num_return_sequences，sequence_length，hidden_​​size）`
 。
* **解码器\_attentions**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size * num_beams * num_return_sequences，num_heads，生成的长度，序列长度）`
 。
* **交叉\_注意力**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size * num_beams * num_return_sequences，生成的长度，隐藏的大小）`
 。


 使用波束搜索的编码器-解码器生成模型的输出的基类。隐藏状态和注意力权重
解码器（分别是编码器）的可以通过编码器\_attentions和编码器\_hidden\_states访问
属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 BeamSearchDecoderOnly 输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L298)


（


 序列
 
 : 长张量 = 无


序列\_scores
 
 : 打字.Optional[torch.FloatTensor] = None


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None


注意事项
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size*num_return_sequences)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的波束转换分数。梁跃迁分数包括
令牌的对数概率以该波束中先前生成的令牌的 log softmax 为条件。
元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），
每个形状张量
 `(batch_size*num_beams*num_return_sequences, config.vocab_size)`
 。
* **梁\_索引**
 （
 `torch.LongTensor`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **注意**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size * num_beams，num_heads，generate_length，sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size * num_beams * num_return_sequences，生成的长度，隐藏的大小）`
 。


 使用波束搜索的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 BeamSampleEncoderDecoderOutput


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L418)


（


 序列
 
 : 长张量 = 无


序列\_scores
 
 : 打字.Optional[torch.FloatTensor] = None


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None


编码器\_注意力
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_beams，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_return_sequence)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的波束转换分数。梁跃迁分数包括
令牌的对数概率以该波束中先前生成的令牌的 log softmax 为条件。
元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），
每个形状张量
 `(batch_size*num_beams, config.vocab_size)`
 ）。
* **梁\_索引**
 （
 `torch.LongTensor`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **编码器\_注意**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `torch.FloatTensor`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `torch.FloatTensor`
 （一个用于嵌入的输出+一个用于每一层的输出）
形状
 `（batch_size * num_beams，sequence_length，hidden_​​size）`
 。
* **解码器\_attentions**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size * num_beams，num_heads，generate_length，sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `(batch_size*num_beams, generated_length, hide_size)`
 。


 使用波束采样的编码器-解码器生成模型的输出的基类。隐藏状态和注意力
解码器（分别是编码器）的权重可以通过编码器\_attentions和
编码器\_hidden\_states属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 BeamSampleDecoderOnly 输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L383)


（


 序列
 
 : 长张量 = 无


序列\_scores
 
 : 打字.Optional[torch.FloatTensor] = None


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None


注意事项
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_return_sequence)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的波束转换分数。梁跃迁分数包括
令牌的对数概率以该波束中先前生成的令牌的 log softmax 为条件。
元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），
每个形状张量
 `(batch_size*num_beams*num_return_sequences, config.vocab_size)`
 。
* **梁\_索引**
 （
 `torch.LongTensor`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **注意**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size * num_beams，num_heads，generate_length，sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `(batch_size*num_beams, generated_length, hide_size)`
 。


 使用波束样本的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 对比搜索编码器解码器输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L116)


（


 序列
 
 : 长张量 = 无


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（一个元素用于
每个生成的标记），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **编码器\_注意**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `torch.FloatTensor`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组（火炬.FloatTensor）`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `torch.FloatTensor`
 （一个用于嵌入的输出+一个用于每一层的输出）
形状
 `（批量大小，序列长度，隐藏大小）`
 。
* **解码器\_attentions**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用对比搜索的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 对比搜索解码器仅输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L155)


（


 序列
 
 : 长张量 = 无


分数
 
 : 打字.可选[打字.元组[火炬.FloatTensor]] =无


注意事项
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.元组[打字.元组[火炬.FloatTensor]]] =无


）


 参数


* **序列**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组（火炬.FloatTensor）`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当——
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `torch.FloatTensor`
 最多
 `max_new_tokens`
 元素（一个元素用于
每个生成的标记），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **注意**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（火炬.FloatTensor））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 是 -
* **通过**
 或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `torch.FloatTensor`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用对比搜索的仅解码器生成模型的输出的基类。



### 


 TensorFlow



### 


班级
 

 变形金刚.一代.
 

 TFGreedySearchEncoderDecoderOutput


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L86)


（


 序列
 
 : 张量 = 无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


）


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个元素一个元素
生成的令牌），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **编码器\_注意**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `tf.张量`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `tf.张量`
 （一个用于嵌入的输出+一个用于每一层的输出）形状
 `（批量大小，序列长度，隐藏大小）`
 。
* **解码器\_attentions**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用贪婪搜索的编码器-解码器生成模型的输出的基类。隐藏状态和注意力
解码器（分别是编码器）的权重可以通过编码器\_attentions和
编码器\_hidden\_states属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 TFGreedySearchDecoderOnlyOutput TFGreedySearchDecoderOnlyOutput


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L58)


（


 序列
 
 : 张量 = 无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


注意事项
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


）


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个元素一个元素
生成的令牌），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **注意**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用贪婪搜索的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 TFSampleEncoderDecoder输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L156)



 (
 


 序列
 
 : 张量 = 无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个元素一个元素
生成的令牌），每个张量的形状
 `(batch_size*num_return_sequences, config.vocab_size)`
 。
* **编码器\_注意**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `tf.张量`
 （解码器的每一层都有一个）形状
 `（batch_size * num_return_sequences，num_heads，sequence_length，sequence_length）`
 。
* **编码器\_隐藏\_状态**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `tf.张量`
 （一个用于嵌入的输出+一个用于每一层的输出）形状
 `（batch_size * num_return_sequences，sequence_length，hidden_​​size）`
 。
* **解码器\_attentions**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，num_heads，generate_length，sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `(batch_size*num_return_sequences, generated_length,hidden_​​size)`
 。


 使用采样的编码器-解码器生成模型的输出的基类。的隐藏状态和注意力权重
解码器（分别是编码器）可以通过编码器\_attentions和编码器\_hidden\_states访问
属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 TFSampleDecoderOnly 输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L128)



 (
 


 序列
 
 : 张量 = 无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


注意事项
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个元素一个元素
生成的令牌），每个张量的形状
 `(batch_size*num_return_sequences, config.vocab_size)`
 。
* **注意**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `(num_return_sequences*batch_size, num_heads, generated_length, sequence_length)`
 。
* **隐藏\_状态**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `(num_return_sequences*batch_size, generated_length, hide_size)`
 。


 使用采样的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 TBeamSearchEncoderDecoderOutput


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L233)



 (
 


 序列
 
 : 张量 = 无


序列\_scores
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


梁\_索引
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `tf.张量`
 形状的
 `(batch_size*num_return_sequences)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的处理光束分数。梁分数由日志组成
每个词汇标记的 softmax 分数以及之前生成的标记的 log softmax 之和
光束。
 `元组`
 tf张量
 `最多`
 最大\_新\_令牌
 `elements（每个生成的 token 一个元素），每个张量的形状`
 (batch\_size\*num\_beams, config.vocab\_size)`。
* **梁\_索引**
 （
 `tf.张量`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **编码器\_注意**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `tf.张量`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `tf.张量`
 （一个用于嵌入的输出+一个用于每一层的输出）形状
 `（batch_size * num_beams * num_return_sequences，sequence_length，hidden_​​size）`
 。
* **解码器\_attentions**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size * num_beams * num_return_sequences，num_heads，生成的长度，序列长度）`
 。
* **交叉\_注意力**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size * num_beams * num_return_sequences，生成的长度，隐藏的大小）`
 。


 使用波束搜索的编码器-解码器生成模型的输出的基类。隐藏状态和注意力权重
解码器（分别是编码器）的可以通过编码器\_attentions和编码器\_hidden\_states访问
属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 TBeamSearchDecoderOnly 输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L198)



 (
 


 序列
 
 : 张量 = 无


序列\_scores
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


梁\_索引
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


注意事项
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `tf.张量`
 形状的
 `(batch_size*num_return_sequences)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的处理光束分数。梁分数由日志组成
每个词汇标记的 softmax 分数以及之前生成的标记的 log softmax 之和
光束。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），
每个形状张量
 `(batch_size*num_beams*num_return_sequences, config.vocab_size)`
 。
* **梁\_索引**
 （
 `tf.张量`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **注意**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size * num_beams，num_heads，generate_length，sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size * num_beams * num_return_sequences，生成的长度，隐藏的大小）`
 。


 使用波束搜索的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 TBeamSampleEncoderDecoderOutput


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L318)



 (
 


 序列
 
 : 张量 = 无


序列\_scores
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


梁\_索引
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（batch_size * num_beams，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `tf.张量`
 形状的
 `(batch_size * num_return_sequence)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的处理光束分数。梁分数由日志组成
每个词汇标记的 softmax 分数以及之前生成的标记的 log softmax 之和
光束。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），
每个形状张量
 `(batch_size*num_beams, config.vocab_size)`
 。
* **梁\_索引**
 （
 `tf.张量`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **编码器\_注意**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `tf.张量`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `tf.张量`
 （一个用于嵌入的输出+一个用于每一层的输出）形状
 `（batch_size * num_beams，sequence_length，hidden_​​size）`
 。
* **解码器\_注意**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size * num_beams，num_heads，generate_length，sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `(batch_size*num_beams, generated_length, hide_size)`
 。


 使用波束采样的编码器-解码器生成模型的输出的基类。隐藏状态和注意力
解码器（分别是编码器）的权重可以通过编码器\_attentions和
编码器\_hidden\_states属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 TBeamSampleDecoderOnly 输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L283)



 (
 


 序列
 
 : 张量 = 无


序列\_scores
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


梁\_索引
 
 ：打字。可选[tensorflow.python.framework.ops.Tensor] =无


注意事项
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **序列\_分数**
 （
 `tf.张量`
 形状的
 `(batch_size * num_return_sequence)`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
生成的最终光束分数
 `序列`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤中每个词汇标记的处理光束分数。梁分数由日志组成
每个词汇标记的 softmax 分数以及之前生成的标记的 log softmax 之和
光束。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个生成的令牌一个元素），
每个形状张量
 `(batch_size*num_beams*num_return_sequences, config.vocab_size)`
 。
* **梁\_索引**
 （
 `tf.张量`
 ,
 *选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
每个生成步骤生成的令牌 ID 的波束索引。
 `tf.张量`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`
 。
* **注意**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size * num_beams，num_heads，generate_length，sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `(batch_size*num_beams, generated_length, hide_size)`
 。


 使用波束样本的仅解码器生成模型的输出的基类。




### 


班级
 

 变形金刚.一代.
 

 TFContrastiveSearchEncoderDecoderOutput


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L394)



 (
 


 序列
 
 : 张量 = 无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器\_注意力
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


编码器隐藏状态
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


解码器\_attentions
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


交叉\_注意力
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


解码器\_hidden\_states
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个元素一个元素
生成的令牌），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **编码器\_注意**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组
 `tf.张量`
 （解码器的每一层都有一个）形状
 `(batch_size、num_heads、sequence_length、sequence_length)`
 。
* **编码器\_隐藏\_状态**
 （
 `元组(tf.张量)`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组
 `tf.张量`
 （一个用于嵌入的输出+一个用于每一层的输出）形状
 `（批量大小，序列长度，隐藏大小）`
 。
* **解码器\_注意**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **交叉\_注意力**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **解码器\_hidden\_states**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用对比搜索的编码器-解码器生成模型输出的基类。隐藏状态和注意力
解码器（分别是编码器）的权重可以通过编码器\_attentions和
编码器\_hidden\_states属性（分别是解码器\_attentions和解码器\_hidden\_states属性）




### 


班级
 

 变形金刚.一代.
 

 TFContrastiveSearchDecoderOnly 输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L367)



 (
 


 序列
 
 : 张量 = 无


分数
 
 : 打字.可选[打字.元组[tensorflow.python.framework.ops.Tensor]] =无


注意事项
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无


隐藏\_状态
 
 : 打字.可选[打字.Tuple[打字.Tuple[tensorflow.python.framework.ops.Tensor]]] =无



 )
 


 参数


* **序列**
 （
 `tf.张量`
 形状的
 `（批量大小，序列长度）`
 )—
生成的序列。第二个维度（sequence\_length）等于
 `最大长度`
 或更短
如果所有批次由于以下原因提前完成
 `eos_token_id`
 。
* **分数**
 （
 `元组(tf.张量)`
*选修的*
 ，返回时
 `output_scores=True`
 通过或当
 `config.output_scores=True`
 )—
语言建模头处理后的预测分数（SoftMax 之前每个词汇标记的分数）
在每个世代步骤。元组
 `tf.张量`
 最多
 `max_new_tokens`
 元素（每个元素一个元素
生成的令牌），每个张量的形状
 `(batch_size, config.vocab_size)`
 。
* **注意**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_attentions=True`
 已通过或
 `config.output_attentions=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（batch_size、num_heads、 generated_length、sequence_length）`
 。
* **隐藏\_状态**
 （
 `元组（元组（tf.张量））`
 ,
 *选修的*
 ，返回时
 `output_hidden_​​states=True`
 通过或当
 `config.output_hidden_​​states=True`
 )—
元组（解码器每一层一个元素）的元组（每个生成的标记一个元素）
 `tf.张量`
 形状的
 `（批量大小，生成长度，隐藏大小）`
 。


 使用对比搜索的仅解码器生成模型的输出的基类。



### 


 亚麻



### 


班级
 

 变形金刚.一代.
 

 Flax样本输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_utils.py#L69)



 (
 


 序列
 
 ：数组=无



 )
 


 参数


* **序列**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，最大长度）`
 )—
生成的序列。


 Flax 基类，用于使用采样的仅解码器生成模型的输出。



#### 


代替


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/flax/struct.py#L111)



 (
 


 \*\*更新




 )
 


“返回一个新对象，用新值替换指定字段。


### 


班级
 

 变形金刚.一代.
 

 Flax贪婪搜索输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_utils.py#L55)



 (
 


 序列
 
 ：数组=无



 )
 


 参数


* **序列**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，最大长度）`
 )—
生成的序列。


 Flax 基类，用于使用贪婪搜索的仅解码器生成模型的输出。



#### 


代替


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/flax/struct.py#L111)



 (
 


 \*\*更新




 )
 


“返回一个新对象，用新值替换指定字段。


### 


班级
 

 变形金刚.一代.
 

 FlaxBeam搜索输出


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_utils.py#L83)



 (
 


 序列
 
 ：数组=无


分数
 
 ：数组=无



 )
 


 参数


* **序列**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，最大长度）`
 )—
生成的序列。
* **分数**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，）`
 )—
生成序列的分数（对数概率）。


 Flax 基类，用于使用贪婪搜索的仅解码器生成模型的输出。



#### 


代替


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/flax/struct.py#L111)



 (
 


 \*\*更新




 )
 


“返回一个新对象，用新值替换指定字段。


## 逻辑处理器



A
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 可用于修改语言模型头的预测分数
一代。


### 


 火炬



### 


class
 

 transformers.
 

 AlternatingCodebooksLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1600)



 (
 


 输入\_start\_len
 
 ：整数


语义\_词汇\_大小
 
 ：整数


码本\_size
 
 ：整数



 )
 


 参数


* **输入\_start\_len**
 （
 `int`
 )—
初始输入序列的长度。
* **语义\_vocab\_size**
 （
 `int`
 )—
语义部分的词汇大小，即与语义词汇相关的标记数量。
* **密码本\_size**
 （
 `int`
 )—
与码本关联的令牌数量。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制两个码本之间交替生成
 `树皮`
 的精细子模型。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1621)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量



 )
 



### 


class
 

 transformers.
 

 ClassifierFreeGuidanceLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1560)



 (
 


 指导\_规模




 )
 


 参数


* **指导\_scale**
 （漂浮） -
无分类器引导（CFG）的引导量表。 CFG 通过设置启用
 `guidance_scale > 1`
 。
更高的指导尺度鼓励模型生成与输入联系更紧密的样本
及时，通常以质量较差为代价。


 用于分类器免费指导 (CFG) 的 Logits 处理器。分数按批次维度划分，
其中前半部分对应于条件 logits（根据输入提示预测），后半部分对应于条件 logits
对应于无条件 logits（根据空或“null”提示预测）。处理器计算一个
条件和无条件 logits 的加权平均值，参数化为
 `指导规模`
 。


看
 [论文](https://arxiv.org/abs/2306.05284)
 了解更多信息。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1584)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


class
 

 transformers.
 

 EncoderNoRepeatNGramLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L775)



 (
 


 编码器\_ngram\_size
 
 ：整数


编码器\_输入\_ids
 
 : 长张量



 )
 


 参数


* **编码器\_ngram\_size**
 （
 `int`
 )—
所有 ngram 大小
 `ngram_size`
 只能出现在编码器输入 id 内。
* **编码器\_input\_ids**
 （
 `int`
 )—
编码器输入 id 不应在解码器 id 中重复。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制解码器 id 的编码器输入 id n-gram 不重复。看
 [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350)
 。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L798)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


class
 

 transformers.
 

 EncoderRepetitionPenaltyLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L328)



 (
 


 惩罚
 
 ： 漂浮


编码器\_输入\_ids
 
 : 长张量



 )
 


 参数


* **惩罚**
 （
 `浮动`
 )—
幻觉惩罚的参数。 1.0 表示没有处罚。高于 1.0 会导致幻觉。之间
0.0 和 1.0 奖励幻觉。
* **编码器\_input\_ids**
 （
 `torch.LongTensor`
 )—
应在解码器 ID 内重复的编码器输入 ID。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 通过提高在原始内容中找到标记的概率来避免幻觉
输入。


这种技术也可以用来奖励，从而以类似的方式鼓励幻觉（或创造力）。到
惩罚和减少幻觉，使用
 `惩罚`
 值高于 1.0，值越高，惩罚越重。到
奖励和鼓励幻觉，使用
 `惩罚`
 值介于 0.0 和 1.0 之间，值越低奖励越多
强烈。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L353)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 EpsilonLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L505)



 (
 


 厄普西隆
 
 ： 漂浮


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1



 )
 


 参数


* **厄普西隆**
 （
 `浮动`
 )—
如果设置为 > 0，则仅包含概率最多的标记
 `epsilon`
 或更高版本保留一代。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 执行 epsilon 采样，即限制为以下标记
 `prob >= epsilon`
 。采取
如果没有令牌满足此约束，则最大的 min\_tokens\_to\_keep 令牌。看
 [截断采样作为语言模型
去平滑](https://arxiv.org/abs/2210.15191)
 了解更多信息。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

>>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
>>> # Top P sampling, which restricts tokens based on their cumulative probability.
>>> # Pro tip: The paper recomends using `epsilon\_cutoff` values between 3e-4 and 9e-4
>>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L558)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 EtaLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L572)



 (
 


 厄普西隆
 
 ： 漂浮


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1



 )
 


 参数


* **厄普西隆**
 （
 `浮动`
 )—
(0, 1) 范围内的浮点值。用于计算动态截止值的超参数，
 `埃塔`
 。这
论文中的建议值范围为 3e-4 到 4e-3，具体取决于模型的大小。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
发现低于动态截止值的所有值，
 `埃塔`
 ，设置为此浮点值。这
当需要针对应排除的极低概率标记修改 logits 时，该参数非常有用
完全从一代开始。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
指定生成时必须保留的最小令牌数，无论其概率如何。
例如，如果
 `min_tokens_to_keep`
 设置为 1 时，将始终保留至少一个令牌用于生成，
即使所有代币的概率都低于截止值
 `埃塔`
 。


[LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 执行 eta 采样，这是一种过滤掉概率低于动态的标记的技术
截止值，
 `埃塔`
 ，根据超参数的组合计算得出
 `epsilon`
 和熵
令牌概率，即
 `eta := min(epsilon, sqrt(epsilon * e^-熵(概率)))`
 。取最大的
如果没有令牌满足此约束，则需要保留最少令牌。它解决了长期质量差的问题
由神经语言模型生成的文本样本可以产生更加连贯和流畅的文本。看
 [截断
采样作为语言模型去平滑](https://arxiv.org/abs/2210.15191)
 了解更多信息。笔记：
 `做样本`
 必须设置为
 '真实'
 为了这
 `LogitsWarper`
 上班。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

>>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
>>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
>>> # Pro tip: The paper recomends using `eta\_cutoff` values between 3e-4 to 4e-3
>>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L635)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 指数衰减长度罚分


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1310)


（


 指数\_衰减\_长度\_惩罚
 
 : 打字.Tuple[int, float]


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int]]


输入\_ids\_seq\_length
 
 ：整数


）


 参数


* **指数\_衰减\_长度\_惩罚**
 （
 `元组(int, float)`
 )—
该元组应包括：
 `（开始索引，衰减因子）`
 在哪里
 `开始索引`
 表示处罚地点
开始并且
 `衰减因子`
 代表指数衰减因子
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。
* **输入\_ids\_seq\_length**
 （
 `int`
 )—
输入序列的长度。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 从而使分数成倍增加
 `eos_token_id`
 后
 `开始索引`
 已经
到达。这允许生成更短的序列而无需硬截止，从而允许
 `eos_token`
 成为
预测在一个有意义的位置。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")

>>> text = "Just wanted to let you know, I"
>>> inputs = tokenizer(text, return_tensors="pt")

>>> # Generate sequences without exponential penalty. We want short sentences, so we limit max\_length=30
>>> # see that the answer tends to end abruptly
>>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I'm not even a lawyer. I'm a man. I have no real knowledge of politics. I'm a

>>> # Generate sequences with exponential penalty, we add the exponential\_decay\_length\_penalty=(start\_index, decay\_factor)
>>> # We see that instead of cutting at max\_tokens, the output comes to an end before (at 25 tokens) and with more meaning
>>> # What happens is that starting from `start\_index` the EOS token score will be increased by decay\_factor exponentially
>>> outputs = model.generate(
... \*\*inputs,
... do\_sample=True,
... temperature=0.9,
... max\_length=30,
... pad\_token\_id=50256,
... exponential\_decay\_length\_penalty=(15, 1.6),
... )
>>> print(tokenizer.batch\_decode(outputs)[0])
Just wanted to let you know, I've got a very cool t-shirt educating people on how to use the Internet<|endoftext|>

>>> # Generate sequences with smaller decay\_factor, still improving the hard cutoff mid-sentence
>>> outputs = model.generate(
...     **inputs,
...     do_sample=True,
...     temperature=0.9,
...     max_length=30,
...     pad_token_id=50256,
...     exponential_decay_length_penalty=(15, 1.05),
... )
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I've been working on it for about 6 months and now it's in Alpha.<|endoftext|>
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1383)


（


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 ForcedBOSSTokenLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1242)


（


 bos\_token\_id
 
 ：整数


）


 参数


* **bos\_token\_id**
 （
 `int`
 )—
强制作为第一个生成的令牌的令牌的 ID。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制指定的令牌作为第一个生成的令牌。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1254)


（


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 强制EOSTokenLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1264)



 (
 


 最长长度
 
 ：整数


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int]]



 )
 


 参数


* **最长长度**
 （
 `int`
 )—
要生成的序列的最大长度。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 )—
强制作为最后生成的令牌的令牌 ID
 `最大长度`
 到达了。或者，使用
列表设置多个
 *序列结束*
 代币。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制指定的令牌作为最后生成的令牌
 `最大长度`
 到达了。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1282)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 ForceTokensLogits处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1440)



 (
 


 强制\_token\_map
 
 : 打字.List[打字.List[int]]



 )
 


该处理器采用整数对列表，指示从生成索引到令牌的映射
采样前强制执行的索引。处理器会将其日志概率设置为
 `inf`
 以便他们
在其相应的索引处采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1448)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 HammingDiversityLogits处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1079)



 (
 


 多样性\_惩罚
 
 ： 漂浮


num\_beams
 
 ：整数


num\_beam\_groups
 
 ：整数



 )
 


 参数


* **多样性\_惩罚**
 （
 `浮动`
 )—
如果它生成的令牌与来自其他组的任何光束相同的令牌，则从光束的分数中减去该值。
特定的时间。注意
 `多样性惩罚`
 仅当启用组波束搜索时才有效。这
当生成已被另一个波束选择的令牌时，应用于波束得分的惩罚
在同一时间步的同一组内。更高的
 `多样性惩罚`
 将强制执行更大的
波束之间的多样性，使得多个波束选择相同令牌的可能性较小。相反，一个
较低的惩罚将允许梁更自由地选择类似的令牌。调整该值可以帮助实现
多样性和自然可能性之间的平衡。
* **num\_beams**
 （
 `int`
 )—
用于组波束搜索的波束数量。波束搜索是一种用于维护波束（或“多个
假设”）在每一步，扩展每个假设并保留得分最高的序列。更高的
 `num_beams`
 将探索更多潜在的序列。这可以增加找到高质量输出的机会，但也
增加计算成本。
* **num\_beam\_groups**
 （
 `int`
 )—
划分的组数
 `num_beams`
 以确保不同组波束之间的多样性。
每组光束将独立运行，选择令牌而不考虑其他光束的选择
组。这种划分通过确保不同组内的波束探索不同的方式来促进多样性
路径。例如，如果
 `num_beams`
 是 6 并且
 `num_beam_groups`
 为 2，则有 2 组，每组包含
3 根横梁。的选择
 `num_beam_groups`
 应考虑所需的输出多样性水平
和梁的总数。看
 [本文](https://arxiv.org/pdf/1610.02424.pdf)
 更多细节。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制执行多样化的波束搜索。


请注意，此 logits 处理器仅对
 [PreTrainedModel.group\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.group_beam_search)
 。看
 [多样化光束
搜索：从神经序列模型解码多种解决方案](https://arxiv.org/pdf/1610.02424.pdf)
 了解更多
细节。


多样化波束搜索在需要各种不同输出的场景中特别有用，而不是
比多个相似的序列。它允许模型探索不同的生成路径并提供更广泛的
覆盖可能的产出。


 此 Logits 处理器可能会占用大量资源，尤其是在使用大型模型或长序列时。


传统的波束搜索通常会在不同的波束上生成非常相似的序列。
 `HammingDiversityLogitsProcessor`
 通过惩罚生成已被其他人选择的令牌的波束来解决这个问题
光束在同一时间步长。


怎么运行的：


* **分组光束**
 ：梁被分成组。每个组独立地选择代币。
* **惩罚重复的令牌**
 ：如果一个组中的一个梁选择了一个已经被另一个组中选择的令牌
在同一步骤中，该代币的分数将受到惩罚。
* **促进多元化**
 ：此惩罚会阻止组内的光束选择与
其他组中的梁。


好处：


* **多样化的输出**
 ：产生各种不同的序列。
* **探索**
 ：允许模型探索不同的路径。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> # Initialize the model and tokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> # A long text about the solar system
>>> text = "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud."
>>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

>>> # Generate diverse summary
>>> outputs_diverse = model.generate(
...     **inputs,
...     num_beam_groups=2,
...     diversity_penalty=10.0,
...     max_length=100,
...     num_beams=4,
...     num_return_sequences=2,
... )
>>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

>>> # Generate non-diverse summary
>>> outputs_non_diverse = model.generate(
...     **inputs,
...     max_length=100,
...     num_beams=4,
...     num_return_sequences=2,
... )
>>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

>>> # With `diversity\_penalty`, the resulting beams are much more diverse
>>> print(summary_non_diverse)
['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

>>> print(summaries_diverse)
['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1196)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


当前\_代币
 
 : 长张量


梁\_group\_idx
 
 ：整数


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用时，这些可以是每个词汇的逻辑
使用波束搜索时，针对每个词汇标记进行波束搜索或 log softmax
* **当前\_tokens**
 （
 `torch.LongTensor`
 形状的
 `（批量大小）`
 )—
词汇表中输入序列标记的索引，对应于对方选择的标记
当前生成步骤中的梁组。
* **梁\_group\_idx**
 （
 `int`
 )—
当前正在处理的光束组的索引。


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 InfNanRemoveLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1293)


（
 

 ）


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 删除所有
 ‘南’
 和
 `inf`
 值以避免生成方法失败。请注意，使用
logits 处理器仅应在必要时使用，因为它会减慢生成方法的速度。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1299)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 Logit归一化


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1394)


（
 

 ）


[LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 和
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 使用 log-softmax 标准化分数。正常化很重要
在应用 Logits 处理器或扭曲器之后，波束搜索期间的分数，因为使用的搜索算法
这个库没有这样做（它以前只这样做过，但它们可能需要重新规范化），但它仍然假设
比较假设时，分数被归一化。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1402)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 逻辑处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L44)


（
 

 ）


可以在生成过程中应用的所有 Logit 处理器的抽象基类。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L47)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 Logits处理器列表


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L64)



 (
 


 可迭代的
 
 = ()



 )
 


该类可用于创建列表
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 或者
 [LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 随后处理
 `分数`
 输入张量。该类继承自list并添加了特定的
 ***称呼***
 方法来应用每个
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 或者
 [LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 到输入。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L71)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用时，这些可以是每个词汇的逻辑
使用波束搜索时，针对每个词汇标记进行波束搜索或 log softmax
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
特定于 Logits 处理器的附加 kwargs。


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 逻辑扭曲器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L54)


（
 

 ）


所有 Logit 扭曲器的抽象基类，可在多项采样生成过程中应用。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L57)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 最小长度日志处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L101)



 (
 


 最小\_长度
 
 ：整数


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int]]



 )
 


 参数


* **最小\_长度**
 （
 `int`
 )—
低于该分数的最小长度
 `eos_token_id`
 被设定为
 `-float("Inf")`
 。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 通过将 EOS 概率设置为 0 来强制执行最小长度。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L124)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


class
 

 transformers.
 

 MinNewTokensLengthLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L133)



 (
 


 提示\_length\_to\_skip
 
 ：整数


最少\_新\_令牌
 
 ：整数


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int]]



 )
 


 参数


* **提示\_length\_to\_skip**
 （
 `int`
 )—
输入标记长度。与以下命令一起使用时不是有效的参数
 `生成`
 因为它会自动分配
输入长度。
* **min\_new\_tokens**
 （
 `int`
 )—
最低
 *新的*
 标记长度低于该分数
 `eos_token_id`
 被设定为
 `-float("Inf")`
 。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 通过将 EOS（序列结束）令牌概率设置为 0 来强制执行新令牌的最小长度。
请注意，对于仅解码器模型，例如 Llama2，
 `最小长度`
 将计算的长度
 `提示+新生成的令牌`
 而对于其他模型，它将表现为
 `min_new_tokens`
 ，也就是说，只考虑
新生成的。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> model.config.pad_token_id = model.config.eos_token_id
>>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

>>> # If the maximum length (default = 20) is smaller than the minimum length constraint, the latter is ignored!
>>> outputs = model.generate(**inputs, min_new_tokens=30)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Hugging Face Company is a company that has been working on a new product for the past year.

>>> # For testing purposes, let's set `eos\_token` to `"company"`, the first generated token. This will make
>>> # generation end there.
>>> outputs = model.generate(**inputs, eos_token_id=1664)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Hugging Face Company is a company

>>> # Increasing `min\_new\_tokens` will make generation ignore occurences `"company"` (eos token) before the
>>> # minimum length condition is honored.
>>> outputs = model.generate(**inputs, min_new_tokens=2, eos_token_id=1664)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Hugging Face Company is a new company
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L195)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 NoBadWordsLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L965)



 (
 


 坏\_words\_ids
 
 : 打字.List[打字.List[int]]


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int]]



 )
 


 参数


* **坏\_words\_ids**
 （
 `列表[列表[int]]`
 )—
不允许生成的令牌 ID 列表的列表。
* **eos\_token\_id**
 （
 `联合[int，列表[int]]`
 )—
的 ID
 *序列结束*
 令牌。 （可选）使用列表来设置多个
 *序列结束*
 代币。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制指定的序列永远不会被选择。


为了获取不应出现在生成文本中的单词的 token id，请确保设置
 `add_prefix_space=True`
 初始化分词器时，并使用
 `tokenizer(bad_words, add_special_tokens=False).input_ids`
 。这
 `添加前缀空间`
 仅某些缓慢的分词器支持参数，
因为快速分词器的前缀行为来自
 `预分词器`
 。阅读更多
 [此处](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)
 。


例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

>>> output_ids = model.generate(inputs["input\_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a mess.

>>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
>>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


>>> def get\_tokens\_as\_list(word\_list):
...“将单词序列转换为标记列表”
... tokens_list = []
...对于 word_list 中的单词：
... tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
... tokens_list.append(tokenized_word)
...返回 tokens_list


>>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
>>> output_ids = model.generate(
...     inputs["input\_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
... )
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a surprise.
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L888)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 NoRepeatNGramLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L718)



 (
 


 ngram\_size
 
 ：整数



 )
 


 参数


* **ngram\_size**
 （
 `int`
 )—
所有 ngram 大小
 `ngram_size`
 只能发生一次。


 N-gram 是取自文本序列的“n”个连续单词、字符或标记的组。鉴于
句子：“她跑得快”，二元语法（n=2）将是（“她”，“跑”）和（“跑”，“快”）。在文本生成中，
避免单词序列的重复可以提供更加多样化的输出。这
 [LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制执行不
通过将禁止标记的分数设置为负无穷来重复 n 元语法，从而消除这些标记
进一步处理分数时予以考虑。
 [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345)
 。


谨慎使用 n 元语法惩罚。例如，在一篇关于纽约市的文章中惩罚 2 克（二元组）
可能会导致不良结果，即城市名称在整个文本中只出现一次。
 [参考](https://huggingface.co/blog/how-to-generate)


例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> inputs = tokenizer(["Today I"], return_tensors="pt")

>>> output = model.generate(**inputs)
>>> print(tokenizer.decode(output[0], skip_special_tokens=True))
Today I’m not sure if I’m going to be able to do it.

>>> # Now let's add ngram size using `no\_repeat\_ngram\_size`. This stops the repetitions ("I’m") in the output.
>>> output = model.generate(**inputs, no_repeat_ngram_size=2)
>>> print(tokenizer.decode(output[0], skip_special_tokens=True))
Today I’m not sure if I can get a better understanding of the nature of this issue
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L764)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 前缀约束逻辑处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1052)



 (
 


 前缀\_允许\_令牌\_fn
 
 : 打字.Callable[[int, torch.Tensor], 打字.List[int]]


num\_beams
 
 ：整数



 )
 


 参数


* **前缀\_允许\_令牌\_fn**
 （
 `Callable[[int, torch.Tensor], List[int]]`
 )—
该函数将波束搜索限制为仅在每一步允许的标记。该函数需要 2
论点
 `inputs_ids`
 和批次 ID
 `batch_id`
 。它必须返回一个列表，其中包含允许的令牌
下一代步骤以先前生成的令牌为条件
 `inputs_ids`
 和批次 ID
 `batch_id`
 。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 强制约束生成，对于前缀条件约束很有用
一代。看
 [自回归实体检索](https://arxiv.org/abs/2010.00904)
 了解更多信息。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1069)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 重复处罚日志处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L270)



 (
 


 惩罚
 
 ： 漂浮



 )
 


 参数


* **惩罚**
 （
 `浮动`
 )—
重复惩罚的参数。 1.0 表示没有处罚。高于 1.0 会惩罚之前生成的
代币。 0.0 到 1.0 之间奖励先前生成的代币。看
 [这
论文](https://arxiv.org/pdf/1909.05858.pdf)
 更多细节。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 通过指数惩罚来防止先前令牌的重复。这种技术
与覆盖机制和其他旨在减少重复的机制有一些相似之处。正文期间
生成过程中，下一个标记的概率分布是使用包含以下公式的公式确定的：
令牌分数基于它们在生成序列中的出现。分数越高的代币更有可能被
已选择。公式可以看原文
 [论文](https://arxiv.org/pdf/1909.05858.pdf)
 。根据
paper 大约 1.2 的惩罚在真实生成和缺乏重复之间取得了良好的平衡。


这种技术也可以用来奖励，从而鼓励以类似的方式重复。予以处罚和减少
重复、使用
 `惩罚`
 值高于 1.0，值越高，惩罚越重。奖励和鼓励
重复、使用
 `惩罚`
 值介于 0.0 和 1.0 之间，其中值越低，奖励越强烈。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> # Initializing the model and tokenizer for it
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

>>> # This shows a normal generate without any specific parameters
>>> summary_ids = model.generate(**inputs)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
I'm not going to be able to do that. I'm going to be able to do that

>>> # This generates a penalty for repeated tokens
>>> penalized_ids = model.generate(**inputs, repetition_penalty=1.1)
>>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
I'm not going to be able to do that. I'll just have to go out and play
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L317)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 序列偏差逻辑处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L817)



 (
 


 序列\_偏差
 
 : 打字.Dict[打字.Tuple[int], float]



 )
 


 参数


* **序列\_偏差**
 （
 `字典[元组[int]，浮点数]`
 )—
将标记序列映射到其偏置项的字典。积极的偏见会增加
选择序列，而负偏差则相反。如果一个序列的长度为1，它的偏差
将始终被应用。否则，只有当相关序列即将被应用时，才会应用偏差
已完成（在应用该处理器后的令牌选择步骤中）。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 对序列应用附加偏差。偏差应用于序列的最后一个标记
当下一个生成的令牌可以完成它时。因此，要充分利用超过
一个令牌，考虑使用波束方法（优雅地解决部分完成的序列，该序列具有
负偏差）并将偏差应用于其前缀（以确保更早地应用偏差）。


为了获取您想要偏置的序列的 token id，请确保设置
 `add_prefix_space=True`
 什么时候
初始化标记器，并使用
 `tokenizer(bad_words, add_special_tokens=False).input_ids`
 。这
 `添加前缀空间`
 仅某些慢速分词器支持参数，因为快速分词器的前缀行为
来自
 `预分词器`
 。阅读更多
 [此处](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)
 。


例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

>>> summary_ids = model.generate(inputs["input\_ids"], max_new_tokens=4)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald J. Trump Jr

>>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
>>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


>>> def get\_tokens\_as\_tuple(word):
...返回元组(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


>>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
>>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
>>> biased_ids = model.generate(inputs["input\_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald J. Donald,

>>> biased_ids = model.generate(inputs["input\_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald Rumsfeld,

>>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
>>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
>>> biased_ids = model.generate(inputs["input\_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald Duck.
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L888)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


class
 

 transformers.
 

 SuppressTokensAtBeginLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1408)



 (
 


 开始\_抑制\_令牌


 开始\_索引




 )
 


[SuppressTokensAtBeginLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.SuppressTokensAtBeginLogitsProcessor)
 立即抑制令牌列表
 `生成`
 函数开始
生成使用
 `开始索引`
 代币。这应该确保由定义的令牌
 `begin_suppress_tokens`
 在不
在一代开始时进行采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1419)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 抑制令牌日志处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1427)



 (
 


 抑制\_tokens




 )
 


该处理器可用于抑制令牌列表。处理器会将其日志概率设置为
 `-inf`
 以便他们
没有被抽样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1434)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 温度LogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L205)



 (
 


 温度
 
 ： 漂浮



 )
 


 参数


* **温度**
 （
 `浮动`
 )—
用于调节 logits 分布的严格正浮点值。一个值小于
 `1`
 减少
随机性（反之亦然），
 `0`
 相当于将所有概率质量转移到最有可能的位置
令牌。


[LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 对于温度（指数缩放输出概率分布），这实际上意味着
它可以控制预测标记的随机性。


确保
 `do_sample=True`
 包含在
 `生成`
 否则温度值不会有参数
任何效果。


例子：



```
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)  # for reproducibility

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> model.config.pad_token_id = model.config.eos_token_id
>>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

>>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
>>> generate_kwargs = {"max\_new\_tokens": 10, "do\_sample": True, "temperature": 1.0, "num\_return\_sequences": 2}
>>> outputs = model.generate(**inputs, **generate_kwargs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Hugging Face Company is a joint venture between GEO Group, one of',
'Hugging Face Company is not an exact science – but what we believe does']

>>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
>>> generate_kwargs["temperature"] = 0.0001
>>> outputs = model.generate(**inputs, **generate_kwargs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Hugging Face Company is a company that has been around for over 20 years',
'Hugging Face Company is a company that has been around for over 20 years']
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L264)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 顶级KLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L427)



 (
 


 顶部\_k
 
 ：整数


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1



 )
 


 参数


* **顶部\_k**
 （
 `int`
 )—
为 top-k 过滤保留的最高概率词汇标记的数量。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 执行 top-k，即限制到 k 个最高概率元素。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L447)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 顶部PLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L364)



 (
 


 顶部\_p
 
 ： 漂浮


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1



 )
 


 参数


* **顶部\_p**
 （
 `浮动`
 )—
如果设置为< 1，则仅最有可能的标记的最小集合，其概率总计为
 `top_p`
 或者
更高的保留一代。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 执行 top-p，即限制顶部标记总和为 prob\_cut\_off <= prob\_cut\_off。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> # With sampling, the output is unexpected -- sometimes too unexpected.
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

>>> # With `top\_p` sampling, the output gets restricted to high-probability tokens.
>>> # Pro tip: In practice, LLMs use `top\_p` in the 0.9-0.95 range.
>>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L411)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 典型LogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L456)



 (
 


 大量的
 
 ：浮动= 0.9


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1



 )
 


 参数


* **大量的**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 0.9) —
典型值介于 0 和 1 之间（含 0 和 1），默认为 0.9。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[LogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsWarper)
 执行典型的解码。看
 [自然语言的典型解码
一代](https://arxiv.org/abs/2202.00666)
 了解更多信息。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L481)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


class
 

 transformers.
 

 UnbatchedClassifierFreeGuidanceLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1636)



 (
 


 指导\_规模
 
 ： 漂浮


模型


 无条件\_ids
 
 : 打字.Optional[torch.LongTensor] = None


无条件\_attention\_mask
 
 : 打字.Optional[torch.LongTensor] = None


使用\_cache
 
 : 打字.Optional[bool] = True



 )
 


 参数


* **指导\_scale**
 （
 `浮动`
 )—
无分类器引导（CFG）的引导量表。 CFG 通过设置启用
 `guidance_scale！= 1`
 。
更高的指导尺度鼓励模型生成与输入联系更紧密的样本
及时，通常以质量较差为代价。小于 1 的值具有相反的效果，而
使由 negative\_prompt\_ids（如果有）提供的否定提示充当肯定提示。
* **模型**
 （
 `预训练模型`
 )—
计算无条件分数的模型。据说与计算条件的相同
分数。两个模型必须使用相同的分词器。
* **无条件\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 ,
 *选修的*
 )—
无条件分支词汇表中输入序列标记的索引。如果未设置，将默认为
提示的最后一个标记。
* **无条件\_attention\_mask**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 ,
 *选修的*
 )—
无条件\_ids 的注意掩码。
* **使用\_cache**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 '真实'
 )—
是否在否定提示前向传递期间缓存键/值。


 用于无分类器指导 (CFG) 的 Logits 处理器。处理器
计算提示条件和提示无条件（或负）logits 分数的加权平均值，
参数化为
 `指导规模`
 。无条件分数是通过提示在内部计算的
 `模型`
 和
这
 `无条件_id`
 分支。


看
 [论文](https://arxiv.org/abs/2306.17806)
 了解更多信息。


 例子：



```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
>>> out = model.generate(inputs["input\_ids"], guidance_scale=1.5)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

>>> # with a negative prompt
>>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
>>> out = model.generate(inputs["input\_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input\_ids"])
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

>>> # with a positive prompt
>>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
>>> out = model.generate(inputs["input\_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input\_ids"])
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1742)



 (
 


 输入\_ids


 分数




 )
 



### 


班级
 

 变压器。
 

 WhisperTimeStampLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1458)



 (
 


 生成\_config




 )
 


 参数


* **生成\_config**
 （
 `生成配置`
 )—
用于生成输出的生成配置。需要以下参数：
eos\_token\_id (
 `int`
 ,
 *选修的*
 ，默认为 50257):
的 ID
 *序列结束*
 令牌。
无\_timestamps\_token\_id (
 `int`
 ,
 *选修的*
 ，默认为 50363）：
的 ID
 `“<|无时间戳|>”`
 令牌。
最大初始时间戳索引 (
 `int`
 ,
 *选修的*
 ，默认为 1)：
用于设置初始时间戳的最大值。这用于防止模型
预测未来太远的时间戳。


[LogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.LogitsProcessor)
 修改转录中生成时间戳的逻辑。当输入
令牌处于特定阈值时，处理器将分数设置为负无穷大。处理器确保
通过屏蔽掉会破坏这种配对模式的逻辑，时间戳标记成对出现。这是
这样做是为了保持生成的时间戳的一致性和结构。它还确保当预测
对任何时间戳记号进行采样的概率大于对任何单个非时间戳记号进行采样的概率，这些
非时间戳 logits 设置为负无穷大。这样做是为了确保时间戳的生成优于其他时间戳
潜在的代币。


看
 [论文](https://arxiv.org/abs/2212.04356)
 了解更多信息。


 例子：



```
>>> import torch
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration,GenerationConfig
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
>>> ds = load_dataset("hf-internal-testing/librispeech\_asr\_dummy", "clean", split="validation")
>>> inputs = processor(ds[3]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features

>>> #Displaying timestamps
>>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
>>> print("Transcription:", transcription)
Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>


>>> #No timestamps & change EOS:
>>> #This allows the user to select a specific token to terminate the sequence on, in this case it's the word "can"(460)
>>> model.generation_config.eos_token_id = 460
>>> generated_ids = model.generate(inputs=input_features,return_timestamps=False)
>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print("Transcription:", transcription)
Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can
```




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/logits_process.py#L1522)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


）
 

 →


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 [什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax


退货


导出常量元数据='未定义';
 

`torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。




### 


 TensorFlow



### 


班级
 

 变压器。
 

 TFForcedBOSTokenLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L448)



 (
 


 bos\_token\_id
 
 ：整数



 )
 


 参数


* **bos\_token\_id**
 （
 `int`
 )—
强制作为第一个生成的令牌的令牌的 ID。


[TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 强制指定的令牌作为第一个生成的令牌。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L462)



 (
 


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数



 )
 



### 


班级
 

 变压器。
 

 TFForcedEOSTokenLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L478)



 (
 


 最长长度
 
 ：整数


eos\_token\_id
 
 ：整数



 )
 


 参数


* **最长长度**
 （
 `int`
 )—
要生成的序列的最大长度。
* **eos\_token\_id**
 （
 `int`
 )—
强制作为最后生成的令牌的令牌 ID
 `最大长度`
 到达了。


[TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 强制指定的令牌作为最后生成的令牌
 `最大长度`
 到达了。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L495)



 (
 


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数



 )
 



### 


班级
 

 变压器。
 

 TForceTokensLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L551)



 (
 


 强制\_token\_map
 
 : 打字.List[打字.List[int]]



 )
 


该处理器采用整数对列表，指示从生成索引到令牌的映射
采样前强制执行的索引。处理器会将其日志概率设置为
 `0`
 以及所有其他代币
 `-inf`
 以便在相应的索引处对它们进行采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L567)



 (
 


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数



 )
 



### 


班级
 

 变压器。
 

 TFLogits处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L53)


（
 

 ）


可以在生成过程中应用的所有 Logit 处理器的抽象基类。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L56)



 (
 


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）
 

 →


导出常量元数据='未定义';
 

`tf.张量`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `tf.张量`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `tf.张量`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时，搜索或记录每个词汇标记的 softmax。
* **当前\_len**
 （
 `int`
 )—
有效输入序列标记的当前长度。在TF实现中，输入\_ids的序列长度
是生成可以产生的最大长度，我们需要知道它的哪些标记是有效的。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
附加 logits 处理器特定的 kwargs。


退货


导出常量元数据='未定义';
 

`tf.张量`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


处理 logits 的 TF 方法。


### 


班级
 

 变压器。
 

 TFLogits处理器列表


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L75)



 (
 


 可迭代的
 
 = ()



 )
 


该类可用于创建列表
 [TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 随后处理
 `分数`
 输入张量。
该类继承自list并添加了特定的
 ***称呼***
 方法来应用每个
 [TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 到
输入。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L82)



 (
 


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`tf.张量`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `tf.张量`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `tf.张量`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时，搜索或记录每个词汇标记的 softmax。
* **当前\_len**
 （
 `int`
 )—
有效输入序列标记的当前长度。在TF实现中，输入\_ids的序列长度
是生成可以产生的最大长度，我们需要知道它的哪些标记是有效的。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
附加 logits 处理器特定的 kwargs。


退货


导出常量元数据='未定义';
 

`tf.张量`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 TFLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L64)


（
 

 ）


所有 Logit 扭曲器的抽象基类，可在多项采样生成过程中应用。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L67)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）
 

 →


导出常量元数据='未定义';
 

`tf.张量`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `tf.张量`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `tf.张量`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时，搜索或记录每个词汇标记的 softmax。
* **当前\_len**
 （
 `int`
 )—
有效输入序列标记的当前长度。在TF实现中，输入\_ids的序列长度
是生成可以产生的最大长度，我们需要知道它的哪些标记是有效的。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
附加 logits 处理器特定的 kwargs。


退货


导出常量元数据='未定义';
 

`tf.张量`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


用于扭曲 logits 的 TF 方法。


### 


班级
 

 变压器。
 

 TFMinLengthLogits处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L202)


（


 最小\_长度
 
 ：整数


eos\_token\_id
 
 ：整数


）


 参数


* **最小\_长度**
 （
 `int`
 )—
低于该分数的最小长度
 `eos_token_id`
 被设定为
 `-float("Inf")`
 。
* **eos\_token\_id**
 （
 `int`
 )—
的 ID
 *序列结束*
 令牌。


[TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 通过将 EOS 概率设置为 0 来强制执行最小长度。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L228)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 TFNoBadWordsLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L288)


（


 坏\_words\_ids
 
 : 打字.List[打字.List[int]]


eos\_token\_id
 
 ：整数


）


 参数


* **坏\_words\_ids**
 （
 `列表[列表[int]]`
 )—
不允许生成的令牌 ID 列表的列表。为了得到单词的token
不应出现在生成的文本中，请确保设置
 `add_prefix_space=True`
 初始化时
分词器，并使用
 `tokenizer(bad_words, add_special_tokens=False).input_ids`
 。这
 `添加前缀空间`
 仅某些慢速分词器支持参数，因为快速分词器的前缀行为来自
 `预分词器`
 。阅读更多
 [此处](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)
 。
* **eos\_token\_id**
 （
 `int`
 )—
的 ID
 *序列结束*
 令牌。


[TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 强制指定的序列永远不会被采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L367)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 TFNoRepeatNGramLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L388)


（


 ngram\_size
 
 ：整数


）


 参数


* **ngram\_size**
 （
 `int`
 )—
所有 ngram 大小
 `ngram_size`
 只能发生一次。


[TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 强制 n 元语法不重复。看
 [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345)
 。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L427)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


class
 

 transformers.
 

 TFRepetitionPenaltyLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L238)


（


 惩罚
 
 ： 漂浮


）


 参数


* **重复\_惩罚**
 （
 `浮动`
 )—
重复惩罚的参数。 1.0 表示没有处罚。看
 [这
论文](https://arxiv.org/pdf/1909.05858.pdf)
 更多细节。


[TFLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsProcessor)
 对重复序列实施指数惩罚。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L280)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


class
 

 transformers.
 

 TFSuppressTokensAtBeginLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L511)


（


 开始\_抑制\_令牌


 开始\_索引


）


[TFSuppressTokensAtBeginLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFSuppressTokensAtBeginLogitsProcessor)
 立即抑制令牌列表
 `生成`
 函数开始
生成使用
 `开始索引`
 代币。这应该确保由定义的令牌
 `begin_suppress_tokens`
 在不
在一代开始时进行采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L522)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 TFSuppressTokensLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L535)


（


 抑制\_tokens


）


该处理器可用于抑制令牌列表。处理器会将其日志概率设置为
 `-inf`
 以便他们
没有被抽样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L542)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 TFTemperatureLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L98)


（


 温度
 
 ： 漂浮


）


 参数


* **温度**
 （
 `浮动`
 )—
用于对 Logits 分布进行建模的值。


[TFLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsWarper)
 对于温度（指数缩放输出概率分布）。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L113)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 TTopKLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L118)


（


 顶部\_k
 
 ：整数


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1


）


 参数


* **顶部\_k**
 （
 `int`
 )—
为 top-k 过滤保留的最高概率词汇标记的数量。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[TFLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsWarper)
 执行 top-k，即限制到 k 个最高概率元素。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L138)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 TTopPLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L146)


（


 顶部\_p
 
 ： 漂浮


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1


）


 参数


* **顶部\_p**
 （
 `浮动`
 )—
如果设置为< 1，则仅最有可能的标记的最小集合，其概率总计为
 `top_p`
 或者
更高的保留一代。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[TFLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.TFLogitsWarper)
 执行 top-p，即限制顶部标记总和 <= prob\_cut\_off。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_logits_process.py#L170)


（


 输入\_ids
 
 ：张量


分数
 
 ：张量


当前\_len
 
 ：整数


）


### 


 亚麻



### 


class
 

 transformers.
 

 FlaxForcedBOSTokenLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L194)


（


 bos\_token\_id
 
 ：整数


）


 参数


* **bos\_token\_id**
 （
 `int`
 )—
强制作为第一个生成的令牌的令牌的 ID。


[FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 强制指定的令牌作为第一个生成的令牌。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L206)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数


）



### 


class
 

 transformers.
 

 FlaxForcedEOSTokenLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L216)


（


 最长长度
 
 ：整数


eos\_token\_id
 
 ：整数


）


 参数


* **最长长度**
 （
 `int`
 )—
要生成的序列的最大长度。
* **eos\_token\_id**
 （
 `int`
 )—
强制作为最后生成的令牌的令牌 ID
 `最大长度`
 到达了。


[FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 强制指定的令牌作为最后生成的令牌
 `最大长度`
 到达了。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L231)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 FlaxForceTokensLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L315)


（


 强制\_token\_map


）


 参数


* **强制\_token\_map**
 （
 `列表`
 )—
提供令牌 ID 和索引的映射，将在其中强制对它们进行采样。


[FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 它采用整数对列表，指示从生成索引到
采样前将强制执行的令牌索引。处理器会将其日志概率设置为 0 以及所有其他标记
到
 `-inf`
 以便在相应的索引处对它们进行采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L337)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 FlaxLogits处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L50)


（
 

 ）


可以在生成过程中应用的所有 Logit 处理器的抽象基类。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L53)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


）
 

 →


导出常量元数据='未定义';
 

`jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
附加 logits 处理器特定的 kwargs。


退货


导出常量元数据='未定义';
 

`jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


用于处理 logits 的 Flax 方法。


### 


班级
 

 变压器。
 

 FlaxLogits处理器列表


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L72)


（


 可迭代的
 
 = ()


）


该类可用于创建列表
 [FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 或者
 [FlaxLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsWarper)
 进行后续处理
A
 `分数`
 输入张量。该类继承自list并添加了特定的
 ***称呼***
 方法来应用每个
 [FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 或者
 [FlaxLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsWarper)
 到输入。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L79)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
附加 logits 处理器特定的 kwargs。


退货


导出常量元数据='未定义';
 

`jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


### 


班级
 

 变压器。
 

 亚麻Logits整经机


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L61)


（
 

 ）


所有 Logit 扭曲器的抽象基类，可在多项采样生成过程中应用。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L64)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


）
 

 →


导出常量元数据='未定义';
 

`jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`


 参数


* **输入\_ids**
 （
 `jnp.ndarray`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。当不使用 Beam 时，这些可以是每个词汇的 Logits
使用波束搜索时搜索或记录每个词汇标记的 softmax
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
附加 logits 处理器特定的 kwargs。


退货


导出常量元数据='未定义';
 

`jnp.ndarray`
 形状的
 `(batch_size, config.vocab_size)`


导出常量元数据='未定义';


处理后的预测分数。


用于扭曲 Logits 的亚麻方法。


### 


班级
 

 变压器。
 

 FlaxMinLengthLogits处理器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L241)


（


 最小\_长度
 
 ：整数


eos\_token\_id
 
 ：整数


）


 参数


* **最小\_长度**
 （
 `int`
 )—
低于该分数的最小长度
 `eos_token_id`
 被设定为
 `-float("Inf")`
 。
* **eos\_token\_id**
 （
 `int`
 )—
的 ID
 *序列结束*
 令牌。


[FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 通过将 EOS 概率设置为 0 来强制执行最小长度。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L262)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数


）



### 


class
 

 transformers.
 

 FlaxSuppressTokensAtBeginLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L271)


（


 开始\_抑制\_令牌


 开始\_索引


）


 参数


* **开始\_抑制\_令牌**
 （
 `列表[整数]`
 )—
不进行采样的令牌。
* **开始\_index**
 （
 `int`
 )—
标记被抑制的索引。


[FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 立即抑制令牌列表
 `生成`
 函数开始生成使用
 `开始索引`
 代币。这应该确保由定义的令牌
 `begin_suppress_tokens`
 不在采样
一代的开始。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L288)


（


 输入\_ids


 分数


 当前\_len
 
 ：整数


）



### 


class
 

 transformers.
 

 FlaxSuppressTokensLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L296)


（


 抑制\_tokens
 
 ： 列表


）


 参数


* **抑制\_tokens**
 （
 `列表`
 )—
不进行采样的令牌。


[FlaxLogitsProcessor](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsProcessor)
 在每个解码步骤抑制令牌列表。处理器将设置它们的日志概率
成为
 `-inf`
 所以它们没有被采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L309)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 亚麻温度LogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L95)


（


 温度
 
 ： 漂浮


）


 参数


* **温度**
 （
 `浮动`
 )—
用于对 Logits 分布进行建模的值。


[FlaxLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsWarper)
 对于温度（指数缩放输出概率分布）。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L110)


（


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数


）



### 


班级
 

 变压器。
 

 FlaxTopKLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L159)


（


 顶部\_k
 
 ：整数


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1


）


 参数


* **顶部\_k**
 （
 `int`
 )—
为 top-k 过滤保留的最高概率词汇标记的数量。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[FlaxLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsWarper)
 执行 top-k，即限制到 k 个最高概率元素。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L179)



 (
 


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数



 )
 



### 


班级
 

 变压器。
 

 FlaxTopPLogitsWarper


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L115)



 (
 


 顶部\_p
 
 ： 漂浮


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1



 )
 


 参数


* **顶部\_p**
 （
 `浮动`
 )—
如果设置为< 1，则仅最有可能的标记的最小集合，其概率总计为
 `top_p`
 或者
更高的保留一代。
* **过滤\_值**
 （
 `浮动`
 ,
 *选修的*
 ，默认为-inf）—
所有过滤的值都将设置为此浮点值。
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
无法过滤的最小令牌数。


[FlaxLogitsWarper](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.FlaxLogitsWarper)
 执行 top-p，即限制顶部标记总和为 prob\_cut\_off <= prob\_cut\_off。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L139)



 (
 


 输入\_ids
 
 ： 大批


分数
 
 ： 大批


当前\_len
 
 ：整数



 )
 



### 


class
 

 transformers.
 

 FlaxWhisperTimeStampLogitsProcessor


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L363)



 (
 


 生成\_config


 模型\_config


 解码器\_输入\_长度




 )
 


 参数


* **生成\_config**
 （
 `生成配置`
 )—
用于生成输出的生成配置。需要以下参数：
eos\_token\_id (
 `int`
 ,
 *选修的*
 ，默认为 50257):
的 ID
 *序列结束*
 令牌。
无\_timestamps\_token\_id (
 `int`
 ,
 *选修的*
 ，默认为 50363）：
的 ID
 `“<|无时间戳|>”`
 令牌。
最大初始时间戳索引 (
 `int`
 ,
 *选修的*
 ，默认为 1)：
用于设置初始时间戳的最大值。这用于防止模型
预测未来太远的时间戳。


 Whisper 特定处理器。该处理器可用于强制生成令牌列表。处理器将设置他们的日志
问题
 `inf`
 以便在相应的索引处对它们进行采样。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/flax_logits_process.py#L397)



 (
 


 输入\_ids


 分数


 当前\_len




 )
 


## 停止标准



A
 [StoppingCriteria](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.StoppingCriteria)
 可用于更改停止生成的时间（EOS 代币除外）。请注意，这仅适用于我们的 PyTorch 实现。



### 


班级
 

 变压器。
 

 停止标准


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L37)


（
 

 ）


可在生成期间应用的所有停止条件的抽象基类。


如果您的停止标准取决于
 `分数`
 输入，确保通过
 `return_dict_in_generate=True，output_scores=True`
 到
 `生成`
 。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L44)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


\*\*夸格




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [AutoTokenizer](/docs/transformers/v4.35.2/en/model_doc/auto#transformers.AutoTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。这些可以是 SoftMax 之前每个词汇标记的分数
或 SoftMax 之后每个词汇标记的分数。如果此停止标准取决于
 `分数`
 输入，
确保你通过
 `return_dict_in_generate=True，output_scores=True`
 到
 `生成`
 。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
特定 kwargs 的附加停止标准。




### 


班级
 

 变压器。
 

 停止标准列表


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L129)



 (
 


 可迭代的
 
 = ()



 )
 




#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L130)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


\*\*夸格




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [AutoTokenizer](/docs/transformers/v4.35.2/en/model_doc/auto#transformers.AutoTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。这些可以是 SoftMax 之前每个词汇标记的分数
或 SoftMax 之后每个词汇标记的分数。如果此停止标准取决于
 `分数`
 输入，
确保你通过
 `return_dict_in_generate=True，output_scores=True`
 到
 `生成`
 。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
特定 kwargs 的附加停止标准。




### 


班级
 

 变压器。
 

 最大长度标准


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L49)



 (
 


 最长长度
 
 ：整数


最大\_位置\_嵌入
 
 : 打字.Optional[int] = None



 )
 


 参数


* **最长长度**
 （
 `int`
 )—
输出序列可以具有的最大长度（以标记数计）。
* **最大\_位置\_嵌入**
 （
 `int`
 ,
 *选修的*
 )—
最大模型长度，由模型的定义
 `config.max_position_embeddings`
 属性。


 当令牌的完整生成数量超过时，此类可用于停止生成
 `最大长度`
 。保持
请记住，对于仅解码器类型的变压器，这将包括初始提示令牌。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L65)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


\*\*夸格




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [AutoTokenizer](/docs/transformers/v4.35.2/en/model_doc/auto#transformers.AutoTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。这些可以是 SoftMax 之前每个词汇标记的分数
或 SoftMax 之后每个词汇标记的分数。如果此停止标准取决于
 `分数`
 输入，
确保你通过
 `return_dict_in_generate=True，output_scores=True`
 到
 `生成`
 。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
特定 kwargs 的附加停止标准。




### 


班级
 

 变压器。
 

 最大时间标准


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L107)



 (
 


 最长\_时间
 
 ： 漂浮


初始时间戳
 
 : 打字.可选[浮点] = 无



 )
 


 参数


* **最大\_时间**
 （
 `浮动`
 )—
生成所允许的最大时间（以秒为单位）。
* **初始\_时间**
 （
 `浮动`
 ,
 *选修的*
 ，默认为
 `时间.时间()`
 )—
这一代的开始给了时间。


 当完整生成超过一定时间时，此类可用于停止生成。默认情况下，
当您初始化此函数时，时间将开始计算。您可以通过传递一个来覆盖它
 `初始时间`
 。



#### 


\_\_称呼\_\_


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/stopping_criteria.py#L124)



 (
 


 输入\_ids
 
 : 长张量


分数
 
 : 浮点张量


\*\*夸格




 )
 


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（批量大小，序列长度）`
 )—
词汇表中输入序列标记的索引。
 
 指数可以使用以下方式获得
 [AutoTokenizer](/docs/transformers/v4.35.2/en/model_doc/auto#transformers.AutoTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, config.vocab_size)`
 )—
语言建模头的预测分数。这些可以是 SoftMax 之前每个词汇标记的分数
或 SoftMax 之后每个词汇标记的分数。如果此停止标准取决于
 `分数`
 输入，
确保你通过
 `return_dict_in_generate=True，output_scores=True`
 到
 `生成`
 。
* **夸格斯**
 （
 `字典[str，任意]`
 ,
 *选修的*
 )—
特定 kwargs 的附加停止标准。


## 约束条件



A
 [约束](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.Constraint)
 可用于强制生成在输出中包含特定标记或序列。请注意，这仅适用于我们的 PyTorch 实现。



### 


班级
 

 变压器。
 

 约束


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L5)


（
 

 ）


可在生成期间应用的所有约束的抽象基类。
它必须定义如何满足约束。


所有继承 Constraint 的类都必须遵循以下要求：



```
completed = False
while not completed:
    _, completed = constraint.update(constraint.advance())
```


将始终终止（停止）。



#### 


进步


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L48)


（
 

 ）
 

 →


导出常量元数据='未定义';
 

 令牌\_ids(
 `火炬张量`
 ）


 退货


导出常量元数据='未定义';
 

 令牌\_ids(
 `火炬张量`
 ）


导出常量元数据='未定义';


必须是可索引标记列表的张量，而不是某个整数。


调用时，返回令牌，该令牌将使此约束更接近实现。




#### 


复制


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L113)



 (
 


 有状态的
 
 = 假


）
 

 →


导出常量元数据='未定义';
 

 约束（
 `约束`
 ）


 退货


导出常量元数据='未定义';
 

 约束（
 `约束`
 ）


导出常量元数据='未定义';


与被调用的约束相同。


创建此约束的新实例。




#### 


确实\_前进


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L60)



 (
 


 令牌\_id
 
 ：整数



 )
 


读入令牌并返回它是否创建进度。




#### 


其余的


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L104)


（
 

 ）


返回剩余步数
 `前进()`
 为了完成这个约束。




#### 


重置


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L94)


（
 

 ）


将此约束的状态重置为其初始化状态。在满足以下条件的情况下，我们将称之为
约束被不需要的标记破坏。




#### 


测试


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L24)


（
 

 ）


测试此约束是否已正确定义。




#### 


更新


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L69)



 (
 


 令牌\_id
 
 ：整数


）
 

 →


导出常量元数据='未定义';
 

 阶梯（
 `布尔`
 ）


 退货


导出常量元数据='未定义';
 

 阶梯（
 `布尔`
 ）


导出常量元数据='未定义';


这一约束是否距离实现又近了一步。
完全的（
 `布尔`
 ）：
生成的令牌是否完全满足此约束。
重置 （
 `布尔`
 ）：
此约束是否已通过生成此令牌重置其进度。


读入令牌并返回指示其进度的布尔值。该功能将更新
该对象的状态与
 `does_advance(self, token_id: int)`
 。


这并不是为了测试某个代币是否会推进进度；它会更新它的状态，就像它已经
已生成。如果 token\_id != 所需的令牌（请参阅中的 else 语句），这一点就变得很重要
短语约束）


### 


班级
 

 变压器。
 

 短语约束


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L129)



 (
 


 令牌\_ids
 
 : 打字.List[int]



 )
 


 参数


* **令牌\_ids**
 （
 `列表[整数]`
 )—
必须由输出生成的令牌的 id。


[约束](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.Constraint)
 强制在输出中包含有序的标记序列。




### 


班级
 

 变压器。
 

 析取约束


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L261)



 (
 


 嵌套\_token\_ids
 
 : 打字.List[打字.List[int]]



 )
 


 参数


* **嵌套\_token\_ids**
 （
 `列表[列表[int]]`
 )—
单词列表，其中每个单词都是 id 列表。通过仅生成一个来满足此约束
单词列表。


 一个特别的
 [约束](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.Constraint)
 只需满足几个约束之一即可实现这一点。




### 


班级
 

 变压器。
 

 约束列表状态


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L351)



 (
 


 限制条件
 
 : 打字.List[transformers. Generation.beam\_constraints.Constraint]



 )
 


 参数


* **限制**
 （
 `列表[约束]`
 )—
的列表
 [约束](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.Constraint)
 光束刻划机必须实现的目标。


 光束评分器通过一系列约束来跟踪其进度的类。



#### 


进步


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L383)


（
 

 ）


要生成的令牌列表，以便我们能够取得进展。
我们所说的“列表”并不是指完全满足约束的令牌列表。


给定约束条件
 `c_i = {t_ij | j == 标记数量}`
 ，如果我们没有处于进展之中
特定约束
 `c_i`
 ， 我们回来：


`[t_k1 表示未满足约束的索引中的 k]`


如果我们处于约束中间，那么我们返回：
 `[t_ij]`
 ， 在哪里
 ‘我’
 是进行中约束的索引，
 `j`
 是约束的下一步。


尽管我们不关心首先满足哪个约束，但如果我们正在满足约束，
这是我们唯一会回来的。




#### 


重置


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_constraints.py#L418)



 (
 


 令牌\_ids
 
 : 打字.可选[打字.List[int]]



 )
 


token\_ids：到目前为止生成的令牌，用于通过约束重置进度状态。


## 光束搜索




### 


班级
 

 变压器。
 

 光束记录仪


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L91)


（
 

 ）


用于所有光束评分器的抽象基类
 [beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_search)
 和
 [beam\_sample()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.beam_sample)
 。



#### 


过程


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L97)



 (
 


 输入\_ids
 
 : 长张量


下一个\_分数
 
 : 浮点张量


下一个\_令牌
 
 : 长张量


下一个\_索引
 
 : 长张量


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`用户字典`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_beams，sequence_length）`
 )—
词汇表中输入序列标记的索引。
 
 可以使用继承自的任何类来获取索引
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **下一个\_分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, 2 * num_beams)`
 )—
目前成绩名列前茅
 `2 * num_beams`
 未完成的梁假设。
* **下一个\_令牌**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, 2 * num_beams)`
 )—
 `输入ID`
 与顶部对应的令牌的数量
 `2 * num_beams`
 未完成的梁假设。
* **下一个\_索引**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, 2 * num_beams)`
 )—
梁指数表明哪个梁假设
 `下一个令牌`
 对应。
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
* **梁\_索引**
 （
 `torch.LongTensor`
 ,
 *选修的*
 )—
波束索引指示每个标记对应于哪个波束假设。
* **组\_索引**
 （
 `int`
 ,
 *选修的*
 )—
梁组的索引。与使用
 [group\_beam\_search()](/docs/transformers/v4.35.2/en/main_classes/text_ Generation#transformers.GenerationMixin.group_beam_search)
 。


退货


导出常量元数据='未定义';
 

`用户字典`


导出常量元数据='未定义';


由上面定义的字段组成的字典：


* **下一个\_beam\_scores**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 ) — 所有的更新分数
未完成的梁。
* **下一个\_beam\_tokens**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 ) — 下一个要添加的标记
到未完成的梁\_假设。
* **下一个\_beam\_indices**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 ) — 梁指数
指示下一个令牌应添加到哪个波束。




#### 


敲定


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L109)



 (
 


 输入\_ids
 
 : 长张量


下一个\_分数
 
 : 浮点张量


下一个\_令牌
 
 : 长张量


下一个\_索引
 
 : 长张量


最长长度
 
 ：整数


\*\*夸格


）
 

 →


导出常量元数据='未定义';
 

`torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_beams，sequence_length）`
 )—
词汇表中输入序列标记的索引。
 
 可以使用继承自的任何类来获取索引
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **最终\_beam\_scores**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 )—
所有未完成梁的最终得分。
* **最终\_beam\_tokens**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 )—
要添加到未完成的梁\_假设的最后一个标记。
* **最终\_beam\_indices**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 )—
波束索引指示哪个波束
 `final_beam_tokens`
 应添加。
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


退货


导出常量元数据='未定义';
 

`torch.LongTensor`
 形状的
 `（batch_size * num_return_sequences，sequence_length）`


导出常量元数据='未定义';


生成的序列。
第二个维度（sequence\_length）等于
 `最大长度`
 如果所有批次都提前完成，则或更短
因为
 `eos_token_id`
 。


### 


班级
 

 变压器。
 

 光束搜索记分器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L123)



 (
 


 批量\_大小
 
 ：整数


num\_beams
 
 ：整数


设备
 
 ： 设备


长度\_惩罚
 
 : 打字.可选[浮点] = 1.0


尽早停止
 
 : 打字.Union[str, bool, NoneType] = False


num\_beam\_hyps\_to\_keep
 
 : 打字.Optional[int] = 1


num\_beam\_groups
 
 : 打字.Optional[int] = 1


最长长度
 
 : 打字.Optional[int] = None



 )
 


 参数


* **批量\_大小**
 （
 `int`
 )—
批量大小
 `输入ID`
 标准波束搜索解码是并行运行的。
* **num\_beams**
 （
 `int`
 )—
用于波束搜索的波束数量。
* **设备**
 （
 `火炬设备`
 )—
定义设备类型（
 *例如。*
 ,
 `“CPU”`
 或者
 `“cuda”`
 ）在这个实例上
 `BeamSearchScorer`
 将
分配。
* **长度\_惩罚**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 1.0) —
对基于光束的生成所使用的长度的指数惩罚。它被用作指数
序列长度，进而用于除以序列的分数。由于分数是对数
序列的可能性（即负数），
 `长度惩罚`
 > 0.0 促进更长的序列，而
 `长度惩罚`
 < 0.0 鼓励更短的序列。
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
* **num\_beam\_hyps\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
调用时应返回的波束假设数量
 `~transformer.BeamSearchScorer.finalize`
 。
* **num\_beam\_groups**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
划分的组数
 `num_beams`
 以确保不同组波束之间的多样性。
看
 [本文](https://arxiv.org/pdf/1610.02424.pdf)
 更多细节。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 )—
要生成的序列的最大长度。


[BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 实现标准波束搜索解码。


部分改编自
 [Facebook 的 XLM 束搜索
代码](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529)
 。


多种波束搜索算法及实现参考
 [Ashwin Kalyan 的 DBS
实现](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)




#### 


过程


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L215)



 (
 


 输入\_ids
 
 : 长张量


下一个\_分数
 
 : 浮点张量


下一个\_令牌
 
 : 长张量


下一个\_索引
 
 : 长张量


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None


组\_索引
 
 : 打字.Optional[int] = 0



 )
 


#### 


敲定


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L315)



 (
 


 输入\_ids
 
 : 长张量


最终\_beam\_分数
 
 : 浮点张量


最终\_beam\_tokens
 
 : 长张量


最终\_beam\_索引
 
 : 长张量


最长长度
 
 ：整数


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None



 )
 



### 


班级
 

 变压器。
 

 ConstrainedBeamSearchScorer


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L410)



 (
 


 批量\_大小
 
 ：整数


num\_beams
 
 ：整数


限制条件
 
 : 打字.List[transformers. Generation.beam\_constraints.Constraint]


设备
 
 ： 设备


长度\_惩罚
 
 : 打字.可选[浮点] = 1.0


尽早停止
 
 : 打字.Union[str, bool, NoneType] = False


num\_beam\_hyps\_to\_keep
 
 : 打字.Optional[int] = 1


num\_beam\_groups
 
 : 打字.Optional[int] = 1


最长长度
 
 : 打字.Optional[int] = None



 )
 


 参数


* **批量\_大小**
 （
 `int`
 )—
批量大小
 `输入ID`
 标准波束搜索解码是并行运行的。
* **num\_beams**
 （
 `int`
 )—
用于波束搜索的波束数量。
* **限制**
 （
 `列表[约束]`
 )—
正约束列表表示为
 `约束`
 一代人必须实现的目标
输出。有关更多信息，请参阅文档
 [约束](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.Constraint)
 应该阅读。
* **设备**
 （
 `火炬设备`
 )—
定义设备类型（
 *例如。*
 ,
 `“CPU”`
 或者
 `“cuda”`
 ）在这个实例上
 `BeamSearchScorer`
 将
分配。
* **长度\_惩罚**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 1.0) —
对基于光束的生成所使用的长度的指数惩罚。它被用作指数
序列长度，进而用于除以序列的分数。由于分数是对数
序列的可能性（即负数），
 `长度惩罚`
 > 0.0 促进更长的序列，而
 `长度惩罚`
 < 0.0 鼓励更短的序列。
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
* **num\_beam\_hyps\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
调用时应返回的波束假设数量
 `~transformer.BeamSearchScorer.finalize`
 。
* **num\_beam\_groups**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
划分的组数
 `num_beams`
 以确保不同组波束之间的多样性。
看
 [本文](https://arxiv.org/pdf/1610.02424.pdf)
 更多细节。
* **最长长度**
 （
 `int`
 ,
 *选修的*
 )—
要生成的序列的最大长度。


[BeamScorer](/docs/transformers/v4.35.2/en/internal/Generation_utils#transformers.BeamScorer)
 实施约束波束搜索解码。



#### 


过程


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L504)



 (
 


 输入\_ids
 
 : 长张量


下一个\_分数
 
 : 浮点张量


下一个\_令牌
 
 : 长张量


下一个\_索引
 
 : 长张量


\_所有\_词汇的分数
 
 : 浮点张量


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None


）
 

 →


导出常量元数据='未定义';
 

`用户字典`


 参数


* **输入\_ids**
 （
 `torch.LongTensor`
 形状的
 `（batch_size * num_beams，sequence_length）`
 )—
词汇表中输入序列标记的索引。
 
 可以使用继承自的任何类来获取索引
 [PreTrainedTokenizer](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
 。看
 [PreTrainedTokenizer.encode()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode)
 和
 [预训练分词器。
 **称呼**
 ()](/docs/transformers/v4.35.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__)
 了解详情。


[什么是输入 ID？](../glossary#input-ids)
* **下一个\_分数**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size, 2 * num_beams)`
 )—
目前成绩名列前茅
 `2 * num_beams`
 未完成的梁假设。
* **下一个\_令牌**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, 2 * num_beams)`
 )—
 `输入ID`
 与顶部对应的令牌的数量
 `2 * num_beams`
 未完成的梁假设。
* **下一个\_索引**
 （
 `torch.LongTensor`
 形状的
 `(batch_size, 2 * num_beams)`
 )—
梁指数表明哪个梁假设
 `下一个令牌`
 对应。
* **分数\_for\_all\_vocab**
 （
 `torch.FloatTensor`
 形状的
 `（batch_size * num_beams，sequence_length）`
 )—
每个梁假设的词汇表中所有标记的分数。
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
* **梁\_索引**
 （
 `torch.LongTensor`
 ,
 *选修的*
 )—
波束索引指示每个标记对应于哪个波束假设。


退货


导出常量元数据='未定义';
 

`用户字典`


导出常量元数据='未定义';


由上面定义的字段组成的字典：


* **下一个\_beam\_scores**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 ) — 更新分数
全部
未完成的梁。
* **下一个\_beam\_tokens**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 ) — 下一个令牌是
添加
到未完成的梁\_假设。
* **下一个\_beam\_indices**
 （
 `torch.FloatTensor`
 形状的
 `(batch_size * num_beams)`
 ) — 梁指数
指示下一个令牌应添加到哪个波束。




#### 


敲定


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/beam_search.py​​#L798)



 (
 


 输入\_ids
 
 : 长张量


最终\_beam\_分数
 
 : 浮点张量


最终\_beam\_tokens
 
 : 长张量


最终\_beam\_索引
 
 : 长张量


最长长度
 
 ：整数


垫\_token\_id
 
 : 打字.Optional[int] = None


eos\_token\_id
 
 : 打字.Union[int, 打字.List[int], NoneType] = 无


梁\_索引
 
 : 打字.Optional[torch.LongTensor] = None



 )
 


## 公用事业




#### 


Transformers.top\_k\_top\_p\_filtering


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/utils.py#L4761)



 (
 


 逻辑数
 
 : 浮点张量


顶部\_k
 
 ：整数=0


顶部\_p
 
 ：浮动= 1.0


过滤器\_值
 
 : 浮点数 = -inf


保留的最少\_tokens\_
 
 ：整数= 1



 )
 


 参数


* **顶部\_k**
 （
 `int`
 ,
 *选修的*
 ，默认为 0) —
如果 > 0，则仅保留概率最高的前 k 个标记（top-k 过滤）
* **顶部\_p**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 1.0) —
如果 < 1.0，则仅保留累积概率 >= top\_p （核心过滤）的顶部标记。核
Holtzman 等人描述了过滤。 （
 <http://arxiv.org/abs/1904.09751>
 ）
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
我们在输出中保留每批示例的最小令牌数。


 使用 top-k 和/或 core (top-p) 过滤来过滤 logits 分布


从：
 <https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317>


#### 


变压器.tf\_top\_k\_top\_p\_filtering


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/tf_utils.py#L3068)



 (
 


 逻辑数


 顶部\_k
 
 = 0


顶部\_p
 
 = 1.0


过滤器\_值
 
 =-inf


保留的最少\_tokens\_
 
 = 1



 )
 


 参数


* **顶部\_k**
 （
 `int`
 ,
 *选修的*
 ，默认为 0) —
如果 > 0，则仅保留概率最高的前 k 个标记（top-k 过滤）
* **顶部\_p**
 （
 `浮动`
 ,
 *选修的*
 ，默认为 1.0) —
如果 < 1.0，则仅保留累积概率 >= top\_p （核心过滤）的顶部标记。核
Holtzman 等人描述了过滤。 （
 <http://arxiv.org/abs/1904.09751>
 ）
* **最少\_tokens\_to\_keep**
 （
 `int`
 ,
 *选修的*
 ，默认为 1) —
我们在输出中保留每批示例的最小令牌数。


 使用 top-k 和/或 core (top-p) 过滤来过滤 logits 分布


从：
 <https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317>


## 飘带




### 


班级
 

 变压器。
 

 文本流媒体


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/streamers.py#L38)



 (
 


 分词器
 
 ：自动标记器


跳过\_提示
 
 ：布尔=假


\*\*解码\_kwargs




 )
 


 参数


* **标记器**
 （
 `自动标记器`
 )—
标记化用于解码标记。
* **跳过\_提示**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否跳过提示
 `.generate()`
 或不。有用的，例如对于聊天机器人。
* **解码\_kwargs**
 （
 `字典`
 ,
 *选修的*
 )—
要传递给标记器的附加关键字参数
 `解码`
 方法。


 简单的文本流送器，一旦形成整个单词，就将标记打印到标准输出。


流媒体类的 API 仍在开发中，将来可能会发生变化。


例子：



```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```




#### 


结尾


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/streamers.py#L116)


（
 

 ）


刷新所有剩余的缓存并将换行符打印到标准输出。




#### 


在\_最终\_文本上


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/streamers.py#L130)



 (
 


 文本
 
 : 字符串


流\_end
 
 ：布尔=假



 )
 


将新文本打印到标准输出。如果流结束，还会打印换行符。




#### 


放


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/streamers.py#L82)



 (
 


 价值




 )
 


接收令牌，对其进行解码，并在它们形成完整单词后将其打印到标准输出。


### 


班级
 

 变压器。
 

 文本迭代器流式传输器


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/streamers.py#L159)



 (
 


 分词器
 
 ：自动标记器


跳过\_提示
 
 ：布尔=假


暂停
 
 : 打字.可选[浮点] = 无


\*\*解码\_kwargs




 )
 


 参数


* **标记器**
 （
 `自动标记器`
 )—
标记化用于解码标记。
* **跳过\_提示**
 （
 `布尔`
 ,
 *选修的*
 ，默认为
 ‘假’
 )—
是否跳过提示
 `.generate()`
 或不。有用的，例如对于聊天机器人。
* **暂停**
 （
 `浮动`
 ,
 *选修的*
 )—
文本队列的超时。如果
 `无`
 ，队列将无限期阻塞。有助于处理异常
在
 `.generate()`
 ，当它在单独的线程中调用时。
* **解码\_kwargs**
 （
 `字典`
 ,
 *选修的*
 )—
要传递给标记器的附加关键字参数
 `解码`
 方法。


 Streamer 将打印就绪文本存储在队列中，供下游应用程序用作迭代器。这是
对于那些受益于以非阻塞方式访问生成的文本的应用程序（例如，以交互式方式）非常有用
广播演示）。


流媒体类的 API 仍在开发中，将来可能会发生变化。


例子：



```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
>>> from threading import Thread

>>> tok = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextIteratorStreamer(tok)

>>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
>>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
>>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
>>> thread.start()
>>> generated_text = ""
>>> for new_text in streamer:
...     generated_text += new_text
>>> generated_text
'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
```




#### 


在\_最终\_文本上


[<
 

 来源
 

 >](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/Generation/streamers.py#L213)



 (
 


 文本
 
 : 字符串


流\_end
 
 ：布尔=假



 )
 


将新文本放入队列中。如果流即将结束，还要在队列中放置一个停止信号。