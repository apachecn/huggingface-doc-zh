# Attention Processor

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/attnprocessor>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/attnprocessor>



 An attention processor is a class for applying different types of attention mechanisms.
 


## AttnProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 AttnProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L694)



 (
 

 )
 




 Default processor for performing attention-related computations.
 


## AttnProcessor2\_0




### 




 class
 

 diffusers.models.attention\_processor.
 

 AttnProcessor2\_0




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1166)



 (
 

 )
 




 Processor for implementing scaled dot-product attention (enabled by default if you’re using PyTorch 2.0).
 


## LoRAAttnProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 LoRAAttnProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1693)



 (
 


 hidden\_size
 
 : int
 




 cross\_attention\_dim
 
 : typing.Optional[int] = None
 




 rank
 
 : int = 4
 




 network\_alpha
 
 : typing.Optional[int] = None
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **hidden\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The hidden size of the attention layer.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of channels in the
 `encoder_hidden_states` 
.
* **rank** 
 (
 `int` 
 , defaults to 4) —
The dimension of the LoRA update matrices.
* **network\_alpha** 
 (
 `int` 
 ,
 *optional* 
 ) —
Equivalent to
 `alpha` 
 but it’s usage is specific to Kohya (A1111) style LoRAs.
* **kwargs** 
 (
 `dict` 
 ) —
Additional keyword arguments to pass to the
 `LoRALinearLayer` 
 layers.


 Processor for implementing the LoRA attention mechanism.
 


## LoRAAttnProcessor2\_0




### 




 class
 

 diffusers.models.attention\_processor.
 

 LoRAAttnProcessor2\_0




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1765)



 (
 


 hidden\_size
 
 : int
 




 cross\_attention\_dim
 
 : typing.Optional[int] = None
 




 rank
 
 : int = 4
 




 network\_alpha
 
 : typing.Optional[int] = None
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **hidden\_size** 
 (
 `int` 
 ) —
The hidden size of the attention layer.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of channels in the
 `encoder_hidden_states` 
.
* **rank** 
 (
 `int` 
 , defaults to 4) —
The dimension of the LoRA update matrices.
* **network\_alpha** 
 (
 `int` 
 ,
 *optional* 
 ) —
Equivalent to
 `alpha` 
 but it’s usage is specific to Kohya (A1111) style LoRAs.
* **kwargs** 
 (
 `dict` 
 ) —
Additional keyword arguments to pass to the
 `LoRALinearLayer` 
 layers.


 Processor for implementing the LoRA attention mechanism using PyTorch 2.0’s memory-efficient scaled dot-product
attention.
 


## CustomDiffusionAttnProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 CustomDiffusionAttnProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L763)



 (
 


 train\_kv
 
 : bool = True
 




 train\_q\_out
 
 : bool = True
 




 hidden\_size
 
 : typing.Optional[int] = None
 




 cross\_attention\_dim
 
 : typing.Optional[int] = None
 




 out\_bias
 
 : bool = True
 




 dropout
 
 : float = 0.0
 



 )
 


 Parameters
 




* **train\_kv** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to newly train the key and value matrices corresponding to the text features.
* **train\_q\_out** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to newly train query matrices corresponding to the latent image features.
* **hidden\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The hidden size of the attention layer.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The number of channels in the
 `encoder_hidden_states` 
.
* **out\_bias** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to include the bias parameter in
 `train_q_out` 
.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
The dropout probability to use.


 Processor for implementing attention for the Custom Diffusion method.
 


## CustomDiffusionAttnProcessor2\_0




### 




 class
 

 diffusers.models.attention\_processor.
 

 CustomDiffusionAttnProcessor2\_0




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1370)



 (
 


 train\_kv
 
 : bool = True
 




 train\_q\_out
 
 : bool = True
 




 hidden\_size
 
 : typing.Optional[int] = None
 




 cross\_attention\_dim
 
 : typing.Optional[int] = None
 




 out\_bias
 
 : bool = True
 




 dropout
 
 : float = 0.0
 



 )
 


 Parameters
 




* **train\_kv** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to newly train the key and value matrices corresponding to the text features.
* **train\_q\_out** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to newly train query matrices corresponding to the latent image features.
* **hidden\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The hidden size of the attention layer.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The number of channels in the
 `encoder_hidden_states` 
.
* **out\_bias** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to include the bias parameter in
 `train_q_out` 
.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
The dropout probability to use.


 Processor for implementing attention for the Custom Diffusion method using PyTorch 2.0’s memory-efficient scaled
dot-product attention.
 


## AttnAddedKVProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 AttnAddedKVProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L867)



 (
 

 )
 




 Processor for performing attention-related computations with extra learnable key and value matrices for the text
encoder.
 


## AttnAddedKVProcessor2\_0




### 




 class
 

 diffusers.models.attention\_processor.
 

 AttnAddedKVProcessor2\_0




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L931)



 (
 

 )
 




 Processor for performing scaled dot-product attention (enabled by default if you’re using PyTorch 2.0), with extra
learnable key and value matrices for the text encoder.
 


## LoRAAttnAddedKVProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 LoRAAttnAddedKVProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1919)



 (
 


 hidden\_size
 
 : int
 




 cross\_attention\_dim
 
 : typing.Optional[int] = None
 




 rank
 
 : int = 4
 




 network\_alpha
 
 : typing.Optional[int] = None
 



 )
 


 Parameters
 




* **hidden\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The hidden size of the attention layer.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The number of channels in the
 `encoder_hidden_states` 
.
* **rank** 
 (
 `int` 
 , defaults to 4) —
The dimension of the LoRA update matrices.
* **network\_alpha** 
 (
 `int` 
 ,
 *optional* 
 ) —
Equivalent to
 `alpha` 
 but it’s usage is specific to Kohya (A1111) style LoRAs.
* **kwargs** 
 (
 `dict` 
 ) —
Additional keyword arguments to pass to the
 `LoRALinearLayer` 
 layers.


 Processor for implementing the LoRA attention mechanism with extra learnable key and value matrices for the text
encoder.
 


## XFormersAttnProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 XFormersAttnProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1075)



 (
 


 attention\_op
 
 : typing.Optional[typing.Callable] = None
 



 )
 


 Parameters
 




* **attention\_op** 
 (
 `Callable` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The base
 [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) 
 to
use as the attention operator. It is recommended to set to
 `None` 
 , and allow xFormers to choose the best
operator.


 Processor for implementing memory efficient attention using xFormers.
 


## LoRAXFormersAttnProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 LoRAXFormersAttnProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1840)



 (
 


 hidden\_size
 
 : int
 




 cross\_attention\_dim
 
 : int
 




 rank
 
 : int = 4
 




 attention\_op
 
 : typing.Optional[typing.Callable] = None
 




 network\_alpha
 
 : typing.Optional[int] = None
 




 \*\*kwargs
 




 )
 


 Parameters
 




* **hidden\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The hidden size of the attention layer.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of channels in the
 `encoder_hidden_states` 
.
* **rank** 
 (
 `int` 
 , defaults to 4) —
The dimension of the LoRA update matrices.
* **attention\_op** 
 (
 `Callable` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The base
 [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) 
 to
use as the attention operator. It is recommended to set to
 `None` 
 , and allow xFormers to choose the best
operator.
* **network\_alpha** 
 (
 `int` 
 ,
 *optional* 
 ) —
Equivalent to
 `alpha` 
 but it’s usage is specific to Kohya (A1111) style LoRAs.
* **kwargs** 
 (
 `dict` 
 ) —
Additional keyword arguments to pass to the
 `LoRALinearLayer` 
 layers.


 Processor for implementing the LoRA attention mechanism with memory efficient attention using xFormers.
 


## CustomDiffusionXFormersAttnProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 CustomDiffusionXFormersAttnProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1254)



 (
 


 train\_kv
 
 : bool = True
 




 train\_q\_out
 
 : bool = False
 




 hidden\_size
 
 : typing.Optional[int] = None
 




 cross\_attention\_dim
 
 : typing.Optional[int] = None
 




 out\_bias
 
 : bool = True
 




 dropout
 
 : float = 0.0
 




 attention\_op
 
 : typing.Optional[typing.Callable] = None
 



 )
 


 Parameters
 




* **train\_kv** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to newly train the key and value matrices corresponding to the text features.
* **train\_q\_out** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to newly train query matrices corresponding to the latent image features.
* **hidden\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The hidden size of the attention layer.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The number of channels in the
 `encoder_hidden_states` 
.
* **out\_bias** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to include the bias parameter in
 `train_q_out` 
.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
The dropout probability to use.
* **attention\_op** 
 (
 `Callable` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The base
 [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) 
 to use
as the attention operator. It is recommended to set to
 `None` 
 , and allow xFormers to choose the best operator.


 Processor for implementing memory efficient attention using xFormers for the Custom Diffusion method.
 


## SlicedAttnProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 SlicedAttnProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1484)



 (
 


 slice\_size
 
 : int
 



 )
 


 Parameters
 




* **slice\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of steps to compute attention. Uses as many slices as
 `attention_head_dim //slice_size` 
 , and
 `attention_head_dim` 
 must be a multiple of the
 `slice_size` 
.


 Processor for implementing sliced attention.
 


## SlicedAttnAddedKVProcessor




### 




 class
 

 diffusers.models.attention\_processor.
 

 SlicedAttnAddedKVProcessor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/attention_processor.py#L1571)



 (
 


 slice\_size
 




 )
 


 Parameters
 




* **slice\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of steps to compute attention. Uses as many slices as
 `attention_head_dim //slice_size` 
 , and
 `attention_head_dim` 
 must be a multiple of the
 `slice_size` 
.


 Processor for implementing sliced attention with extra learnable key and value matrices for the text encoder.