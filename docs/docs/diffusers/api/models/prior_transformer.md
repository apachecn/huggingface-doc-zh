# Prior Transformer

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/prior_transformer>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/prior_transformer>



 The Prior Transformer was originally introduced in
 [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://huggingface.co/papers/2204.06125) 
 by Ramesh et al. It is used to predict CLIP image embeddings from CLIP text embeddings; image embeddings are predicted through a denoising diffusion process.
 



 The abstract from the paper is:
 



*Contrastive models like CLIP have been shown to learn robust representations of images that capture both semantics and style. To leverage these representations for image generation, we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. We show that explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity. Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation. Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion. We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples.* 


## PriorTransformer




### 




 class
 

 diffusers.
 

 PriorTransformer




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/prior_transformer.py#L36)



 (
 


 num\_attention\_heads
 
 : int = 32
 




 attention\_head\_dim
 
 : int = 64
 




 num\_layers
 
 : int = 20
 




 embedding\_dim
 
 : int = 768
 




 num\_embeddings
 
 = 77
 




 additional\_embeddings
 
 = 4
 




 dropout
 
 : float = 0.0
 




 time\_embed\_act\_fn
 
 : str = 'silu'
 




 norm\_in\_type
 
 : typing.Optional[str] = None
 




 embedding\_proj\_norm\_type
 
 : typing.Optional[str] = None
 




 encoder\_hid\_proj\_type
 
 : typing.Optional[str] = 'linear'
 




 added\_emb\_type
 
 : typing.Optional[str] = 'prd'
 




 time\_embed\_dim
 
 : typing.Optional[int] = None
 




 embedding\_proj\_dim
 
 : typing.Optional[int] = None
 




 clip\_embed\_dim
 
 : typing.Optional[int] = None
 



 )
 


 Parameters
 




* **num\_attention\_heads** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 32) — The number of heads to use for multi-head attention.
* **attention\_head\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 64) — The number of channels in each head.
* **num\_layers** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 20) — The number of layers of Transformer blocks to use.
* **embedding\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 768) — The dimension of the model input
 `hidden_states`
* **num\_embeddings** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 77) —
The number of embeddings of the model input
 `hidden_states`
* **additional\_embeddings** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) — The number of additional tokens appended to the
projected
 `hidden_states` 
. The actual length of the used
 `hidden_states` 
 is
 `num_embeddings + additional_embeddings` 
.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) — The dropout probability to use.
* **time\_embed\_act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to ‘silu’) —
The activation function to use to create timestep embeddings.
* **norm\_in\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to None) — The normalization layer to apply on hidden states before
passing to Transformer blocks. Set it to
 `None` 
 if normalization is not needed.
* **embedding\_proj\_norm\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to None) —
The normalization layer to apply on the input
 `proj_embedding` 
. Set it to
 `None` 
 if normalization is not
needed.
* **encoder\_hid\_proj\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `linear` 
 ) —
The projection layer to apply on the input
 `encoder_hidden_states` 
. Set it to
 `None` 
 if
 `encoder_hidden_states` 
 is
 `None` 
.
* **added\_emb\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `prd` 
 ) — Additional embeddings to condition the model.
Choose from
 `prd` 
 or
 `None` 
. if choose
 `prd` 
 , it will prepend a token indicating the (quantized) dot
product between the text embedding and image embedding as proposed in the unclip paper
 <https://arxiv.org/abs/2204.06125>
 If it is
 `None` 
 , no additional embeddings will be prepended.
* **time\_embed\_dim** 
 (
 `int, *optional*, defaults to None) -- The dimension of timestep embeddings. If None, will be set to` 
 num\_attention\_heads \* attention\_head\_dim`
* **embedding\_proj\_dim** 
 (
 `int` 
 ,
 *optional* 
 , default to None) —
The dimension of
 `proj_embedding` 
. If None, will be set to
 `embedding_dim` 
.
* **clip\_embed\_dim** 
 (
 `int` 
 ,
 *optional* 
 , default to None) —
The dimension of the output. If None, will be set to
 `embedding_dim` 
.


 A Prior Transformer model.
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/prior_transformer.py#L247)



 (
 


 hidden\_states
 


 timestep
 
 : typing.Union[torch.Tensor, float, int]
 




 proj\_embedding
 
 : FloatTensor
 




 encoder\_hidden\_states
 
 : typing.Optional[torch.FloatTensor] = None
 




 attention\_mask
 
 : typing.Optional[torch.BoolTensor] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[PriorTransformerOutput](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.models.prior_transformer.PriorTransformerOutput) 
 or
 `tuple` 


 Parameters
 




* **hidden\_states** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, embedding_dim)` 
 ) —
The currently predicted image embeddings.
* **timestep** 
 (
 `torch.LongTensor` 
 ) —
Current denoising step.
* **proj\_embedding** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, embedding_dim)` 
 ) —
Projected embedding vector the denoising process is conditioned on.
* **encoder\_hidden\_states** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_embeddings, embedding_dim)` 
 ) —
Hidden states of the text embeddings the denoising process is conditioned on.
* **attention\_mask** 
 (
 `torch.BoolTensor` 
 of shape
 `(batch_size, num_embeddings)` 
 ) —
Text mask for the text embeddings.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [PriorTransformerOutput](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.models.prior_transformer.PriorTransformerOutput) 
 instead of a plain
tuple.




 Returns
 




 export const metadata = 'undefined';
 

[PriorTransformerOutput](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.models.prior_transformer.PriorTransformerOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is True, a
 [PriorTransformerOutput](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.models.prior_transformer.PriorTransformerOutput) 
 is returned, otherwise a
tuple is returned where the first element is the sample tensor.
 



 The
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
 forward method.
 




#### 




 set\_attn\_processor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/prior_transformer.py#L195)



 (
 


 processor
 
 : typing.Union[diffusers.models.attention\_processor.AttnProcessor, diffusers.models.attention\_processor.AttnProcessor2\_0, diffusers.models.attention\_processor.XFormersAttnProcessor, diffusers.models.attention\_processor.SlicedAttnProcessor, diffusers.models.attention\_processor.AttnAddedKVProcessor, diffusers.models.attention\_processor.SlicedAttnAddedKVProcessor, diffusers.models.attention\_processor.AttnAddedKVProcessor2\_0, diffusers.models.attention\_processor.XFormersAttnAddedKVProcessor, diffusers.models.attention\_processor.CustomDiffusionAttnProcessor, diffusers.models.attention\_processor.CustomDiffusionXFormersAttnProcessor, diffusers.models.attention\_processor.CustomDiffusionAttnProcessor2\_0, diffusers.models.attention\_processor.LoRAAttnProcessor, diffusers.models.attention\_processor.LoRAAttnProcessor2\_0, diffusers.models.attention\_processor.LoRAXFormersAttnProcessor, diffusers.models.attention\_processor.LoRAAttnAddedKVProcessor, typing.Dict[str, typing.Union[diffusers.models.attention\_processor.AttnProcessor, diffusers.models.attention\_processor.AttnProcessor2\_0, diffusers.models.attention\_processor.XFormersAttnProcessor, diffusers.models.attention\_processor.SlicedAttnProcessor, diffusers.models.attention\_processor.AttnAddedKVProcessor, diffusers.models.attention\_processor.SlicedAttnAddedKVProcessor, diffusers.models.attention\_processor.AttnAddedKVProcessor2\_0, diffusers.models.attention\_processor.XFormersAttnAddedKVProcessor, diffusers.models.attention\_processor.CustomDiffusionAttnProcessor, diffusers.models.attention\_processor.CustomDiffusionXFormersAttnProcessor, diffusers.models.attention\_processor.CustomDiffusionAttnProcessor2\_0, diffusers.models.attention\_processor.LoRAAttnProcessor, diffusers.models.attention\_processor.LoRAAttnProcessor2\_0, diffusers.models.attention\_processor.LoRAXFormersAttnProcessor, diffusers.models.attention\_processor.LoRAAttnAddedKVProcessor]]]
 




 \_remove\_lora
 
 = False
 



 )
 


 Parameters
 




* **processor** 
 (
 `dict` 
 of
 `AttentionProcessor` 
 or only
 `AttentionProcessor` 
 ) —
The instantiated processor class or a dictionary of processor classes that will be set as the processor
for
 **all** 
`Attention` 
 layers.
 
 If
 `processor` 
 is a dict, the key needs to define the path to the corresponding cross attention
processor. This is strongly recommended when setting trainable attention processors.


 Sets the attention processor to use to compute attention.
 




#### 




 set\_default\_attn\_processor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/prior_transformer.py#L232)



 (
 

 )
 




 Disables custom attention processors and sets the default attention implementation.
 


## PriorTransformerOutput




### 




 class
 

 diffusers.models.prior\_transformer.
 

 PriorTransformerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/prior_transformer.py#L24)



 (
 


 predicted\_image\_embedding
 
 : FloatTensor
 



 )
 


 Parameters
 




* **predicted\_image\_embedding** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, embedding_dim)` 
 ) —
The predicted CLIP image embedding conditioned on the CLIP text embedding input.


 The output of
 [PriorTransformer](/docs/diffusers/v0.23.0/en/api/models/prior_transformer#diffusers.PriorTransformer) 
.