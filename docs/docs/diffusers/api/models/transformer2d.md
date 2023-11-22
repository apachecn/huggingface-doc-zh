# Transformer2D

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/transformer2d>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/transformer2d>



 A Transformer model for image-like data from
 [CompVis](https://huggingface.co/CompVis) 
 that is based on the
 [Vision Transformer](https://huggingface.co/papers/2010.11929) 
 introduced by Dosovitskiy et al. The
 [Transformer2DModel](/docs/diffusers/v0.23.0/en/api/models/transformer2d#diffusers.Transformer2DModel) 
 accepts discrete (classes of vector embeddings) or continuous (actual embeddings) inputs.
 



 When the input is
 **continuous** 
 :
 


1. Project the input and reshape it to
 `(batch_size, sequence_length, feature_dimension)` 
.
2. Apply the Transformer blocks in the standard way.
3. Reshape to image.



 When the input is
 **discrete** 
 :
 




 It is assumed one of the input classes is the masked latent pixel. The predicted classes of the unnoised image don’t contain a prediction for the masked pixel because the unnoised image cannot be masked.
 



1. Convert input (classes of latent pixels) to embeddings and apply positional embeddings.
2. Apply the Transformer blocks in the standard way.
3. Predict classes of unnoised image.


## Transformer2DModel




### 




 class
 

 diffusers.
 

 Transformer2DModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/transformer_2d.py#L45)



 (
 


 num\_attention\_heads
 
 : int = 16
 




 attention\_head\_dim
 
 : int = 88
 




 in\_channels
 
 : typing.Optional[int] = None
 




 out\_channels
 
 : typing.Optional[int] = None
 




 num\_layers
 
 : int = 1
 




 dropout
 
 : float = 0.0
 




 norm\_num\_groups
 
 : int = 32
 




 cross\_attention\_dim
 
 : typing.Optional[int] = None
 




 attention\_bias
 
 : bool = False
 




 sample\_size
 
 : typing.Optional[int] = None
 




 num\_vector\_embeds
 
 : typing.Optional[int] = None
 




 patch\_size
 
 : typing.Optional[int] = None
 




 activation\_fn
 
 : str = 'geglu'
 




 num\_embeds\_ada\_norm
 
 : typing.Optional[int] = None
 




 use\_linear\_projection
 
 : bool = False
 




 only\_cross\_attention
 
 : bool = False
 




 double\_self\_attention
 
 : bool = False
 




 upcast\_attention
 
 : bool = False
 




 norm\_type
 
 : str = 'layer\_norm'
 




 norm\_elementwise\_affine
 
 : bool = True
 




 norm\_eps
 
 : float = 1e-05
 




 attention\_type
 
 : str = 'default'
 




 caption\_channels
 
 : int = None
 



 )
 


 Parameters
 




* **num\_attention\_heads** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 16) — The number of heads to use for multi-head attention.
* **attention\_head\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 88) — The number of channels in each head.
* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of channels in the input and output (specify if the input is
 **continuous** 
 ).
* **num\_layers** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) — The number of layers of Transformer blocks to use.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) — The dropout probability to use.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 ) — The number of
 `encoder_hidden_states` 
 dimensions to use.
* **sample\_size** 
 (
 `int` 
 ,
 *optional* 
 ) — The width of the latent images (specify if the input is
 **discrete** 
 ).
This is fixed during training since it is used to learn a number of position embeddings.
* **num\_vector\_embeds** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of classes of the vector embeddings of the latent pixels (specify if the input is
 **discrete** 
 ).
Includes the class for the masked latent pixel.
* **activation\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"geglu"` 
 ) — Activation function to use in feed-forward.
* **num\_embeds\_ada\_norm** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of diffusion steps used during training. Pass if at least one of the norm\_layers is
 `AdaLayerNorm` 
. This is fixed during training since it is used to learn a number of embeddings that are
added to the hidden states.
 
 During inference, you can denoise for up to but not more steps than
 `num_embeds_ada_norm` 
.
* **attention\_bias** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Configure if the
 `TransformerBlocks` 
 attention should contain a bias parameter.


 A 2D Transformer model for image-like data.
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/transformer_2d.py#L240)



 (
 


 hidden\_states
 
 : Tensor
 




 encoder\_hidden\_states
 
 : typing.Optional[torch.Tensor] = None
 




 timestep
 
 : typing.Optional[torch.LongTensor] = None
 




 added\_cond\_kwargs
 
 : typing.Dict[str, torch.Tensor] = None
 




 class\_labels
 
 : typing.Optional[torch.LongTensor] = None
 




 cross\_attention\_kwargs
 
 : typing.Dict[str, typing.Any] = None
 




 attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 encoder\_attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 return\_dict
 
 : bool = True
 



 )
 


 Parameters
 




* **hidden\_states** 
 (
 `torch.LongTensor` 
 of shape
 `(batch size, num latent pixels)` 
 if discrete,
 `torch.FloatTensor` 
 of shape
 `(batch size, channel, height, width)` 
 if continuous) —
Input
 `hidden_states` 
.
* **encoder\_hidden\_states** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch size, sequence len, embed dims)` 
 ,
 *optional* 
 ) —
Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
self-attention.
* **timestep** 
 (
 `torch.LongTensor` 
 ,
 *optional* 
 ) —
Used to indicate denoising step. Optional timestep to be applied as an embedding in
 `AdaLayerNorm` 
.
* **class\_labels** 
 (
 `torch.LongTensor` 
 of shape
 `(batch size, num classes)` 
 ,
 *optional* 
 ) —
Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
 `AdaLayerZeroNorm` 
.
* **cross\_attention\_kwargs** 
 (
 `Dict[str, Any]` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.
* **attention\_mask** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 ) —
An attention mask of shape
 `(batch, key_tokens)` 
 is applied to
 `encoder_hidden_states` 
. If
 `1` 
 the mask
is kept, otherwise if
 `0` 
 it is discarded. Mask will be converted into a bias, which adds large
negative values to the attention scores corresponding to “discard” tokens.
* **encoder\_attention\_mask** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 ) —
Cross-attention mask applied to
 `encoder_hidden_states` 
. Two formats supported:
 
	+ Mask
	 `(batch, sequence_length)` 
	 True = keep, False = discard.
	+ Bias
	 `(batch, 1, sequence_length)` 
	 0 = keep, -10000 = discard.

 If
 `ndim == 2` 
 : will be interpreted as a mask, then converted into a bias consistent with the format
above. This bias will be added to the cross-attention scores.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 instead of a plain
tuple.


 The
 [Transformer2DModel](/docs/diffusers/v0.23.0/en/api/models/transformer2d#diffusers.Transformer2DModel) 
 forward method.
 


## Transformer2DModelOutput




### 




 class
 

 diffusers.models.transformer\_2d.
 

 Transformer2DModelOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/transformer_2d.py#L32)



 (
 


 sample
 
 : FloatTensor
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 or
 `(batch size, num_vector_embeds - 1, num_latent_pixels)` 
 if
 [Transformer2DModel](/docs/diffusers/v0.23.0/en/api/models/transformer2d#diffusers.Transformer2DModel) 
 is discrete) —
The hidden states output conditioned on the
 `encoder_hidden_states` 
 input. If discrete, returns probability
distributions for the unnoised latent pixels.


 The output of
 [Transformer2DModel](/docs/diffusers/v0.23.0/en/api/models/transformer2d#diffusers.Transformer2DModel) 
.