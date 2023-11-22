# Transformer Temporal

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/transformer_temporal>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/transformer_temporal>



 A Transformer model for video-like data.
 


## TransformerTemporalModel




### 




 class
 

 diffusers.models.
 

 TransformerTemporalModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/transformer_temporal.py#L39)



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
 




 activation\_fn
 
 : str = 'geglu'
 




 norm\_elementwise\_affine
 
 : bool = True
 




 double\_self\_attention
 
 : bool = True
 




 positional\_embeddings
 
 : typing.Optional[str] = None
 




 num\_positional\_embeddings
 
 : typing.Optional[int] = None
 



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
* **attention\_bias** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Configure if the
 `TransformerBlock` 
 attention should contain a bias parameter.
* **sample\_size** 
 (
 `int` 
 ,
 *optional* 
 ) — The width of the latent images (specify if the input is
 **discrete** 
 ).
This is fixed during training since it is used to learn a number of position embeddings.
* **activation\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"geglu"` 
 ) —
Activation function to use in feed-forward. See
 `diffusers.models.activations.get_activation` 
 for supported
activation functions.
* **norm\_elementwise\_affine** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Configure if the
 `TransformerBlock` 
 should use learnable elementwise affine parameters for normalization.
* **double\_self\_attention** 
 (
 `bool` 
 ,
 *optional* 
 ) —
Configure if each
 `TransformerBlock` 
 should contain two self-attention layers.
positional\_embeddings — (
 `str` 
 ,
 *optional* 
 ):
The type of positional embeddings to apply to the sequence input before passing use.
num\_positional\_embeddings — (
 `int` 
 ,
 *optional* 
 ):
The maximum length of the sequence over which to apply positional embeddings.


 A Transformer model for video-like data.
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/transformer_temporal.py#L119)



 (
 


 hidden\_states
 
 : FloatTensor
 




 encoder\_hidden\_states
 
 : typing.Optional[torch.LongTensor] = None
 




 timestep
 
 : typing.Optional[torch.LongTensor] = None
 




 class\_labels
 
 : LongTensor = None
 




 num\_frames
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[TransformerTemporalModelOutput](/docs/diffusers/v0.23.0/en/api/models/transformer_temporal#diffusers.models.transformer_temporal.TransformerTemporalModelOutput) 
 or
 `tuple` 


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
Input hidden\_states.
* **encoder\_hidden\_states** 
 (
 `torch.LongTensor` 
 of shape
 `(batch size, encoder_hidden_states dim)` 
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
* **num\_frames** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of frames to be processed per batch. This is used to reshape the hidden states.
* **cross\_attention\_kwargs** 
 (
 `dict` 
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




 Returns
 




 export const metadata = 'undefined';
 

[TransformerTemporalModelOutput](/docs/diffusers/v0.23.0/en/api/models/transformer_temporal#diffusers.models.transformer_temporal.TransformerTemporalModelOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is True, an
 [TransformerTemporalModelOutput](/docs/diffusers/v0.23.0/en/api/models/transformer_temporal#diffusers.models.transformer_temporal.TransformerTemporalModelOutput) 
 is
returned, otherwise a
 `tuple` 
 where the first element is the sample tensor.
 



 The
 `TransformerTemporal` 
 forward method.
 


## TransformerTemporalModelOutput




### 




 class
 

 diffusers.models.transformer\_temporal.
 

 TransformerTemporalModelOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/transformer_temporal.py#L27)



 (
 


 sample
 
 : FloatTensor
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size x num_frames, num_channels, height, width)` 
 ) —
The hidden states output conditioned on
 `encoder_hidden_states` 
 input.


 The output of
 `TransformerTemporalModel` 
.