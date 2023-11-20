# UNet2DConditionModel

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/unet2d-cond>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/unet2d-cond>



 The
 [UNet](https://huggingface.co/papers/1505.04597) 
 model was originally introduced by Ronneberger et al for biomedical image segmentation, but it is also commonly used in 🤗 Diffusers because it outputs images that are the same size as the input. It is one of the most important components of a diffusion system because it facilitates the actual diffusion process. There are several variants of the UNet model in 🤗 Diffusers, depending on it’s number of dimensions and whether it is a conditional model or not. This is a 2D UNet conditional model.
 



 The abstract from the paper is:
 



*There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at
 <http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net>
.* 


## UNet2DConditionModel




### 




 class
 

 diffusers.
 

 UNet2DConditionModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L70)



 (
 


 sample\_size
 
 : typing.Optional[int] = None
 




 in\_channels
 
 : int = 4
 




 out\_channels
 
 : int = 4
 




 center\_input\_sample
 
 : bool = False
 




 flip\_sin\_to\_cos
 
 : bool = True
 




 freq\_shift
 
 : int = 0
 




 down\_block\_types
 
 : typing.Tuple[str] = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
 




 mid\_block\_type
 
 : typing.Optional[str] = 'UNetMidBlock2DCrossAttn'
 




 up\_block\_types
 
 : typing.Tuple[str] = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
 




 only\_cross\_attention
 
 : typing.Union[bool, typing.Tuple[bool]] = False
 




 block\_out\_channels
 
 : typing.Tuple[int] = (320, 640, 1280, 1280)
 




 layers\_per\_block
 
 : typing.Union[int, typing.Tuple[int]] = 2
 




 downsample\_padding
 
 : int = 1
 




 mid\_block\_scale\_factor
 
 : float = 1
 




 dropout
 
 : float = 0.0
 




 act\_fn
 
 : str = 'silu'
 




 norm\_num\_groups
 
 : typing.Optional[int] = 32
 




 norm\_eps
 
 : float = 1e-05
 




 cross\_attention\_dim
 
 : typing.Union[int, typing.Tuple[int]] = 1280
 




 transformer\_layers\_per\_block
 
 : typing.Union[int, typing.Tuple[int], typing.Tuple[typing.Tuple]] = 1
 




 reverse\_transformer\_layers\_per\_block
 
 : typing.Optional[typing.Tuple[typing.Tuple[int]]] = None
 




 encoder\_hid\_dim
 
 : typing.Optional[int] = None
 




 encoder\_hid\_dim\_type
 
 : typing.Optional[str] = None
 




 attention\_head\_dim
 
 : typing.Union[int, typing.Tuple[int]] = 8
 




 num\_attention\_heads
 
 : typing.Union[int, typing.Tuple[int], NoneType] = None
 




 dual\_cross\_attention
 
 : bool = False
 




 use\_linear\_projection
 
 : bool = False
 




 class\_embed\_type
 
 : typing.Optional[str] = None
 




 addition\_embed\_type
 
 : typing.Optional[str] = None
 




 addition\_time\_embed\_dim
 
 : typing.Optional[int] = None
 




 num\_class\_embeds
 
 : typing.Optional[int] = None
 




 upcast\_attention
 
 : bool = False
 




 resnet\_time\_scale\_shift
 
 : str = 'default'
 




 resnet\_skip\_time\_act
 
 : bool = False
 




 resnet\_out\_scale\_factor
 
 : int = 1.0
 




 time\_embedding\_type
 
 : str = 'positional'
 




 time\_embedding\_dim
 
 : typing.Optional[int] = None
 




 time\_embedding\_act\_fn
 
 : typing.Optional[str] = None
 




 timestep\_post\_act
 
 : typing.Optional[str] = None
 




 time\_cond\_proj\_dim
 
 : typing.Optional[int] = None
 




 conv\_in\_kernel
 
 : int = 3
 




 conv\_out\_kernel
 
 : int = 3
 




 projection\_class\_embeddings\_input\_dim
 
 : typing.Optional[int] = None
 




 attention\_type
 
 : str = 'default'
 




 class\_embeddings\_concat
 
 : bool = False
 




 mid\_block\_only\_cross\_attention
 
 : typing.Optional[bool] = None
 




 cross\_attention\_norm
 
 : typing.Optional[str] = None
 




 addition\_embed\_type\_num\_heads
 
 = 64
 



 )
 


 Parameters
 




* **sample\_size** 
 (
 `int` 
 or
 `Tuple[int, int]` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Height and width of input/output sample.
* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) — Number of channels in the input sample.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) — Number of channels in the output.
* **center\_input\_sample** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) — Whether to center the input sample.
* **flip\_sin\_to\_cos** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to flip the sin to cos in the time embedding.
* **freq\_shift** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) — The frequency shift to apply to the time embedding.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")` 
 ) —
The tuple of downsample blocks to use.
* **mid\_block\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"UNetMidBlock2DCrossAttn"` 
 ) —
Block type for middle of UNet, it can be one of
 `UNetMidBlock2DCrossAttn` 
 ,
 `UNetMidBlock2D` 
 , or
 `UNetMidBlock2DSimpleCrossAttn` 
. If
 `None` 
 , the mid block layer is skipped.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")` 
 ) —
The tuple of upsample blocks to use.
* **only\_cross\_attention(
 `bool`**
 or
 `Tuple[bool]` 
 ,
 *optional* 
 , default to
 `False` 
 ) —
Whether to include self-attention in the basic transformer blocks, see
 `BasicTransformerBlock` 
.
* **block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(320, 640, 1280, 1280)` 
 ) —
The tuple of output channels for each block.
* **layers\_per\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 2) — The number of layers per block.
* **downsample\_padding** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) — The padding to use for the downsampling convolution.
* **mid\_block\_scale\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) — The scale factor to use for the mid block.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) — The dropout probability to use.
* **act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"silu"` 
 ) — The activation function to use.
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 32) — The number of groups to use for the normalization.
If
 `None` 
 , normalization and activation layers is skipped in post-processing.
* **norm\_eps** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1e-5) — The epsilon to use for the normalization.
* **cross\_attention\_dim** 
 (
 `int` 
 or
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to 1280) —
The dimension of the cross attention features.
* **transformer\_layers\_per\_block** 
 (
 `int` 
 ,
 `Tuple[int]` 
 , or
 `Tuple[Tuple]` 
 ,
 *optional* 
 , defaults to 1) —
The number of transformer blocks of type
 `BasicTransformerBlock` 
. Only relevant for
 `CrossAttnDownBlock2D` 
 ,
 `CrossAttnUpBlock2D` 
 ,
 `UNetMidBlock2DCrossAttn` 
.


 A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
shaped output.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



 reverse\_transformer\_layers\_per\_block : (
 `Tuple[Tuple]` 
 ,
 *optional* 
 , defaults to None):
The number of transformer blocks of type
 `BasicTransformerBlock` 
 , in the upsampling
blocks of the U-Net. Only relevant if
 `transformer_layers_per_block` 
 is of type
 `Tuple[Tuple]` 
 and for
 `CrossAttnDownBlock2D` 
 ,
 `CrossAttnUpBlock2D` 
 ,
 `UNetMidBlock2DCrossAttn` 
.
encoder\_hid\_dim (
 `int` 
 ,
 *optional* 
 , defaults to None):
If
 `encoder_hid_dim_type` 
 is defined,
 `encoder_hidden_states` 
 will be projected from
 `encoder_hid_dim` 
 dimension to
 `cross_attention_dim` 
.
encoder\_hid\_dim\_type (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
If given, the
 `encoder_hidden_states` 
 and potentially other embeddings are down-projected to text
embeddings of dimension
 `cross_attention` 
 according to
 `encoder_hid_dim_type` 
.
attention\_head\_dim (
 `int` 
 ,
 *optional* 
 , defaults to 8): The dimension of the attention heads.
num\_attention\_heads (
 `int` 
 ,
 *optional* 
 ):
The number of attention heads. If not defined, defaults to
 `attention_head_dim` 
 resnet\_time\_scale\_shift (
 `str` 
 ,
 *optional* 
 , defaults to
 `"default"` 
 ): Time scale shift config
for ResNet blocks (see
 `ResnetBlock2D` 
 ). Choose from
 `default` 
 or
 `scale_shift` 
.
class\_embed\_type (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
The type of class embedding to use which is ultimately summed with the time embeddings. Choose from
 `None` 
 ,
 `"timestep"` 
 ,
 `"identity"` 
 ,
 `"projection"` 
 , or
 `"simple_projection"` 
.
addition\_embed\_type (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
Configures an optional embedding which will be summed with the time embeddings. Choose from
 `None` 
 or
“text”. “text” will use the
 `TextTimeEmbedding` 
 layer.
addition\_time\_embed\_dim: (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
Dimension for the timestep embeddings.
num\_class\_embeds (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
Input dimension of the learnable embedding matrix to be projected to
 `time_embed_dim` 
 , when performing
class conditioning with
 `class_embed_type` 
 equal to
 `None` 
.
time\_embedding\_type (
 `str` 
 ,
 *optional* 
 , defaults to
 `positional` 
 ):
The type of position embedding to use for timesteps. Choose from
 `positional` 
 or
 `fourier` 
.
time\_embedding\_dim (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
An optional override for the dimension of the projected time embedding.
time\_embedding\_act\_fn (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
Optional activation function to use only once on the time embeddings before they are passed to the rest of
the UNet. Choose from
 `silu` 
 ,
 `mish` 
 ,
 `gelu` 
 , and
 `swish` 
.
timestep\_post\_act (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
The second activation function to use in timestep embedding. Choose from
 `silu` 
 ,
 `mish` 
 and
 `gelu` 
.
time\_cond\_proj\_dim (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
The dimension of
 `cond_proj` 
 layer in the timestep embedding.
conv\_in\_kernel (
 `int` 
 ,
 *optional* 
 , default to
 `3` 
 ): The kernel size of
 `conv_in` 
 layer. conv\_out\_kernel (
 `int` 
 ,
 *optional* 
 , default to
 `3` 
 ): The kernel size of
 `conv_out` 
 layer. projection\_class\_embeddings\_input\_dim (
 `int` 
 ,
 *optional* 
 ): The dimension of the
 `class_labels` 
 input when
 `class_embed_type="projection"` 
. Required when
 `class_embed_type="projection"` 
.
class\_embeddings\_concat (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ): Whether to concatenate the time
embeddings with the class embeddings.
mid\_block\_only\_cross\_attention (
 `bool` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
Whether to use cross attention with the mid block when using the
 `UNetMidBlock2DSimpleCrossAttn` 
. If
 `only_cross_attention` 
 is given as a single boolean and
 `mid_block_only_cross_attention` 
 is
 `None` 
 , the
 `only_cross_attention` 
 value is used as the value for
 `mid_block_only_cross_attention` 
. Default to
 `False` 
 otherwise.
 



#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L789)



 (
 

 )
 




 Disables the FreeU mechanism.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L765)



 (
 


 s1
 


 s2
 


 b1
 


 b2
 




 )
 


 Parameters
 




* **s1** 
 (
 `float` 
 ) —
Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
mitigate the “oversmoothing effect” in the enhanced denoising process.
* **s2** 
 (
 `float` 
 ) —
Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
mitigate the “oversmoothing effect” in the enhanced denoising process.
* **b1** 
 (
 `float` 
 ) — Scaling factor for stage 1 to amplify the contributions of backbone features.
* **b2** 
 (
 `float` 
 ) — Scaling factor for stage 2 to amplify the contributions of backbone features.


 Enables the FreeU mechanism from
 <https://arxiv.org/abs/2309.11497>
.
 



 The suffixes after the scaling factors represent the stage blocks where they are being applied.
 



 Please refer to the
 [official repository](https://github.com/ChenyangSi/FreeU) 
 for combinations of values that
are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
 




#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L797)



 (
 


 sample
 
 : FloatTensor
 




 timestep
 
 : typing.Union[torch.Tensor, float, int]
 




 encoder\_hidden\_states
 
 : Tensor
 




 class\_labels
 
 : typing.Optional[torch.Tensor] = None
 




 timestep\_cond
 
 : typing.Optional[torch.Tensor] = None
 




 attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 added\_cond\_kwargs
 
 : typing.Union[typing.Dict[str, torch.Tensor], NoneType] = None
 




 down\_block\_additional\_residuals
 
 : typing.Optional[typing.Tuple[torch.Tensor]] = None
 




 mid\_block\_additional\_residual
 
 : typing.Optional[torch.Tensor] = None
 




 down\_intrablock\_additional\_residuals
 
 : typing.Optional[typing.Tuple[torch.Tensor]] = None
 




 encoder\_attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 or
 `tuple` 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) —
The noisy input tensor with the following shape
 `(batch, channel, height, width)` 
.
* **timestep** 
 (
 `torch.FloatTensor` 
 or
 `float` 
 or
 `int` 
 ) — The number of timesteps to denoise an input.
* **encoder\_hidden\_states** 
 (
 `torch.FloatTensor` 
 ) —
The encoder hidden states with shape
 `(batch, sequence_length, feature_dim)` 
.
* **class\_labels** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
timestep\_cond — (
 `torch.Tensor` 
 ,
 *optional* 
 , defaults to
 `None` 
 ):
Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
through the
 `self.time_embedding` 
 layer to obtain the timestep embeddings.
* **attention\_mask** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 , defaults to
 `None` 
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
added\_cond\_kwargs — (
 `dict` 
 ,
 *optional* 
 ):
A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
are passed along to the UNet blocks.
down\_block\_additional\_residuals — (
 `tuple` 
 of
 `torch.Tensor` 
 ,
 *optional* 
 ):
A tuple of tensors that if specified are added to the residuals of down unet blocks.
mid\_block\_additional\_residual — (
 `torch.Tensor` 
 ,
 *optional* 
 ):
A tensor that if specified is added to the residual of the middle unet block.
* **encoder\_attention\_mask** 
 (
 `torch.Tensor` 
 ) —
A cross-attention mask of shape
 `(batch, sequence_length)` 
 is applied to
 `encoder_hidden_states` 
. If
 `True` 
 the mask is kept, otherwise if
 `False` 
 it is discarded. Mask will be converted into a bias,
which adds large negative values to the attention scores corresponding to “discard” tokens.
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
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttnProcessor` 
.
added\_cond\_kwargs — (
 `dict` 
 ,
 *optional* 
 ):
A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
are passed along to the UNet blocks.
* **down\_block\_additional\_residuals** 
 (
 `tuple` 
 of
 `torch.Tensor` 
 ,
 *optional* 
 ) —
additional residuals to be added to UNet long skip connections from down blocks to up blocks for
example from ControlNet side model(s)
* **mid\_block\_additional\_residual** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 ) —
additional residual to be added to UNet mid block output, for example from ControlNet side model
* **down\_intrablock\_additional\_residuals** 
 (
 `tuple` 
 of
 `torch.Tensor` 
 ,
 *optional* 
 ) —
additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)




 Returns
 




 export const metadata = 'undefined';
 

[UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is True, an
 [UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 is returned, otherwise
a
 `tuple` 
 is returned where the first element is the sample tensor.
 



 The
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 forward method.
 




#### 




 set\_attention\_slice




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L696)



 (
 


 slice\_size
 




 )
 


 Parameters
 




* **slice\_size** 
 (
 `str` 
 or
 `int` 
 or
 `list(int)` 
 ,
 *optional* 
 , defaults to
 `"auto"` 
 ) —
When
 `"auto"` 
 , input to the attention heads is halved, so attention is computed in two steps. If
 `"max"` 
 , maximum amount of memory is saved by running only one slice at a time. If a number is
provided, uses as many slices as
 `attention_head_dim //slice_size` 
. In this case,
 `attention_head_dim` 
 must be a multiple of
 `slice_size` 
.


 Enable sliced attention computation.
 



 When this option is enabled, the attention module splits the input tensor in slices to compute attention in
several steps. This is useful for saving some memory in exchange for a small decrease in speed.
 




#### 




 set\_attn\_processor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L645)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L681)



 (
 

 )
 




 Disables custom attention processors and sets the default attention implementation.
 


## UNet2DConditionOutput




### 




 class
 

 diffusers.models.unet\_2d\_condition.
 

 UNet2DConditionOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition.py#L58)



 (
 


 sample
 
 : FloatTensor = None
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 ) —
The hidden states output conditioned on
 `encoder_hidden_states` 
 input. Output of last layer of model.


 The output of
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
.
 


## FlaxUNet2DConditionModel




### 




 class
 

 diffusers.
 

 FlaxUNet2DConditionModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition_flax.py#L49)



 (
 


 sample\_size
 
 : int = 32
 




 in\_channels
 
 : int = 4
 




 out\_channels
 
 : int = 4
 




 down\_block\_types
 
 : typing.Tuple[str] = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
 




 up\_block\_types
 
 : typing.Tuple[str] = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
 




 only\_cross\_attention
 
 : typing.Union[bool, typing.Tuple[bool]] = False
 




 block\_out\_channels
 
 : typing.Tuple[int] = (320, 640, 1280, 1280)
 




 layers\_per\_block
 
 : int = 2
 




 attention\_head\_dim
 
 : typing.Union[int, typing.Tuple[int]] = 8
 




 num\_attention\_heads
 
 : typing.Union[int, typing.Tuple[int], NoneType] = None
 




 cross\_attention\_dim
 
 : int = 1280
 




 dropout
 
 : float = 0.0
 




 use\_linear\_projection
 
 : bool = False
 




 dtype
 
 : dtype = <class 'jax.numpy.float32'>
 




 flip\_sin\_to\_cos
 
 : bool = True
 




 freq\_shift
 
 : int = 0
 




 use\_memory\_efficient\_attention
 
 : bool = False
 




 split\_head\_dim
 
 : bool = False
 




 transformer\_layers\_per\_block
 
 : typing.Union[int, typing.Tuple[int]] = 1
 




 addition\_embed\_type
 
 : typing.Optional[str] = None
 




 addition\_time\_embed\_dim
 
 : typing.Optional[int] = None
 




 addition\_embed\_type\_num\_heads
 
 : int = 64
 




 projection\_class\_embeddings\_input\_dim
 
 : typing.Optional[int] = None
 




 parent
 
 : typing.Union[typing.Type[flax.linen.module.Module], typing.Type[flax.core.scope.Scope], typing.Type[flax.linen.module.\_Sentinel], NoneType] = <flax.linen.module.\_Sentinel object at 0x7fad2e096f10>
 




 name
 
 : typing.Optional[str] = None
 



 )
 


 Parameters
 




* **sample\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The size of the input sample.
* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) —
The number of channels in the input sample.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) —
The number of channels in the output.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")` 
 ) —
The tuple of downsample blocks to use.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")` 
 ) —
The tuple of upsample blocks to use.
* **block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(320, 640, 1280, 1280)` 
 ) —
The tuple of output channels for each block.
* **layers\_per\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 2) —
The number of layers per block.
* **attention\_head\_dim** 
 (
 `int` 
 or
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to 8) —
The dimension of the attention heads.
* **num\_attention\_heads** 
 (
 `int` 
 or
 `Tuple[int]` 
 ,
 *optional* 
 ) —
The number of attention heads.
* **cross\_attention\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 768) —
The dimension of the cross attention features.
* **dropout** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0) —
Dropout probability for down, up and bottleneck blocks.
* **flip\_sin\_to\_cos** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to flip the sin to cos in the time embedding.
* **freq\_shift** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) — The frequency shift to apply to the time embedding.
* **use\_memory\_efficient\_attention** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Enable memory efficient attention as described
 [here](https://arxiv.org/abs/2112.05682) 
.
* **split\_head\_dim** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.


 A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
shaped output.
 



 This model inherits from
 [FlaxModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin) 
. Check the superclass documentation for it’s generic methods
implemented for all models (such as downloading or saving).
 



 This model is also a Flax Linen
 [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) 
 subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
general usage and behavior.
 



 Inherent JAX features such as the following are supported:
 


* [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
* [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
* [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
* [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)


## FlaxUNet2DConditionOutput




### 




 class
 

 diffusers.models.unet\_2d\_condition\_flax.
 

 FlaxUNet2DConditionOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d_condition_flax.py#L36)



 (
 


 sample
 
 : Array
 



 )
 


 Parameters
 




* **sample** 
 (
 `jnp.ndarray` 
 of shape
 `(batch_size, num_channels, height, width)` 
 ) —
The hidden states output conditioned on
 `encoder_hidden_states` 
 input. Output of last layer of model.


 The output of
 [FlaxUNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.FlaxUNet2DConditionModel) 
.
 



#### 




 replace




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/flax/struct.py#L111)



 (
 


 \*\*updates
 




 )
 




 “Returns a new object replacing the specified fields with new values.