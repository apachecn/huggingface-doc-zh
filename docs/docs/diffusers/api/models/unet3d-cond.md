# UNet3DConditionModel

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/unet3d-cond>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/unet3d-cond>



 The
 [UNet](https://huggingface.co/papers/1505.04597) 
 model was originally introduced by Ronneberger et al for biomedical image segmentation, but it is also commonly used in 🤗 Diffusers because it outputs images that are the same size as the input. It is one of the most important components of a diffusion system because it facilitates the actual diffusion process. There are several variants of the UNet model in 🤗 Diffusers, depending on it’s number of dimensions and whether it is a conditional model or not. This is a 3D UNet conditional model.
 



 The abstract from the paper is:
 



*There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at
 <http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net>
.* 


## UNet3DConditionModel




### 




 class
 

 diffusers.
 

 UNet3DConditionModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L62)



 (
 


 sample\_size
 
 : typing.Optional[int] = None
 




 in\_channels
 
 : int = 4
 




 out\_channels
 
 : int = 4
 




 down\_block\_types
 
 : typing.Tuple[str] = ('CrossAttnDownBlock3D', 'CrossAttnDownBlock3D', 'CrossAttnDownBlock3D', 'DownBlock3D')
 




 up\_block\_types
 
 : typing.Tuple[str] = ('UpBlock3D', 'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D')
 




 block\_out\_channels
 
 : typing.Tuple[int] = (320, 640, 1280, 1280)
 




 layers\_per\_block
 
 : int = 2
 




 downsample\_padding
 
 : int = 1
 




 mid\_block\_scale\_factor
 
 : float = 1
 




 act\_fn
 
 : str = 'silu'
 




 norm\_num\_groups
 
 : typing.Optional[int] = 32
 




 norm\_eps
 
 : float = 1e-05
 




 cross\_attention\_dim
 
 : int = 1024
 




 attention\_head\_dim
 
 : typing.Union[int, typing.Tuple[int]] = 64
 




 num\_attention\_heads
 
 : typing.Union[int, typing.Tuple[int], NoneType] = None
 



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
 , defaults to 4) — The number of channels in the input sample.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) — The number of channels in the output.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")` 
 ) —
The tuple of downsample blocks to use.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")` 
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
 ,
 *optional* 
 , defaults to 1280) — The dimension of the cross attention features.
* **attention\_head\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 8) — The dimension of the attention heads.
* **num\_attention\_heads** 
 (
 `int` 
 ,
 *optional* 
 ) — The number of attention heads.


 A conditional 3D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
shaped output.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L492)



 (
 

 )
 




 Disables the FreeU mechanism.
 




#### 




 enable\_forward\_chunking




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L406)



 (
 


 chunk\_size
 
 = None
 




 dim
 
 = 0
 



 )
 


 Parameters
 




* **chunk\_size** 
 (
 `int` 
 ,
 *optional* 
 ) —
The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
over each tensor of dim=
 `dim` 
.
* **dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `0` 
 ) —
The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
or dim=1 (sequence length).


 Sets the attention processor to use
 [feed forward
chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers) 
.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L467)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L500)



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
 




 down\_block\_additional\_residuals
 
 : typing.Optional[typing.Tuple[torch.Tensor]] = None
 




 mid\_block\_additional\_residual
 
 : typing.Optional[torch.Tensor] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[UNet3DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet-motion#diffusers.models.unet_3d_condition.UNet3DConditionOutput) 
 or
 `tuple` 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) —
The noisy input tensor with the following shape
 `(batch, num_frames, channel, height, width` 
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
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [UNet3DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet-motion#diffusers.models.unet_3d_condition.UNet3DConditionOutput) 
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




 Returns
 




 export const metadata = 'undefined';
 

[UNet3DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet-motion#diffusers.models.unet_3d_condition.UNet3DConditionOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is True, an
 [UNet3DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet-motion#diffusers.models.unet_3d_condition.UNet3DConditionOutput) 
 is returned, otherwise
a
 `tuple` 
 is returned where the first element is the sample tensor.
 



 The
 [UNet3DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet3d-cond#diffusers.UNet3DConditionModel) 
 forward method.
 




#### 




 set\_attention\_slice




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L304)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L370)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L447)



 (
 

 )
 




 Disables custom attention processors and sets the default attention implementation.
 


## UNet3DConditionOutput




### 




 class
 

 diffusers.models.unet\_3d\_condition.
 

 UNet3DConditionOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_3d_condition.py#L50)



 (
 


 sample
 
 : FloatTensor
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_frames, num_channels, height, width)` 
 ) —
The hidden states output conditioned on
 `encoder_hidden_states` 
 input. Output of last layer of model.


 The output of
 [UNet3DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet3d-cond#diffusers.UNet3DConditionModel) 
.