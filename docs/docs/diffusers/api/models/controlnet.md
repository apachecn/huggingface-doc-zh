# ControlNet

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/controlnet>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/controlnet>



 The ControlNet model was introduced in
 [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) 
 by Lvmin Zhang and Maneesh Agrawala. It provides a greater degree of control over text-to-image generation by conditioning the model on additional inputs such as edge maps, depth maps, segmentation maps, and keypoints for pose detection.
 



 The abstract from the paper is:
 



*We present a neural network structure, ControlNet, to control pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small (< 50k). Moreover, training a ControlNet is as fast as fine-tuning a diffusion model, and the model can be trained on a personal devices. Alternatively, if powerful computation clusters are available, the model can scale to large amounts (millions to billions) of data. We report that large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like edge maps, segmentation maps, keypoints, etc. This may enrich the methods to control large diffusion models and further facilitate related applications.* 


## Loading from the original format




 By default the
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
 should be loaded with
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained) 
 , but it can also be loaded
from the original format using
 `FromOriginalControlnetMixin.from_single_file` 
 as follows:
 



```
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control\_v11p\_sd15\_canny.pth"  # can also be a local path
controlnet = ControlNetModel.from_single_file(url)

url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
pipe = StableDiffusionControlNetPipeline.from_single_file(url, controlnet=controlnet)
```


## ControlNetModel




### 




 class
 

 diffusers.
 

 ControlNetModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet.py#L110)



 (
 


 in\_channels
 
 : int = 4
 




 conditioning\_channels
 
 : int = 3
 




 flip\_sin\_to\_cos
 
 : bool = True
 




 freq\_shift
 
 : int = 0
 




 down\_block\_types
 
 : typing.Tuple[str] = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
 




 only\_cross\_attention
 
 : typing.Union[bool, typing.Tuple[bool]] = False
 




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
 
 : int = 1280
 




 transformer\_layers\_per\_block
 
 : typing.Union[int, typing.Tuple[int]] = 1
 




 encoder\_hid\_dim
 
 : typing.Optional[int] = None
 




 encoder\_hid\_dim\_type
 
 : typing.Optional[str] = None
 




 attention\_head\_dim
 
 : typing.Union[int, typing.Tuple[int]] = 8
 




 num\_attention\_heads
 
 : typing.Union[int, typing.Tuple[int], NoneType] = None
 




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
 




 projection\_class\_embeddings\_input\_dim
 
 : typing.Optional[int] = None
 




 controlnet\_conditioning\_channel\_order
 
 : str = 'rgb'
 




 conditioning\_embedding\_out\_channels
 
 : typing.Optional[typing.Tuple[int]] = (16, 32, 96, 256)
 




 global\_pool\_conditions
 
 : bool = False
 




 addition\_embed\_type\_num\_heads
 
 = 64
 



 )
 


 Parameters
 




* **in\_channels** 
 (
 `int` 
 , defaults to 4) —
The number of channels in the input sample.
* **flip\_sin\_to\_cos** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to flip the sin to cos in the time embedding.
* **freq\_shift** 
 (
 `int` 
 , defaults to 0) —
The frequency shift to apply to the time embedding.
* **down\_block\_types** 
 (
 `tuple[str]` 
 , defaults to
 `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")` 
 ) —
The tuple of downsample blocks to use.
* **only\_cross\_attention** 
 (
 `Union[bool, Tuple[bool]]` 
 , defaults to
 `False` 
 ) —
* **block\_out\_channels** 
 (
 `tuple[int]` 
 , defaults to
 `(320, 640, 1280, 1280)` 
 ) —
The tuple of output channels for each block.
* **layers\_per\_block** 
 (
 `int` 
 , defaults to 2) —
The number of layers per block.
* **downsample\_padding** 
 (
 `int` 
 , defaults to 1) —
The padding to use for the downsampling convolution.
* **mid\_block\_scale\_factor** 
 (
 `float` 
 , defaults to 1) —
The scale factor to use for the mid block.
* **act\_fn** 
 (
 `str` 
 , defaults to “silu”) —
The activation function to use.
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 32) —
The number of groups to use for the normalization. If None, normalization and activation layers is skipped
in post-processing.
* **norm\_eps** 
 (
 `float` 
 , defaults to 1e-5) —
The epsilon to use for the normalization.
* **cross\_attention\_dim** 
 (
 `int` 
 , defaults to 1280) —
The dimension of the cross attention features.
* **transformer\_layers\_per\_block** 
 (
 `int` 
 or
 `Tuple[int]` 
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
* **encoder\_hid\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to None) —
If
 `encoder_hid_dim_type` 
 is defined,
 `encoder_hidden_states` 
 will be projected from
 `encoder_hid_dim` 
 dimension to
 `cross_attention_dim` 
.
* **encoder\_hid\_dim\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
If given, the
 `encoder_hidden_states` 
 and potentially other embeddings are down-projected to text
embeddings of dimension
 `cross_attention` 
 according to
 `encoder_hid_dim_type` 
.
* **attention\_head\_dim** 
 (
 `Union[int, Tuple[int]]` 
 , defaults to 8) —
The dimension of the attention heads.
* **use\_linear\_projection** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
* **class\_embed\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
 `"timestep"` 
 ,
 `"identity"` 
 ,
 `"projection"` 
 , or
 `"simple_projection"` 
.
* **addition\_embed\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Configures an optional embedding which will be summed with the time embeddings. Choose from
 `None` 
 or
“text”. “text” will use the
 `TextTimeEmbedding` 
 layer.
* **num\_class\_embeds** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) —
Input dimension of the learnable embedding matrix to be projected to
 `time_embed_dim` 
 , when performing
class conditioning with
 `class_embed_type` 
 equal to
 `None` 
.
* **upcast\_attention** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
* **resnet\_time\_scale\_shift** 
 (
 `str` 
 , defaults to
 `"default"` 
 ) —
Time scale shift config for ResNet blocks (see
 `ResnetBlock2D` 
 ). Choose from
 `default` 
 or
 `scale_shift` 
.
* **projection\_class\_embeddings\_input\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The dimension of the
 `class_labels` 
 input when
 `class_embed_type="projection"` 
. Required when
 `class_embed_type="projection"` 
.
* **controlnet\_conditioning\_channel\_order** 
 (
 `str` 
 , defaults to
 `"rgb"` 
 ) —
The channel order of conditional image. Will convert to
 `rgb` 
 if it’s
 `bgr` 
.
* **conditioning\_embedding\_out\_channels** 
 (
 `tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(16, 32, 96, 256)` 
 ) —
The tuple of output channel for each block in the
 `conditioning_embedding` 
 layer.
* **global\_pool\_conditions** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —


 A ControlNet model.
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet.py#L642)



 (
 


 sample
 
 : FloatTensor
 




 timestep
 
 : typing.Union[torch.Tensor, float, int]
 




 encoder\_hidden\_states
 
 : Tensor
 




 controlnet\_cond
 
 : FloatTensor
 




 conditioning\_scale
 
 : float = 1.0
 




 class\_labels
 
 : typing.Optional[torch.Tensor] = None
 




 timestep\_cond
 
 : typing.Optional[torch.Tensor] = None
 




 attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 added\_cond\_kwargs
 
 : typing.Union[typing.Dict[str, torch.Tensor], NoneType] = None
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 guess\_mode
 
 : bool = False
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[ControlNetOutput](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.models.controlnet.ControlNetOutput) 
**or** 
`tuple` 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) —
The noisy input tensor.
* **timestep** 
 (
 `Union[torch.Tensor, float, int]` 
 ) —
The number of timesteps to denoise an input.
* **encoder\_hidden\_states** 
 (
 `torch.Tensor` 
 ) —
The encoder hidden states.
* **controlnet\_cond** 
 (
 `torch.FloatTensor` 
 ) —
The conditional input tensor of shape
 `(batch_size, sequence_length, hidden_size)` 
.
* **conditioning\_scale** 
 (
 `float` 
 , defaults to
 `1.0` 
 ) —
The scale factor for ControlNet outputs.
* **class\_labels** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
* **timestep\_cond** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
timestep\_embedding passed through the
 `self.time_embedding` 
 layer to obtain the final timestep
embeddings.
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
* **added\_cond\_kwargs** 
 (
 `dict` 
 ) —
Additional conditions for the Stable Diffusion XL UNet.
* **cross\_attention\_kwargs** 
 (
 `dict[str]` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttnProcessor` 
.
* **guess\_mode** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
you remove all prompts. A
 `guidance_scale` 
 between 3.0 and 5.0 is recommended.
* **return\_dict** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [ControlNetOutput](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.models.controlnet.ControlNetOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[ControlNetOutput](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.models.controlnet.ControlNetOutput) 
**or** 
`tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 , a
 [ControlNetOutput](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.models.controlnet.ControlNetOutput) 
 is returned, otherwise a tuple is
returned where the first element is the sample tensor.
 



 The
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
 forward method.
 




#### 




 from\_unet




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet.py#L424)



 (
 


 unet
 
 : UNet2DConditionModel
 




 controlnet\_conditioning\_channel\_order
 
 : str = 'rgb'
 




 conditioning\_embedding\_out\_channels
 
 : typing.Optional[typing.Tuple[int]] = (16, 32, 96, 256)
 




 load\_weights\_from\_unet
 
 : bool = True
 



 )
 


 Parameters
 




* **unet** 
 (
 `UNet2DConditionModel` 
 ) —
The UNet model weights to copy to the
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
. All configuration options are also copied
where applicable.


 Instantiate a
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
 from
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
.
 




#### 




 set\_attention\_slice




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet.py#L573)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet.py#L520)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet.py#L557)



 (
 

 )
 




 Disables custom attention processors and sets the default attention implementation.
 


## ControlNetOutput




### 




 class
 

 diffusers.models.controlnet.
 

 ControlNetOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet.py#L46)



 (
 


 down\_block\_res\_samples
 
 : typing.Tuple[torch.Tensor]
 




 mid\_block\_res\_sample
 
 : Tensor
 



 )
 


 Parameters
 




* **down\_block\_res\_samples** 
 (
 `tuple[torch.Tensor]` 
 ) —
A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
be of shape
 `(batch_size, channel * resolution, height //resolution, width //resolution)` 
. Output can be
used to condition the original UNet’s downsampling activations.
* **mid\_down\_block\_re\_sample** 
 (
 `torch.Tensor` 
 ) —
The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
 `(batch_size, channel * lowest_resolution, height //lowest_resolution, width //lowest_resolution)` 
.
Output can be used to condition the original UNet’s middle block activation.


 The output of
 [ControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.ControlNetModel) 
.
 


## FlaxControlNetModel




### 




 class
 

 diffusers.
 

 FlaxControlNetModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet_flax.py#L104)



 (
 


 sample\_size
 
 : int = 32
 




 in\_channels
 
 : int = 4
 




 down\_block\_types
 
 : typing.Tuple[str] = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
 




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
 




 controlnet\_conditioning\_channel\_order
 
 : str = 'rgb'
 




 conditioning\_embedding\_out\_channels
 
 : typing.Tuple[int] = (16, 32, 96, 256)
 




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
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")` 
 ) —
The tuple of downsample blocks to use.
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
* **controlnet\_conditioning\_channel\_order** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `rgb` 
 ) —
The channel order of conditional image. Will convert to
 `rgb` 
 if it’s
 `bgr` 
.
* **conditioning\_embedding\_out\_channels** 
 (
 `tuple` 
 ,
 *optional* 
 , defaults to
 `(16, 32, 96, 256)` 
 ) —
The tuple of output channel for each block in the
 `conditioning_embedding` 
 layer.


 A ControlNet model.
 



 This model inherits from
 [FlaxModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin) 
. Check the superclass documentation for it’s generic methods
implemented for all models (such as downloading or saving).
 



 This model is also a Flax Linen
 [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
 subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
general usage and behavior.
 



 Inherent JAX features such as the following are supported:
 


* [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
* [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
* [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
* [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)


## FlaxControlNetOutput




### 




 class
 

 diffusers.models.controlnet\_flax.
 

 FlaxControlNetOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/controlnet_flax.py#L34)



 (
 


 down\_block\_res\_samples
 
 : Array
 




 mid\_block\_res\_sample
 
 : Array
 



 )
 


 Parameters
 




* **down\_block\_res\_samples** 
 (
 `jnp.ndarray` 
 ) —
* **mid\_block\_res\_sample** 
 (
 `jnp.ndarray` 
 ) —


 The output of
 [FlaxControlNetModel](/docs/diffusers/v0.23.0/en/api/models/controlnet#diffusers.FlaxControlNetModel) 
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