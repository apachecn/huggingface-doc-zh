# UNet2DModel

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/unet2d>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/unet2d>



 The
 [UNet](https://huggingface.co/papers/1505.04597) 
 model was originally introduced by Ronneberger et al for biomedical image segmentation, but it is also commonly used in 🤗 Diffusers because it outputs images that are the same size as the input. It is one of the most important components of a diffusion system because it facilitates the actual diffusion process. There are several variants of the UNet model in 🤗 Diffusers, depending on it’s number of dimensions and whether it is a conditional model or not. This is a 2D UNet model.
 



 The abstract from the paper is:
 



*There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at
 <http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net>
.* 


## UNet2DModel




### 




 class
 

 diffusers.
 

 UNet2DModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d.py#L40)



 (
 


 sample\_size
 
 : typing.Union[int, typing.Tuple[int, int], NoneType] = None
 




 in\_channels
 
 : int = 3
 




 out\_channels
 
 : int = 3
 




 center\_input\_sample
 
 : bool = False
 




 time\_embedding\_type
 
 : str = 'positional'
 




 freq\_shift
 
 : int = 0
 




 flip\_sin\_to\_cos
 
 : bool = True
 




 down\_block\_types
 
 : typing.Tuple[str] = ('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D')
 




 up\_block\_types
 
 : typing.Tuple[str] = ('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D')
 




 block\_out\_channels
 
 : typing.Tuple[int] = (224, 448, 672, 896)
 




 layers\_per\_block
 
 : int = 2
 




 mid\_block\_scale\_factor
 
 : float = 1
 




 downsample\_padding
 
 : int = 1
 




 downsample\_type
 
 : str = 'conv'
 




 upsample\_type
 
 : str = 'conv'
 




 dropout
 
 : float = 0.0
 




 act\_fn
 
 : str = 'silu'
 




 attention\_head\_dim
 
 : typing.Optional[int] = 8
 




 norm\_num\_groups
 
 : int = 32
 




 attn\_norm\_num\_groups
 
 : typing.Optional[int] = None
 




 norm\_eps
 
 : float = 1e-05
 




 resnet\_time\_scale\_shift
 
 : str = 'default'
 




 add\_attention
 
 : bool = True
 




 class\_embed\_type
 
 : typing.Optional[str] = None
 




 num\_class\_embeds
 
 : typing.Optional[int] = None
 




 num\_train\_timesteps
 
 : typing.Optional[int] = None
 



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
Height and width of input/output sample. Dimensions must be a multiple of
 `2 ** (len(block_out_channels) - 1)` 
.
* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 3) — Number of channels in the input sample.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 3) — Number of channels in the output.
* **center\_input\_sample** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) — Whether to center the input sample.
* **time\_embedding\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"positional"` 
 ) — Type of time embedding to use.
* **freq\_shift** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) — Frequency shift for Fourier time embedding.
* **flip\_sin\_to\_cos** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether to flip sin to cos for Fourier time embedding.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")` 
 ) —
Tuple of downsample block types.
* **mid\_block\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"UNetMidBlock2D"` 
 ) —
Block type for middle of UNet, it can be either
 `UNetMidBlock2D` 
 or
 `UnCLIPUNetMidBlock2D` 
.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")` 
 ) —
Tuple of upsample block types.
* **block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(224, 448, 672, 896)` 
 ) —
Tuple of block output channels.
* **layers\_per\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `2` 
 ) — The number of layers per block.
* **mid\_block\_scale\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to
 `1` 
 ) — The scale factor for the mid block.
* **downsample\_padding** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `1` 
 ) — The padding for the downsample convolution.
* **downsample\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `conv` 
 ) —
The downsample type for downsampling layers. Choose between “conv” and “resnet”
* **upsample\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `conv` 
 ) —
The upsample type for upsampling layers. Choose between “conv” and “resnet”
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
* **attention\_head\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `8` 
 ) — The attention head dimension.
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `32` 
 ) — The number of groups for normalization.
* **attn\_norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
If set to an integer, a group norm layer will be created in the mid block’s
 `Attention` 
 layer with the
given number of groups. If left as
 `None` 
 , the group norm layer will only be created if
 `resnet_time_scale_shift` 
 is set to
 `default` 
 , and if created will have
 `norm_num_groups` 
 groups.
* **norm\_eps** 
 (
 `float` 
 ,
 *optional* 
 , defaults to
 `1e-5` 
 ) — The epsilon for normalization.
* **resnet\_time\_scale\_shift** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"default"` 
 ) — Time scale shift config
for ResNet blocks (see
 `ResnetBlock2D` 
 ). Choose from
 `default` 
 or
 `scale_shift` 
.
* **class\_embed\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The type of class embedding to use which is ultimately summed with the time embeddings. Choose from
 `None` 
 ,
 `"timestep"` 
 , or
 `"identity"` 
.
* **num\_class\_embeds** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Input dimension of the learnable embedding matrix to be projected to
 `time_embed_dim` 
 when performing class
conditioning with
 `class_embed_type` 
 equal to
 `None` 
.


 A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d.py#L243)



 (
 


 sample
 
 : FloatTensor
 




 timestep
 
 : typing.Union[torch.Tensor, float, int]
 




 class\_labels
 
 : typing.Optional[torch.Tensor] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[UNet2DOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.models.unet_2d.UNet2DOutput) 
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
* **class\_labels** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [UNet2DOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.models.unet_2d.UNet2DOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[UNet2DOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.models.unet_2d.UNet2DOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is True, an
 [UNet2DOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.models.unet_2d.UNet2DOutput) 
 is returned, otherwise a
 `tuple` 
 is
returned where the first element is the sample tensor.
 



 The
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
 forward method.
 


## UNet2DOutput




### 




 class
 

 diffusers.models.unet\_2d.
 

 UNet2DOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_2d.py#L28)



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
 ) —
The hidden states output from the last layer of the model.


 The output of
 [UNet2DModel](/docs/diffusers/v0.23.0/en/api/models/unet2d#diffusers.UNet2DModel) 
.