# UNet1DModel

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/unet>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/unet>



 The
 [UNet](https://huggingface.co/papers/1505.04597) 
 model was originally introduced by Ronneberger et al for biomedical image segmentation, but it is also commonly used in 🤗 Diffusers because it outputs images that are the same size as the input. It is one of the most important components of a diffusion system because it facilitates the actual diffusion process. There are several variants of the UNet model in 🤗 Diffusers, depending on it’s number of dimensions and whether it is a conditional model or not. This is a 1D UNet model.
 



 The abstract from the paper is:
 



*There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at
 <http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net>
.* 


## UNet1DModel




### 




 class
 

 diffusers.
 

 UNet1DModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_1d.py#L41)



 (
 


 sample\_size
 
 : int = 65536
 




 sample\_rate
 
 : typing.Optional[int] = None
 




 in\_channels
 
 : int = 2
 




 out\_channels
 
 : int = 2
 




 extra\_in\_channels
 
 : int = 0
 




 time\_embedding\_type
 
 : str = 'fourier'
 




 flip\_sin\_to\_cos
 
 : bool = True
 




 use\_timestep\_embedding
 
 : bool = False
 




 freq\_shift
 
 : float = 0.0
 




 down\_block\_types
 
 : typing.Tuple[str] = ('DownBlock1DNoSkip', 'DownBlock1D', 'AttnDownBlock1D')
 




 up\_block\_types
 
 : typing.Tuple[str] = ('AttnUpBlock1D', 'UpBlock1D', 'UpBlock1DNoSkip')
 




 mid\_block\_type
 
 : typing.Tuple[str] = 'UNetMidBlock1D'
 




 out\_block\_type
 
 : str = None
 




 block\_out\_channels
 
 : typing.Tuple[int] = (32, 32, 64)
 




 act\_fn
 
 : str = None
 




 norm\_num\_groups
 
 : int = 8
 




 layers\_per\_block
 
 : int = 1
 




 downsample\_each\_block
 
 : bool = False
 



 )
 


 Parameters
 




* **sample\_size** 
 (
 `int` 
 ,
 *optional* 
 ) — Default length of sample. Should be adaptable at runtime.
* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 2) — Number of channels in the input sample.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 2) — Number of channels in the output.
* **extra\_in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) —
Number of additional channels to be added to the input of the first down block. Useful for cases where the
input data has more channels than what the model was initially designed for.
* **time\_embedding\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"fourier"` 
 ) — Type of time embedding to use.
* **freq\_shift** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) — Frequency shift for Fourier time embedding.
* **flip\_sin\_to\_cos** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to flip sin to cos for Fourier time embedding.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")` 
 ) —
Tuple of downsample block types.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")` 
 ) —
Tuple of upsample block types.
* **block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(32, 32, 64)` 
 ) —
Tuple of block output channels.
* **mid\_block\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"UNetMidBlock1D"` 
 ) — Block type for middle of UNet.
* **out\_block\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) — Optional output processing block of UNet.
* **act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) — Optional activation function in UNet blocks.
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 8) — The number of groups for normalization.
* **layers\_per\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) — The number of layers per block.
* **downsample\_each\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Experimental feature for using a UNet without upsampling.


 A 1D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_1d.py#L195)



 (
 


 sample
 
 : FloatTensor
 




 timestep
 
 : typing.Union[torch.Tensor, float, int]
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[UNet1DOutput](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.models.unet_1d.UNet1DOutput) 
 or
 `tuple` 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) —
The noisy input tensor with the following shape
 `(batch_size, num_channels, sample_size)` 
.
* **timestep** 
 (
 `torch.FloatTensor` 
 or
 `float` 
 or
 `int` 
 ) — The number of timesteps to denoise an input.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [UNet1DOutput](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.models.unet_1d.UNet1DOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[UNet1DOutput](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.models.unet_1d.UNet1DOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is True, an
 [UNet1DOutput](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.models.unet_1d.UNet1DOutput) 
 is returned, otherwise a
 `tuple` 
 is
returned where the first element is the sample tensor.
 



 The
 [UNet1DModel](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.UNet1DModel) 
 forward method.
 


## UNet1DOutput




### 




 class
 

 diffusers.models.unet\_1d.
 

 UNet1DOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/unet_1d.py#L29)



 (
 


 sample
 
 : FloatTensor
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, sample_size)` 
 ) —
The hidden states output from the last layer of the model.


 The output of
 [UNet1DModel](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.UNet1DModel) 
.