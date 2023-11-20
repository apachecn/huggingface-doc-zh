# AutoencoderKL

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/autoencoderkl>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/autoencoderkl>



 The variational autoencoder (VAE) model with KL loss was introduced in
 [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114v11) 
 by Diederik P. Kingma and Max Welling. The model is used in 🤗 Diffusers to encode images into latents and to decode latent representations into images.
 



 The abstract from the paper is:
 



*How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions are two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.* 


## Loading from the original format




 By default the
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 should be loaded with
 [from\_pretrained()](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin.from_pretrained) 
 , but it can also be loaded
from the original format using
 `FromOriginalVAEMixin.from_single_file` 
 as follows:
 



```
from diffusers import AutoencoderKL

url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be local file
model = AutoencoderKL.from_single_file(url)
```


## AutoencoderKL




### 




 class
 

 diffusers.
 

 AutoencoderKL




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L49)



 (
 


 in\_channels
 
 : int = 3
 




 out\_channels
 
 : int = 3
 




 down\_block\_types
 
 : typing.Tuple[str] = ('DownEncoderBlock2D',)
 




 up\_block\_types
 
 : typing.Tuple[str] = ('UpDecoderBlock2D',)
 




 block\_out\_channels
 
 : typing.Tuple[int] = (64,)
 




 layers\_per\_block
 
 : int = 1
 




 act\_fn
 
 : str = 'silu'
 




 latent\_channels
 
 : int = 4
 




 norm\_num\_groups
 
 : int = 32
 




 sample\_size
 
 : int = 32
 




 scaling\_factor
 
 : float = 0.18215
 




 force\_upcast
 
 : float = True
 



 )
 


 Parameters
 




* **in\_channels** 
 (int,
 *optional* 
 , defaults to 3) — Number of channels in the input image.
* **out\_channels** 
 (int,
 *optional* 
 , defaults to 3) — Number of channels in the output.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("DownEncoderBlock2D",)` 
 ) —
Tuple of downsample block types.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("UpDecoderBlock2D",)` 
 ) —
Tuple of upsample block types.
* **block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(64,)` 
 ) —
Tuple of block output channels.
* **act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"silu"` 
 ) — The activation function to use.
* **latent\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) — Number of channels in the latent space.
* **sample\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `32` 
 ) — Sample input size.
* **scaling\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.18215) —
The component-wise standard deviation of the trained latent space computed using the first batch of the
training set. This is used to scale the latent space to have unit variance when training the diffusion
model. The latents are scaled with the formula
 `z = z * scaling_factor` 
 before being passed to the
diffusion model. When decoding, the latents are scaled back to the original scale with the formula:
 `z = 1 /scaling_factor * z` 
. For more details, refer to sections 4.3.2 and D.1 of the
 [High-Resolution Image
Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 
 paper.
* **force\_upcast** 
 (
 `bool` 
 ,
 *optional* 
 , default to
 `True` 
 ) —
If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
can be fine-tuned /trained to a lower range without loosing too much precision in which case
 `force_upcast` 
 can be set to
 `False` 
 - see:
 <https://huggingface.co/madebyollin/sdxl-vae-fp16-fix>


 A VAE model with KL loss for encoding images into latents and decoding latent representations into images.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



#### 




 disable\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L166)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_slicing` 
 was previously enabled, this method will go back to computing
decoding in one step.
 




#### 




 disable\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L152)



 (
 

 )
 




 Disable tiled VAE decoding. If
 `enable_tiling` 
 was previously enabled, this method will go back to computing
decoding in one step.
 




#### 




 enable\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L159)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L144)



 (
 


 use\_tiling
 
 : bool = True
 



 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L439)



 (
 


 sample
 
 : FloatTensor
 




 sample\_posterior
 
 : bool = False
 




 return\_dict
 
 : bool = True
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) — Input sample.
* **sample\_posterior** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to sample from the posterior.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `DecoderOutput` 
 instead of a plain tuple.



#### 




 set\_attn\_processor




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L199)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L236)



 (
 

 )
 




 Disables custom attention processors and sets the default attention implementation.
 




#### 




 tiled\_decode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L391)



 (
 


 z
 
 : FloatTensor
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[DecoderOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.vae.DecoderOutput) 
 or
 `tuple` 


 Parameters
 




* **z** 
 (
 `torch.FloatTensor` 
 ) — Input batch of latent vectors.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [DecoderOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.vae.DecoderOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[DecoderOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.vae.DecoderOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is True, a
 [DecoderOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.vae.DecoderOutput) 
 is returned, otherwise a plain
 `tuple` 
 is
returned.
 



 Decode a batch of images using a tiled decoder.
 




#### 




 tiled\_encode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L337)



 (
 


 x
 
 : FloatTensor
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[AutoencoderKLOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.autoencoder_kl.AutoencoderKLOutput) 
 or
 `tuple` 


 Parameters
 




* **x** 
 (
 `torch.FloatTensor` 
 ) — Input batch of images.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [AutoencoderKLOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.autoencoder_kl.AutoencoderKLOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[AutoencoderKLOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.autoencoder_kl.AutoencoderKLOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is True, a
 [AutoencoderKLOutput](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.models.autoencoder_kl.AutoencoderKLOutput) 
 is returned, otherwise a plain
 `tuple` 
 is returned.
 



 Encode a batch of images using a tiled encoder.
 



 When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
output, but they should be much less noticeable.
 


## AutoencoderKLOutput




### 




 class
 

 diffusers.models.autoencoder\_kl.
 

 AutoencoderKLOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_kl.py#L36)



 (
 


 latent\_dist
 
 : DiagonalGaussianDistribution
 



 )
 


 Parameters
 




* **latent\_dist** 
 (
 `DiagonalGaussianDistribution` 
 ) —
Encoded outputs of
 `Encoder` 
 represented as the mean and logvar of
 `DiagonalGaussianDistribution` 
.
 `DiagonalGaussianDistribution` 
 allows for sampling latents from the distribution.


 Output of AutoencoderKL encoding method.
 


## DecoderOutput




### 




 class
 

 diffusers.models.vae.
 

 DecoderOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/vae.py#L29)



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
The decoded output sample from the last layer of the model.


 Output of decoding method.
 


## FlaxAutoencoderKL




### 




 class
 

 diffusers.
 

 FlaxAutoencoderKL




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/vae_flax.py#L721)



 (
 


 in\_channels
 
 : int = 3
 




 out\_channels
 
 : int = 3
 




 down\_block\_types
 
 : typing.Tuple[str] = ('DownEncoderBlock2D',)
 




 up\_block\_types
 
 : typing.Tuple[str] = ('UpDecoderBlock2D',)
 




 block\_out\_channels
 
 : typing.Tuple[int] = (64,)
 




 layers\_per\_block
 
 : int = 1
 




 act\_fn
 
 : str = 'silu'
 




 latent\_channels
 
 : int = 4
 




 norm\_num\_groups
 
 : int = 32
 




 sample\_size
 
 : int = 32
 




 scaling\_factor
 
 : float = 0.18215
 




 dtype
 
 : dtype = <class 'jax.numpy.float32'>
 




 parent
 
 : typing.Union[typing.Type[flax.linen.module.Module], typing.Type[flax.core.scope.Scope], typing.Type[flax.linen.module.\_Sentinel], NoneType] = <flax.linen.module.\_Sentinel object at 0x7fad2e096f10>
 




 name
 
 : typing.Optional[str] = None
 



 )
 


 Parameters
 




* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 3) —
Number of channels in the input image.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 3) —
Number of channels in the output.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `(DownEncoderBlock2D)` 
 ) —
Tuple of downsample block types.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `(UpDecoderBlock2D)` 
 ) —
Tuple of upsample block types.
* **block\_out\_channels** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `(64,)` 
 ) —
Tuple of block output channels.
* **layers\_per\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `2` 
 ) —
Number of ResNet layer for each block.
* **act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `silu` 
 ) —
The activation function to use.
* **latent\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `4` 
 ) —
Number of channels in the latent space.
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `32` 
 ) —
The number of groups for normalization.
* **sample\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 32) —
Sample input size.
* **scaling\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.18215) —
The component-wise standard deviation of the trained latent space computed using the first batch of the
training set. This is used to scale the latent space to have unit variance when training the diffusion
model. The latents are scaled with the formula
 `z = z * scaling_factor` 
 before being passed to the
diffusion model. When decoding, the latents are scaled back to the original scale with the formula:
 `z = 1 /scaling_factor * z` 
. For more details, refer to sections 4.3.2 and D.1 of the
 [High-Resolution Image
Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 
 paper.
* **dtype** 
 (
 `jnp.dtype` 
 ,
 *optional* 
 , defaults to
 `jnp.float32` 
 ) —
The
 `dtype` 
 of the parameters.


 Flax implementation of a VAE model with KL loss for decoding latent representations.
 



 This model inherits from
 [FlaxModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.FlaxModelMixin) 
. Check the superclass documentation for it’s generic methods
implemented for all models (such as downloading or saving).
 



 This model is a Flax Linen
 [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) 
 subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matter related to its
general usage and behavior.
 



 Inherent JAX features such as the following are supported:
 


* [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
* [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
* [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
* [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)


## FlaxAutoencoderKLOutput




### 




 class
 

 diffusers.models.vae\_flax.
 

 FlaxAutoencoderKLOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/vae_flax.py#L48)



 (
 


 latent\_dist
 
 : FlaxDiagonalGaussianDistribution
 



 )
 


 Parameters
 




* **latent\_dist** 
 (
 `FlaxDiagonalGaussianDistribution` 
 ) —
Encoded outputs of
 `Encoder` 
 represented as the mean and logvar of
 `FlaxDiagonalGaussianDistribution` 
.
 `FlaxDiagonalGaussianDistribution` 
 allows for sampling latents from the distribution.


 Output of AutoencoderKL encoding method.
 



#### 




 replace




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/flax/struct.py#L111)



 (
 


 \*\*updates
 




 )
 




 “Returns a new object replacing the specified fields with new values.
 


## FlaxDecoderOutput




### 




 class
 

 diffusers.models.vae\_flax.
 

 FlaxDecoderOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/vae_flax.py#L33)



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
The decoded output sample from the last layer of the model.
* **dtype** 
 (
 `jnp.dtype` 
 ,
 *optional* 
 , defaults to
 `jnp.float32` 
 ) —
The
 `dtype` 
 of the parameters.


 Output of decoding method.
 



#### 




 replace




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/flax/struct.py#L111)



 (
 


 \*\*updates
 




 )
 




 “Returns a new object replacing the specified fields with new values.