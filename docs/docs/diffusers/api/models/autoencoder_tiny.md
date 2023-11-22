# Tiny AutoEncoder

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/autoencoder_tiny>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/autoencoder_tiny>



 Tiny AutoEncoder for Stable Diffusion (TAESD) was introduced in
 [madebyollin/taesd](https://github.com/madebyollin/taesd) 
 by Ollin Boer Bohan. It is a tiny distilled version of Stable Diffusion’s VAE that can quickly decode the latents in a
 [StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) 
 or
 [StableDiffusionXLPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline) 
 almost instantly.
 



 To use with Stable Diffusion v-2.1:
 



```
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "slice of delicious New York-style berry cheesecake"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("cheesecake.png")
```



 To use with Stable Diffusion XL 1.0
 



```
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "slice of delicious New York-style berry cheesecake"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("cheesecake\_sdxl.png")
```


## AutoencoderTiny




### 




 class
 

 diffusers.
 

 AutoencoderTiny




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L41)



 (
 


 in\_channels
 
 = 3
 




 out\_channels
 
 = 3
 




 encoder\_block\_out\_channels
 
 : typing.Tuple[int] = (64, 64, 64, 64)
 




 decoder\_block\_out\_channels
 
 : typing.Tuple[int] = (64, 64, 64, 64)
 




 act\_fn
 
 : str = 'relu'
 




 latent\_channels
 
 : int = 4
 




 upsampling\_scaling\_factor
 
 : int = 2
 




 num\_encoder\_blocks
 
 : typing.Tuple[int] = (1, 3, 3, 3)
 




 num\_decoder\_blocks
 
 : typing.Tuple[int] = (3, 3, 3, 1)
 




 latent\_magnitude
 
 : int = 3
 




 latent\_shift
 
 : float = 0.5
 




 force\_upcast
 
 : float = False
 




 scaling\_factor
 
 : float = 1.0
 



 )
 


 Parameters
 




* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 3) — Number of channels in the input image.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 3) — Number of channels in the output.
* **encoder\_block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(64, 64, 64, 64)` 
 ) —
Tuple of integers representing the number of output channels for each encoder block. The length of the
tuple should be equal to the number of encoder blocks.
* **decoder\_block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(64, 64, 64, 64)` 
 ) —
Tuple of integers representing the number of output channels for each decoder block. The length of the
tuple should be equal to the number of decoder blocks.
* **act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"relu"` 
 ) —
Activation function to be used throughout the model.
* **latent\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) —
Number of channels in the latent representation. The latent space acts as a compressed representation of
the input image.
* **upsampling\_scaling\_factor** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 2) —
Scaling factor for upsampling in the decoder. It determines the size of the output image during the
upsampling process.
* **num\_encoder\_blocks** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(1, 3, 3, 3)` 
 ) —
Tuple of integers representing the number of encoder blocks at each stage of the encoding process. The
length of the tuple should be equal to the number of stages in the encoder. Each stage has a different
number of encoder blocks.
* **num\_decoder\_blocks** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(3, 3, 3, 1)` 
 ) —
Tuple of integers representing the number of decoder blocks at each stage of the decoding process. The
length of the tuple should be equal to the number of stages in the decoder. Each stage has a different
number of decoder blocks.
* **latent\_magnitude** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 3.0) —
Magnitude of the latent representation. This parameter scales the latent representation values to control
the extent of information preservation.
* **latent\_shift** 
 (float,
 *optional* 
 , defaults to 0.5) —
Shift applied to the latent representation. This parameter controls the center of the latent space.
* **scaling\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
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
 paper. For this Autoencoder,
however, no such scaling factor was used, hence the value of 1.0 as the default.
* **force\_upcast** 
 (
 `bool` 
 ,
 *optional* 
 , default to
 `False` 
 ) —
If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
can be fine-tuned /trained to a lower range without losing too much precision, in which case
 `force_upcast` 
 can be set to
 `False` 
 (see this fp16-friendly
 [AutoEncoder](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) 
 ).


 A tiny distilled VAE model for encoding images into latents and decoding latent representations into images.
 



[AutoencoderTiny](/docs/diffusers/v0.23.0/en/api/models/autoencoder_tiny#diffusers.AutoencoderTiny) 
 is a wrapper around the original implementation of
 `TAESD` 
.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for its generic methods implemented for
all models (such as downloading or saving).
 



#### 




 disable\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L169)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L184)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L162)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L176)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L324)



 (
 


 sample
 
 : FloatTensor
 




 return\_dict
 
 : bool = True
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) — Input sample.
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




 scale\_latents




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L154)



 (
 


 x
 




 )
 




 raw latents -> [0, 1]
 




#### 




 unscale\_latents




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L158)



 (
 


 x
 




 )
 




 [0, 1] -> raw latents
 


## AutoencoderTinyOutput




### 




 class
 

 diffusers.models.autoencoder\_tiny.
 

 AutoencoderTinyOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_tiny.py#L29)



 (
 


 latents
 
 : Tensor
 



 )
 


 Parameters
 




* **latents** 
 (
 `torch.Tensor` 
 ) — Encoded outputs of the
 `Encoder` 
.


 Output of AutoencoderTiny encoding method.