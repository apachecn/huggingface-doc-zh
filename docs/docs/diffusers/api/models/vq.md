# VQModel

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/vq>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/vq>



 The VQ-VAE model was introduced in
 [Neural Discrete Representation Learning](https://huggingface.co/papers/1711.00937) 
 by Aaron van den Oord, Oriol Vinyals and Koray Kavukcuoglu. The model is used in 🤗 Diffusers to decode latent representations into images. Unlike
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 , the
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 works in a quantized latent space.
 



 The abstract from the paper is:
 



*Learning useful representations without supervision remains a key challenge in machine learning. In this paper, we propose a simple yet powerful generative model that learns such discrete representations. Our model, the Vector Quantised-Variational AutoEncoder (VQ-VAE), differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static. In order to learn a discrete latent representation, we incorporate ideas from vector quantisation (VQ). Using the VQ method allows the model to circumvent issues of “posterior collapse” — where the latents are ignored when they are paired with a powerful autoregressive decoder — typically observed in the VAE framework. Pairing these representations with an autoregressive prior, the model can generate high quality images, videos, and speech as well as doing high quality speaker conversion and unsupervised learning of phonemes, providing further evidence of the utility of the learnt representations.* 


## VQModel




### 




 class
 

 diffusers.
 

 VQModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/vq_model.py#L40)



 (
 


 in\_channels
 
 : int = 3
 




 out\_channels
 
 : int = 3
 




 down\_block\_types
 
 : typing.Tuple[str,...] = ('DownEncoderBlock2D',)
 




 up\_block\_types
 
 : typing.Tuple[str,...] = ('UpDecoderBlock2D',)
 




 block\_out\_channels
 
 : typing.Tuple[int,...] = (64,)
 




 layers\_per\_block
 
 : int = 1
 




 act\_fn
 
 : str = 'silu'
 




 latent\_channels
 
 : int = 3
 




 sample\_size
 
 : int = 32
 




 num\_vq\_embeddings
 
 : int = 256
 




 norm\_num\_groups
 
 : int = 32
 




 vq\_embed\_dim
 
 : typing.Optional[int] = None
 




 scaling\_factor
 
 : float = 0.18215
 




 norm\_type
 
 : str = 'group'
 



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
* **layers\_per\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `1` 
 ) — Number of layers per block.
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
 , defaults to
 `3` 
 ) — Number of channels in the latent space.
* **sample\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `32` 
 ) — Sample input size.
* **num\_vq\_embeddings** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `256` 
 ) — Number of codebook vectors in the VQ-VAE.
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `32` 
 ) — Number of groups for normalization layers.
* **vq\_embed\_dim** 
 (
 `int` 
 ,
 *optional* 
 ) — Hidden dim of codebook vectors in the VQ-VAE.
* **scaling\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to
 `0.18215` 
 ) —
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
* **norm\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"group"` 
 ) —
Type of normalization layer to use. Can be one of
 `"group"` 
 or
 `"spatial"` 
.


 A VQ-VAE model for decoding latent representations.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/vq_model.py#L151)



 (
 


 sample
 
 : FloatTensor
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[VQEncoderOutput](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.models.vq_model.VQEncoderOutput) 
 or
 `tuple` 


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
 [models.vq\_model.VQEncoderOutput](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.models.vq_model.VQEncoderOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[VQEncoderOutput](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.models.vq_model.VQEncoderOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is True, a
 [VQEncoderOutput](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.models.vq_model.VQEncoderOutput) 
 is returned, otherwise a plain
 `tuple` 
 is returned.
 



 The
 [VQModel](/docs/diffusers/v0.23.0/en/api/models/vq#diffusers.VQModel) 
 forward method.
 


## VQEncoderOutput




### 




 class
 

 diffusers.models.vq\_model.
 

 VQEncoderOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/vq_model.py#L28)



 (
 


 latents
 
 : FloatTensor
 



 )
 


 Parameters
 




* **latents** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 ) —
The encoded output sample from the last layer of the model.


 Output of VQModel encoding method.