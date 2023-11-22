# AsymmetricAutoencoderKL

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/models/asymmetricautoencoderkl>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/models/asymmetricautoencoderkl>



 Improved larger variational autoencoder (VAE) model with KL loss for inpainting task:
 [Designing a Better Asymmetric VQGAN for StableDiffusion](https://arxiv.org/abs/2306.04632) 
 by Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, Gang Hua.
 



 The abstract from the paper is:
 



*StableDiffusion is a revolutionary text-to-image generator that is causing a stir in the world of image generation and editing. Unlike traditional methods that learn a diffusion model in pixel space, StableDiffusion learns a diffusion model in the latent space via a VQGAN, ensuring both efficiency and quality. It not only supports image generation tasks, but also enables image editing for real images, such as image inpainting and local editing. However, we have observed that the vanilla VQGAN used in StableDiffusion leads to significant information loss, causing distortion artifacts even in non-edited image regions. To this end, we propose a new asymmetric VQGAN with two simple designs. Firstly, in addition to the input from the encoder, the decoder contains a conditional branch that incorporates information from task-specific priors, such as the unmasked image region in inpainting. Secondly, the decoder is much heavier than the encoder, allowing for more detailed recovery while only slightly increasing the total inference cost. The training cost of our asymmetric VQGAN is cheap, and we only need to retrain a new asymmetric decoder while keeping the vanilla VQGAN encoder and StableDiffusion unchanged. Our asymmetric VQGAN can be widely used in StableDiffusion-based inpainting and local editing methods. Extensive experiments demonstrate that it can significantly improve the inpainting and editing performance, while maintaining the original text-to-image capability. The code is available at
 <https://github.com/buxiangzhiren/Asymmetric_VQGAN>*




 Evaluation results can be found in section 4.1 of the original paper.
 


## Available checkpoints



* <https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-1-5>
* <https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-2>


## Example Usage




```
from io import BytesIO
from PIL import Image
import requests
from diffusers import AsymmetricAutoencoderKL, StableDiffusionInpaintPipeline


def download\_image(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


prompt = "a photo of a person"
img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba\_hq\_256.png"
mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask\_256.png"

image = download_image(img_url).resize((256, 256))
mask_image = download_image(mask_url).resize((256, 256))

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipe.vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
pipe.to("cuda")

image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("image.jpeg")
```


## AsymmetricAutoencoderKL




### 




 class
 

 diffusers.
 

 AsymmetricAutoencoderKL




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_asym_kl.py#L26)



 (
 


 in\_channels
 
 : int = 3
 




 out\_channels
 
 : int = 3
 




 down\_block\_types
 
 : typing.Tuple[str] = ('DownEncoderBlock2D',)
 




 down\_block\_out\_channels
 
 : typing.Tuple[int] = (64,)
 




 layers\_per\_down\_block
 
 : int = 1
 




 up\_block\_types
 
 : typing.Tuple[str] = ('UpDecoderBlock2D',)
 




 up\_block\_out\_channels
 
 : typing.Tuple[int] = (64,)
 




 layers\_per\_up\_block
 
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
* **down\_block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(64,)` 
 ) —
Tuple of down block output channels.
* **layers\_per\_down\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `1` 
 ) —
Number layers for down block.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("UpDecoderBlock2D",)` 
 ) —
Tuple of upsample block types.
* **up\_block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(64,)` 
 ) —
Tuple of up block output channels.
* **layers\_per\_up\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `1` 
 ) —
Number layers for up block.
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
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `32` 
 ) —
Number of groups to use for the first normalization layer in ResNet blocks.
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


 Designing a Better Asymmetric VQGAN for StableDiffusion
 <https://arxiv.org/abs/2306.04632>
. A VAE model with KL loss
for encoding images into latents and decoding latent representations into images.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/models/autoencoder_asym_kl.py#L153)



 (
 


 sample
 
 : FloatTensor
 




 mask
 
 : typing.Optional[torch.FloatTensor] = None
 




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
* **mask** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) — Optional inpainting mask.
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