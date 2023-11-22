# Stable Diffusion pipelines

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/stable_diffusion/overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview>



 Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from
 [CompVis](https://github.com/CompVis) 
 ,
 [Stability AI](https://stability.ai/) 
 and
 [LAION](https://laion.ai/) 
. Latent diffusion applies the diffusion process over a lower dimensional latent space to reduce memory and compute complexity. This specific type of diffusion model was proposed in
 [High-Resolution Image Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) 
 by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.
 



 Stable Diffusion is trained on 512x512 images from a subset of the LAION-5B dataset. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and can run on consumer GPUs.
 



 For more details about how Stable Diffusion works and how it differs from the base latent diffusion model, take a look at the Stability AI
 [announcement](https://stability.ai/blog/stable-diffusion-announcement) 
 and our own
 [blog post](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work) 
 for more technical details.
 



 You can find the original codebase for Stable Diffusion v1.0 at
 [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) 
 and Stable Diffusion v2.0 at
 [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion) 
 as well as their original scripts for various tasks. Additional official checkpoints for the different Stable Diffusion versions and tasks can be found on the
 [CompVis](https://huggingface.co/CompVis) 
 ,
 [Runway](https://huggingface.co/runwayml) 
 , and
 [Stability AI](https://huggingface.co/stabilityai) 
 Hub organizations. Explore these organizations to find the best checkpoint for your use-case!
 



 The table below summarizes the available Stable Diffusion pipelines, their supported tasks, and an interactive demo:
 


| 	 Pipeline	  | 	 Supported tasks	  | 	 Space	  |
| --- | --- | --- |
| [StableDiffusion](./text2img)  | 	 text-to-image	  |  |
| [StableDiffusionImg2Img](./img2img)  | 	 image-to-image	  |  |
| [StableDiffusionInpaint](./inpaint)  | 	 inpainting	  |  |
| [StableDiffusionDepth2Img](./depth2img)  | 	 depth-to-image	  |  |
| [StableDiffusionImageVariation](./image_variation)  | 	 image variation	  |  |
| [StableDiffusionPipelineSafe](./stable_diffusion_safe)  | 	 filtered text-to-image	  |  |
| [StableDiffusion2](./stable_diffusion_2)  | 	 text-to-image, inpainting, depth-to-image, super-resolution	  |  |
| [StableDiffusionXL](./stable_diffusion_xl)  | 	 text-to-image, image-to-image	  |  |
| [StableDiffusionLatentUpscale](./latent_upscale)  | 	 super-resolution	  |  |
| [StableDiffusionUpscale](./upscale)  | 	 super-resolution	  |
| [StableDiffusionLDM3D](./ldm3d_diffusion)  | 	 text-to-rgb, text-to-depth	  |  |


## Tips




 To help you get the most out of the Stable Diffusion pipelines, here are a few tips for improving performance and usability. These tips are applicable to all Stable Diffusion pipelines.
 


### 


 Explore tradeoff between speed and quality



[StableDiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) 
 uses the
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
 by default, but 🤗 Diffusers provides many other schedulers (some of which are faster or output better quality) that are compatible. For example, if you want to use the
 [EulerDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler) 
 instead of the default:
 



```
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# or
euler_scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=euler_scheduler)
```


### 


 Reuse pipeline components to save memory



 To save memory and use the same components across multiple pipelines, use the
 `.components` 
 method to avoid loading weights into RAM more than once.
 



```
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)

text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
inpaint = StableDiffusionInpaintPipeline(**text2img.components)

# now you can use text2img(...), img2img(...), inpaint(...) just like the call methods of each respective pipeline
```