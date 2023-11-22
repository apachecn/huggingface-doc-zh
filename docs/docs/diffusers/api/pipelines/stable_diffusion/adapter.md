# Text-to-Image Generation with Adapter Conditioning

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/stable_diffusion/adapter>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/adapter>


## Overview




[T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453) 
 by Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie.
 



 Using the pretrained models we can provide control images (for example, a depth map) to control Stable Diffusion text-to-image generation so that it follows the structure of the depth image and fills in the details.
 



 The abstract of the paper is the following:
 



*The incredible generative ability of large-scale text-to-image (T2I) models has demonstrated strong power of learning complex structures and meaningful semantics. However, relying solely on text prompts cannot fully take advantage of the knowledge learned by the model, especially when flexible and accurate structure control is needed. In this paper, we aim to “dig out” the capabilities that T2I models have implicitly learned, and then explicitly use them to control the generation more granularly. Specifically, we propose to learn simple and small T2I-Adapters to align internal knowledge in T2I models with external control signals, while freezing the original large T2I models. In this way, we can train various adapters according to different conditions, and achieve rich control and editing effects. Further, the proposed T2I-Adapters have attractive properties of practical value, such as composability and generalization ability. Extensive experiments demonstrate that our T2I-Adapter has promising generation quality and a wide range of applications.* 




 This model was contributed by the community contributor
 [HimariO](https://github.com/HimariO) 
 ❤️.
 


## Available Pipelines:



| 	 Pipeline	  | 	 Tasks	  | 	 Demo	  |
| --- | --- | --- |
| [StableDiffusionAdapterPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py)  | *Text-to-Image Generation with T2I-Adapter Conditioning*  | 	 -	  |
| [StableDiffusionXLAdapterPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py)  | *Text-to-Image Generation with T2I-Adapter Conditioning on StableDiffusion-XL*  | 	 -	  |


## Usage example with the base model of StableDiffusion-1.4/1.5




 In the following we give a simple example of how to use a
 *T2IAdapter* 
 checkpoint with Diffusers for inference based on StableDiffusion-1.4/1.5.
All adapters use the same pipeline.
 


1. Images are first converted into the appropriate
 *control image* 
 format.
2. The
 *control image* 
 and
 *prompt* 
 are passed to the
 [StableDiffusionAdapterPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/adapter#diffusers.StableDiffusionAdapterPipeline) 
.



 Let’s have a look at a simple example using the
 [Color Adapter](https://huggingface.co/TencentARC/t2iadapter_color_sd14v1) 
.
 



```
from diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color\_ref.png")
```



![img](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_ref.png)




 Then we can create our color palette by simply resizing it to 8 by 8 pixels and then scaling it back to original size.
 



```
from PIL import Image

color_palette = image.resize((8, 8))
color_palette = color_palette.resize((512, 512), resample=Image.Resampling.NEAREST)
```



 Let’s take a look at the processed image.
 



![img](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_palette.png)




 Next, create the adapter pipeline
 



```
import torch
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter\_color\_sd14v1", torch_dtype=torch.float16)
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    adapter=adapter,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```



 Finally, pass the prompt and control image to the pipeline
 



```
# fix the random seed, so you will get the same result as the example
generator = torch.manual_seed(7)

out_image = pipe(
    "At night, glowing cubes in front of the beach",
    image=color_palette,
    generator=generator,
).images[0]
```



![img](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_output.png)


## Usage example with the base model of StableDiffusion-XL




 In the following we give a simple example of how to use a
 *T2IAdapter* 
 checkpoint with Diffusers for inference based on StableDiffusion-XL.
All adapters use the same pipeline.
 


1. Images are first downloaded into the appropriate
 *control image* 
 format.
2. The
 *control image* 
 and
 *prompt* 
 are passed to the
 [StableDiffusionXLAdapterPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/adapter#diffusers.StableDiffusionXLAdapterPipeline) 
.



 Let’s have a look at a simple example using the
 [Sketch Adapter](https://huggingface.co/Adapter/t2iadapter/tree/main/sketch_sdxl_1.0) 
.
 



```
from diffusers.utils import load_image

sketch_image = load_image("https://huggingface.co/Adapter/t2iadapter/resolve/main/sketch.png").convert("L")
```



![img](https://huggingface.co/Adapter/t2iadapter/resolve/main/sketch.png)




 Then, create the adapter pipeline
 



```
import torch
from diffusers import (
    T2IAdapter,
    StableDiffusionXLAdapterPipeline,
    DDPMScheduler
)
from diffusers.models.unet_2d_condition import UNet2DConditionModel

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter = T2IAdapter.from_pretrained("Adapter/t2iadapter", subfolder="sketch\_sdxl\_1.0",torch_dtype=torch.float16, adapter_type="full\_adapter\_xl")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, adapter=adapter, safety_checker=None, torch_dtype=torch.float16, variant="fp16", scheduler=scheduler
)

pipe.to("cuda")
```



 Finally, pass the prompt and control image to the pipeline
 



```
# fix the random seed, so you will get the same result as the example
generator = torch.Generator().manual_seed(42)

sketch_image_out = pipe(
    prompt="a photo of a dog in real world, high quality", 
    negative_prompt="extra digit, fewer digits, cropped, worst quality, low quality", 
    image=sketch_image, 
    generator=generator, 
    guidance_scale=7.5
).images[0]
```



![img](https://huggingface.co/Adapter/t2iadapter/resolve/main/sketch_output.png)


## Available checkpoints




 Non-diffusers checkpoints can be found under
 [TencentARC/T2I-Adapter](https://huggingface.co/TencentARC/T2I-Adapter/tree/main/models) 
.
 


### 


 T2I-Adapter with Stable Diffusion 1.4


| 	 Model Name	  | 	 Control Image Overview	  | 	 Control Image Example	  | 	 Generated Image Example	  |
| --- | --- | --- | --- |
| [TencentARC/t2iadapter\_color\_sd14v1](https://huggingface.co/TencentARC/t2iadapter_color_sd14v1) 		*Trained with spatial color palette*  | 	 A image with 8x8 color palette.	  |  |  |
| [TencentARC/t2iadapter\_canny\_sd14v1](https://huggingface.co/TencentARC/t2iadapter_canny_sd14v1) 		*Trained with canny edge detection*  | 	 A monochrome image with white edges on a black background.	  |  |  |
| [TencentARC/t2iadapter\_sketch\_sd14v1](https://huggingface.co/TencentARC/t2iadapter_sketch_sd14v1) 		*Trained with	 [PidiNet](https://github.com/zhuoinoulu/pidinet) 	 edge detection*  | 	 A hand-drawn monochrome image with white outlines on a black background.	  |  |  |
| [TencentARC/t2iadapter\_depth\_sd14v1](https://huggingface.co/TencentARC/t2iadapter_depth_sd14v1) 		*Trained with Midas depth estimation*  | 	 A grayscale image with black representing deep areas and white representing shallow areas.	  |  |  |
| [TencentARC/t2iadapter\_openpose\_sd14v1](https://huggingface.co/TencentARC/t2iadapter_openpose_sd14v1) 		*Trained with OpenPose bone image*  | 	 A	 [OpenPose bone](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 	 image.	  |  |  |
| [TencentARC/t2iadapter\_keypose\_sd14v1](https://huggingface.co/TencentARC/t2iadapter_keypose_sd14v1) 		*Trained with mmpose skeleton image*  | 	 A	 [mmpose skeleton](https://github.com/open-mmlab/mmpose) 	 image.	  |  |  |
| [TencentARC/t2iadapter\_seg\_sd14v1](https://huggingface.co/TencentARC/t2iadapter_seg_sd14v1) 		*Trained with semantic segmentation*  | 	 An	 [custom](https://github.com/TencentARC/T2I-Adapter/discussions/25) 	 segmentation protocol image.	  |  |  |
| [TencentARC/t2iadapter\_canny\_sd15v2](https://huggingface.co/TencentARC/t2iadapter_canny_sd15v2)  |  |  |  |
| [TencentARC/t2iadapter\_depth\_sd15v2](https://huggingface.co/TencentARC/t2iadapter_depth_sd15v2)  |  |  |  |
| [TencentARC/t2iadapter\_sketch\_sd15v2](https://huggingface.co/TencentARC/t2iadapter_sketch_sd15v2)  |  |  |  |
| [TencentARC/t2iadapter\_zoedepth\_sd15v1](https://huggingface.co/TencentARC/t2iadapter_zoedepth_sd15v1)  |  |  |  |
| [Adapter/t2iadapter, subfolder=‘sketch\_sdxl\_1.0’](https://huggingface.co/Adapter/t2iadapter/tree/main/sketch_sdxl_1.0)  |  |  |  |
| [Adapter/t2iadapter, subfolder=‘canny\_sdxl\_1.0’](https://huggingface.co/Adapter/t2iadapter/tree/main/canny_sdxl_1.0)  |  |  |  |
| [Adapter/t2iadapter, subfolder=‘openpose\_sdxl\_1.0’](https://huggingface.co/Adapter/t2iadapter/tree/main/openpose_sdxl_1.0)  |  |  |  |


## Combining multiple adapters




`MultiAdapter` 
 can be used for applying multiple conditionings at once.
 



 Here we use the keypose adapter for the character posture and the depth adapter for creating the scene.
 



```
import torch
from PIL import Image
from diffusers.utils import load_image

cond_keypose = load_image(
    "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose\_sample\_input.png"
)
cond_depth = load_image(
    "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth\_sample\_input.png"
)
cond = [[cond_keypose, cond_depth]]

prompt = ["A man walking in an office room with a nice view"]
```



 The two control images look as such:
 



![img](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_input.png)
![img](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_input.png)




`MultiAdapter` 
 combines keypose and depth adapters.
 



`adapter_conditioning_scale` 
 balances the relative influence of the different adapters.
 



```
from diffusers import StableDiffusionAdapterPipeline, MultiAdapter

adapters = MultiAdapter(
    [
        T2IAdapter.from_pretrained("TencentARC/t2iadapter\_keypose\_sd14v1"),
        T2IAdapter.from_pretrained("TencentARC/t2iadapter\_depth\_sd14v1"),
    ]
)
adapters = adapters.to(torch.float16)

pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    adapter=adapters,
)

images = pipe(prompt, cond, adapter_conditioning_scale=[0.8, 0.8])
```



![img](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_depth_sample_output.png)


## T2I Adapter vs ControlNet




 T2I-Adapter is similar to
 [ControlNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet) 
.
T2i-Adapter uses a smaller auxiliary network which is only run once for the entire diffusion process.
However, T2I-Adapter performs slightly worse than ControlNet.
 


## StableDiffusionAdapterPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionAdapterPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py#L121)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 tokenizer
 
 : CLIPTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 adapter
 
 : typing.Union[diffusers.models.adapter.T2IAdapter, diffusers.models.adapter.MultiAdapter, typing.List[diffusers.models.adapter.T2IAdapter]]
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 feature\_extractor
 
 : CLIPFeatureExtractor
 




 requires\_safety\_checker
 
 : bool = True
 



 )
 


 Parameters
 




* **adapter** 
 (
 `T2IAdapter` 
 or
 `MultiAdapter` 
 or
 `List[T2IAdapter]` 
 ) —
Provides additional conditioning to the unet during the denoising process. If you set multiple Adapter as a
list, the outputs from each Adapter are added together to create one combined additional conditioning.
* **adapter\_weights** 
 (
 `List[float]` 
 ,
 *optional* 
 , defaults to None) —
List of floats representing the weight which will be multiply to each adapter’s output before adding them
together.
* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
Frozen text-encoder. Stable Diffusion uses the text portion of
 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 
 , specifically
the
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 variant.
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) — Conditional U-Net architecture to denoise the encoded image latents.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded image latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.
* **safety\_checker** 
 (
 `StableDiffusionSafetyChecker` 
 ) —
Classification module that estimates whether generated images could be considered offensive or harmful.
Please, refer to the
 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
 for details.
* **feature\_extractor** 
 (
 `CLIPFeatureExtractor` 
 ) —
Model that extracts features from generated images to be used as inputs for the
 `safety_checker` 
.


 Pipeline for text-to-image generation using Stable Diffusion augmented with T2I-Adapter
 <https://arxiv.org/abs/2302.08453>




 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py#L613)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 image
 
 : typing.Union[torch.Tensor, PIL.Image.Image, typing.List[PIL.Image.Image]] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 7.5
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 adapter\_conditioning\_scale
 
 : typing.Union[float, typing.List[float]] = 1.0
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **image** 
 (
 `torch.FloatTensor` 
 ,
 `PIL.Image.Image` 
 ,
 `List[torch.FloatTensor]` 
 or
 `List[PIL.Image.Image]` 
 or
 `List[List[PIL.Image.Image]]` 
 ) —
The Adapter input condition. Adapter uses this input condition to generate guidance to Unet. If the
type is specified as
 `Torch.FloatTensor` 
 , it is passed to Adapter as is. PIL.Image.Image` can also be
accepted as an image. The control image is automatically resized to fit the output image.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size \* self.vae\_scale\_factor) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size \* self.vae\_scale\_factor) —
The width in pixels of the generated image.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 7.5) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead. If not defined, one has to pass
 `negative_prompt_embeds` 
. instead.
Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is less than
 `1` 
 ).
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor will ge generated by sampling using the supplied random
 `generator` 
.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput` 
 instead
of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttnProcessor` 
 as defined under
 `self.processor` 
 in
 [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) 
.
* **adapter\_conditioning\_scale** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 1.0) —
The outputs of the adapter are multiplied by
 `adapter_conditioning_scale` 
 before they are added to the
residual in the original unet. If multiple adapters are specified in init, you can set the
corresponding scale as a list.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple. When returning a tuple, the first element is a list with the generated images, and the second element is a list of` 
 bool
 `s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw) content, according to the` 
 safety\_checker`.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> from PIL import Image
>>> from diffusers.utils import load_image
>>> import torch
>>> from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

>>> image = load_image(
...     "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color\_ref.png"
... )

>>> color_palette = image.resize((8, 8))
>>> color_palette = color_palette.resize((512, 512), resample=Image.Resampling.NEAREST)

>>> adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter\_color\_sd14v1", torch_dtype=torch.float16)
>>> pipe = StableDiffusionAdapterPipeline.from_pretrained(
...     "CompVis/stable-diffusion-v1-4",
...     adapter=adapter,
...     torch_dtype=torch.float16,
... )

>>> pipe.to("cuda")

>>> out_image = pipe(
...     "At night, glowing cubes in front of the beach",
...     image=color_palette,
... ).images[0]
```


#### 




 enable\_attention\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2027)



 (
 


 slice\_size
 
 : typing.Union[str, int, NoneType] = 'auto'
 



 )
 


 Parameters
 




* **slice\_size** 
 (
 `str` 
 or
 `int` 
 ,
 *optional* 
 , defaults to
 `"auto"` 
 ) —
When
 `"auto"` 
 , halves the input to the attention heads, so attention will be computed in two steps. If
 `"max"` 
 , maximum amount of memory will be saved by running only one slice at a time. If a number is
provided, uses as many slices as
 `attention_head_dim //slice_size` 
. In this case,
 `attention_head_dim` 
 must be a multiple of
 `slice_size` 
.


 Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
in slices to compute attention in several steps. For more than one attention head, the computation is performed
sequentially over each head. This is useful to save some memory in exchange for a small speed decrease.
 




 ⚠️ Don’t enable attention slicing if you’re already using
 `scaled_dot_product_attention` 
 (SDPA) from PyTorch
2.0 or xFormers. These attention computations are already very memory efficient so you won’t need to enable
this function. If you enable attention slicing with SDPA or xFormers, it can lead to serious slow downs!
 



 Examples:
 



```
>>> import torch
>>> from diffusers import StableDiffusionPipeline

>>> pipe = StableDiffusionPipeline.from_pretrained(
...     "runwayml/stable-diffusion-v1-5",
...     torch_dtype=torch.float16,
...     use_safetensors=True,
... )

>>> prompt = "a photo of an astronaut riding a horse on mars"
>>> pipe.enable_attention_slicing()
>>> image = pipe(prompt).images[0]
```


#### 




 disable\_attention\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2067)



 (
 

 )
 




 Disable sliced attention computation. If
 `enable_attention_slicing` 
 was previously called, attention is
computed in one step.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py#L206)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py#L214)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1966)



 (
 


 attention\_op
 
 : typing.Optional[typing.Callable] = None
 



 )
 


 Parameters
 




* **attention\_op** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
Override the default
 `None` 
 operator for use as
 `op` 
 argument to the
 [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
 function of xFormers.


 Enable memory efficient attention from
 [xFormers](https://facebookresearch.github.io/xformers/) 
. When this
option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
up during training is not guaranteed.
 




 ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
precedent.
 



 Examples:
 



```
>>> import torch
>>> from diffusers import DiffusionPipeline
>>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

>>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")
>>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
>>> # Workaround for not accepting attention shape using VAE for Flash Attention
>>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
```


#### 




 disable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2001)



 (
 

 )
 




 Disable memory efficient attention from
 [xFormers](https://facebookresearch.github.io/xformers/) 
.
 




#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py#L609)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py#L586)



 (
 


 s1
 
 : float
 




 s2
 
 : float
 




 b1
 
 : float
 




 b2
 
 : float
 



 )
 


 Parameters
 




* **s1** 
 (
 `float` 
 ) —
Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **s2** 
 (
 `float` 
 ) —
Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **b1** 
 (
 `float` 
 ) — Scaling factor for stage 1 to amplify the contributions of backbone features.
* **b2** 
 (
 `float` 
 ) — Scaling factor for stage 2 to amplify the contributions of backbone features.


 Enables the FreeU mechanism as in
 <https://arxiv.org/abs/2309.11497>
.
 



 The suffixes after the scaling factors represent the stages where they are being applied.
 



 Please refer to the
 [official repository](https://github.com/ChenyangSi/FreeU) 
 for combinations of the values
that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py#L255)



 (
 


 prompt
 


 device
 


 num\_images\_per\_prompt
 


 do\_classifier\_free\_guidance
 


 negative\_prompt
 
 = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 lora\_scale
 
 : typing.Optional[float] = None
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
device — (
 `torch.device` 
 ):
torch device
* **num\_images\_per\_prompt** 
 (
 `int` 
 ) —
number of images that should be generated per prompt
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ) —
whether to use classifier free guidance or not
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **lora\_scale** 
 (
 `float` 
 ,
 *optional* 
 ) —
A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.


 Encodes the prompt into text encoder hidden states.
 


## StableDiffusionXLAdapterPipeline




### 




 class
 

 diffusers.
 

 StableDiffusionXLAdapterPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L126)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 text\_encoder\_2
 
 : CLIPTextModelWithProjection
 




 tokenizer
 
 : CLIPTokenizer
 




 tokenizer\_2
 
 : CLIPTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 adapter
 
 : typing.Union[diffusers.models.adapter.T2IAdapter, diffusers.models.adapter.MultiAdapter, typing.List[diffusers.models.adapter.T2IAdapter]]
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 force\_zeros\_for\_empty\_prompt
 
 : bool = True
 



 )
 


 Parameters
 




* **adapter** 
 (
 `T2IAdapter` 
 or
 `MultiAdapter` 
 or
 `List[T2IAdapter]` 
 ) —
Provides additional conditioning to the unet during the denoising process. If you set multiple Adapter as a
list, the outputs from each Adapter are added together to create one combined additional conditioning.
* **adapter\_weights** 
 (
 `List[float]` 
 ,
 *optional* 
 , defaults to None) —
List of floats representing the weight which will be multiply to each adapter’s output before adding them
together.
* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
Frozen text-encoder. Stable Diffusion uses the text portion of
 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 
 , specifically
the
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 variant.
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
Tokenizer of class
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) 
.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) — Conditional U-Net architecture to denoise the encoded image latents.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded image latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.
* **safety\_checker** 
 (
 `StableDiffusionSafetyChecker` 
 ) —
Classification module that estimates whether generated images could be considered offensive or harmful.
Please, refer to the
 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
 for details.
* **feature\_extractor** 
 (
 `CLIPFeatureExtractor` 
 ) —
Model that extracts features from generated images to be used as inputs for the
 `safety_checker` 
.


 Pipeline for text-to-image generation using Stable Diffusion augmented with T2I-Adapter
 <https://arxiv.org/abs/2302.08453>




 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L673)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 prompt\_2
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 image
 
 : typing.Union[torch.Tensor, PIL.Image.Image, typing.List[PIL.Image.Image]] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 num\_inference\_steps
 
 : int = 50
 




 denoising\_end
 
 : typing.Optional[float] = None
 




 guidance\_scale
 
 : float = 5.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 negative\_prompt\_2
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_images\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 output\_type
 
 : typing.Optional[str] = 'pil'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 guidance\_rescale
 
 : float = 0.0
 




 original\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




 crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 target\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




 negative\_original\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




 negative\_crops\_coords\_top\_left
 
 : typing.Tuple[int, int] = (0, 0)
 




 negative\_target\_size
 
 : typing.Union[typing.Tuple[int, int], NoneType] = None
 




 adapter\_conditioning\_scale
 
 : typing.Union[float, typing.List[float]] = 1.0
 




 adapter\_conditioning\_factor
 
 : float = 1.0
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput` 
 or
 `tuple` 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide the image generation. If not defined, one has to pass
 `prompt_embeds` 
.
instead.
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to the
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders
* **image** 
 (
 `torch.FloatTensor` 
 ,
 `PIL.Image.Image` 
 ,
 `List[torch.FloatTensor]` 
 or
 `List[PIL.Image.Image]` 
 or
 `List[List[PIL.Image.Image]]` 
 ) —
The Adapter input condition. Adapter uses this input condition to generate guidance to Unet. If the
type is specified as
 `Torch.FloatTensor` 
 , it is passed to Adapter as is. PIL.Image.Image` can also be
accepted as an image. The control image is automatically resized to fit the output image.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size \* self.vae\_scale\_factor) —
The height in pixels of the generated image. Anything below 512 pixels won’t work well for
 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
 and checkpoints that are not specifically fine-tuned on low resolutions.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to self.unet.config.sample\_size \* self.vae\_scale\_factor) —
The width in pixels of the generated image. Anything below 512 pixels won’t work well for
 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
 and checkpoints that are not specifically fine-tuned on low resolutions.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference.
* **denoising\_end** 
 (
 `float` 
 ,
 *optional* 
 ) —
When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
completed before it is intentionally prematurely terminated. As a result, the returned sample will
still retain a substantial amount of noise as determined by the discrete timesteps selected by the
scheduler. The denoising\_end parameter should ideally be utilized when this pipeline forms a part of a
“Mixture of Denoisers” multi-pipeline setup, as elaborated in
 [**Refining the Image
Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 5.0) —
Guidance scale as defined in
 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 
.
 `guidance_scale` 
 is defined as
 `w` 
 of equation 2. of
 [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf) 
. Guidance scale is enabled by setting
 `guidance_scale > 1` 
. Higher guidance scale encourages to generate images that are closely linked to the text
 `prompt` 
 ,
usually at the expense of lower image quality.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders
* **num\_images\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of images to generate per prompt.
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) in the DDIM paper:
 <https://arxiv.org/abs/2010.02502>
. Only applies to
 [schedulers.DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , will be ignored for others.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
One or a list of
 [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) 
 to make generation deterministic.
* **latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor will ge generated by sampling using the supplied random
 `generator` 
.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting.
If not provided, pooled text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, pooled negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"pil"` 
 ) —
The output format of the generate image. Choose between
 [PIL](https://pillow.readthedocs.io/en/stable/) 
 :
 `PIL.Image.Image` 
 or
 `np.array` 
.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~pipelines.stable_diffusion_xl.StableDiffusionAdapterPipelineOutput` 
 instead of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that will be called every
 `callback_steps` 
 steps during inference. The function will be
called with the following arguments:
 `callback(step: int, timestep: int, latents: torch.FloatTensor)` 
.
* **callback\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The frequency at which the
 `callback` 
 function will be called. If not specified, the callback will be
called at every step.
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
* **guidance\_rescale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Guidance rescale factor proposed by
 [Common Diffusion Noise Schedules and Sample Steps are
Flawed](https://arxiv.org/pdf/2305.08891.pdf) 
`guidance_scale` 
 is defined as
 `φ` 
 in equation 16. of
 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 
.
Guidance rescale factor should fix overexposure when using zero terminal SNR.
* **original\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
If
 `original_size` 
 is not the same as
 `target_size` 
 the image will appear to be down- or upsampled.
 `original_size` 
 defaults to
 `(height, width)` 
 if not specified. Part of SDXL’s micro-conditioning as
explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **crops\_coords\_top\_left** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (0, 0)) —
 `crops_coords_top_left` 
 can be used to generate an image that appears to be “cropped” from the position
 `crops_coords_top_left` 
 downwards. Favorable, well-centered images are usually achieved by setting
 `crops_coords_top_left` 
 to (0, 0). Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **target\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
For most cases,
 `target_size` 
 should be set to the desired height and width of the generated image. If
not specified it will default to
 `(height, width)` 
. Part of SDXL’s micro-conditioning as explained in
section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
section 2.2 of
 <https://huggingface.co/papers/2307.01952>
.
* **negative\_original\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
To negatively condition the generation process based on a specific image resolution. Part of SDXL’s
micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **negative\_crops\_coords\_top\_left** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (0, 0)) —
To negatively condition the generation process based on a specific crop coordinates. Part of SDXL’s
micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **negative\_target\_size** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to (1024, 1024)) —
To negatively condition the generation process based on a target image resolution. It should be as same
as the
 `target_size` 
 for most cases. Part of SDXL’s micro-conditioning as explained in section 2.2 of
 <https://huggingface.co/papers/2307.01952>
. For more
information, refer to this issue thread:
 <https://github.com/huggingface/diffusers/issues/4208>
.
* **adapter\_conditioning\_scale** 
 (
 `float` 
 or
 `List[float]` 
 ,
 *optional* 
 , defaults to 1.0) —
The outputs of the adapter are multiplied by
 `adapter_conditioning_scale` 
 before they are added to the
residual in the original unet. If multiple adapters are specified in init, you can set the
corresponding scale as a list.
* **adapter\_conditioning\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) —
The fraction of timesteps for which adapter should be applied. If
 `adapter_conditioning_factor` 
 is
 `0.0` 
 , adapter is not applied at all. If
 `adapter_conditioning_factor` 
 is
 `1.0` 
 , adapter is applied for
all timesteps. If
 `adapter_conditioning_factor` 
 is
 `0.5` 
 , adapter is applied for half of the timesteps.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.




 Returns
 




 export const metadata = 'undefined';
 

`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




`~pipelines.stable_diffusion.StableDiffusionAdapterPipelineOutput` 
 if
 `return_dict` 
 is True, otherwise a
 `tuple` 
. When returning a tuple, the first element is a list with the generated images.
 



 Function invoked when calling the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import T2IAdapter, StableDiffusionXLAdapterPipeline, DDPMScheduler
>>> from diffusers.utils import load_image

>>> sketch_image = load_image("https://huggingface.co/Adapter/t2iadapter/resolve/main/sketch.png").convert("L")

>>> model_id = "stabilityai/stable-diffusion-xl-base-1.0"

>>> adapter = T2IAdapter.from_pretrained(
...     "Adapter/t2iadapter",
...     subfolder="sketch\_sdxl\_1.0",
...     torch_dtype=torch.float16,
...     adapter_type="full\_adapter\_xl",
... )
>>> scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

>>> pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
...     model_id, adapter=adapter, torch_dtype=torch.float16, variant="fp16", scheduler=scheduler
... ).to("cuda")

>>> generator = torch.manual_seed(42)
>>> sketch_image_out = pipe(
...     prompt="a photo of a dog in real world, high quality",
...     negative_prompt="extra digit, fewer digits, cropped, worst quality, low quality",
...     image=sketch_image,
...     generator=generator,
...     guidance_scale=7.5,
... ).images[0]
```


#### 




 enable\_attention\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2027)



 (
 


 slice\_size
 
 : typing.Union[str, int, NoneType] = 'auto'
 



 )
 


 Parameters
 




* **slice\_size** 
 (
 `str` 
 or
 `int` 
 ,
 *optional* 
 , defaults to
 `"auto"` 
 ) —
When
 `"auto"` 
 , halves the input to the attention heads, so attention will be computed in two steps. If
 `"max"` 
 , maximum amount of memory will be saved by running only one slice at a time. If a number is
provided, uses as many slices as
 `attention_head_dim //slice_size` 
. In this case,
 `attention_head_dim` 
 must be a multiple of
 `slice_size` 
.


 Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
in slices to compute attention in several steps. For more than one attention head, the computation is performed
sequentially over each head. This is useful to save some memory in exchange for a small speed decrease.
 




 ⚠️ Don’t enable attention slicing if you’re already using
 `scaled_dot_product_attention` 
 (SDPA) from PyTorch
2.0 or xFormers. These attention computations are already very memory efficient so you won’t need to enable
this function. If you enable attention slicing with SDPA or xFormers, it can lead to serious slow downs!
 



 Examples:
 



```
>>> import torch
>>> from diffusers import StableDiffusionPipeline

>>> pipe = StableDiffusionPipeline.from_pretrained(
...     "runwayml/stable-diffusion-v1-5",
...     torch_dtype=torch.float16,
...     use_safetensors=True,
... )

>>> prompt = "a photo of an astronaut riding a horse on mars"
>>> pipe.enable_attention_slicing()
>>> image = pipe(prompt).images[0]
```


#### 




 disable\_attention\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2067)



 (
 

 )
 




 Disable sliced attention computation. If
 `enable_attention_slicing` 
 was previously called, attention is
computed in one step.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L195)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L203)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L1966)



 (
 


 attention\_op
 
 : typing.Optional[typing.Callable] = None
 



 )
 


 Parameters
 




* **attention\_op** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
Override the default
 `None` 
 operator for use as
 `op` 
 argument to the
 [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
 function of xFormers.


 Enable memory efficient attention from
 [xFormers](https://facebookresearch.github.io/xformers/) 
. When this
option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
up during training is not guaranteed.
 




 ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
precedent.
 



 Examples:
 



```
>>> import torch
>>> from diffusers import DiffusionPipeline
>>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

>>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")
>>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
>>> # Workaround for not accepting attention shape using VAE for Flash Attention
>>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
```


#### 




 disable\_xformers\_memory\_efficient\_attention




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L2001)



 (
 

 )
 




 Disable memory efficient attention from
 [xFormers](https://facebookresearch.github.io/xformers/) 
.
 




#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L669)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L220)



 (
 

 )
 




 Disable tiled VAE decoding. If
 `enable_vae_tiling` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L646)



 (
 


 s1
 
 : float
 




 s2
 
 : float
 




 b1
 
 : float
 




 b2
 
 : float
 



 )
 


 Parameters
 




* **s1** 
 (
 `float` 
 ) —
Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **s2** 
 (
 `float` 
 ) —
Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
mitigate “oversmoothing effect” in the enhanced denoising process.
* **b1** 
 (
 `float` 
 ) — Scaling factor for stage 1 to amplify the contributions of backbone features.
* **b2** 
 (
 `float` 
 ) — Scaling factor for stage 2 to amplify the contributions of backbone features.


 Enables the FreeU mechanism as in
 <https://arxiv.org/abs/2309.11497>
.
 



 The suffixes after the scaling factors represent the stages where they are being applied.
 



 Please refer to the
 [official repository](https://github.com/ChenyangSi/FreeU) 
 for combinations of the values
that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L211)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py#L228)



 (
 


 prompt
 
 : str
 




 prompt\_2
 
 : typing.Optional[str] = None
 




 device
 
 : typing.Optional[torch.device] = None
 




 num\_images\_per\_prompt
 
 : int = 1
 




 do\_classifier\_free\_guidance
 
 : bool = True
 




 negative\_prompt
 
 : typing.Optional[str] = None
 




 negative\_prompt\_2
 
 : typing.Optional[str] = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_pooled\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 lora\_scale
 
 : typing.Optional[float] = None
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 


 Parameters
 




* **prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
prompt to be encoded
* **prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to be sent to the
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `prompt` 
 is
used in both text-encoders
device — (
 `torch.device` 
 ):
torch device
* **num\_images\_per\_prompt** 
 (
 `int` 
 ) —
number of images that should be generated per prompt
* **do\_classifier\_free\_guidance** 
 (
 `bool` 
 ) —
whether to use classifier free guidance or not
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation. If not defined, one has to pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (i.e., ignored if
 `guidance_scale` 
 is
less than
 `1` 
 ).
* **negative\_prompt\_2** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts not to guide the image generation to be sent to
 `tokenizer_2` 
 and
 `text_encoder_2` 
. If not defined,
 `negative_prompt` 
 is used in both text-encoders
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not
provided, text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input
argument.
* **pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting.
If not provided, pooled text embeddings will be generated from
 `prompt` 
 input argument.
* **negative\_pooled\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt
weighting. If not provided, pooled negative\_prompt\_embeds will be generated from
 `negative_prompt` 
 input argument.
* **lora\_scale** 
 (
 `float` 
 ,
 *optional* 
 ) —
A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
* **clip\_skip** 
 (
 `int` 
 ,
 *optional* 
 ) —
Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
the output of the pre-final layer will be used for computing the prompt embeddings.


 Encodes the prompt into text encoder hidden states.