🧪 This pipeline is for research purposes only.
 


# Text-to-video

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/text_to_video>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/text_to_video>



[VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation](https://huggingface.co/papers/2303.08320) 
 is by Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, Tieniu Tan.
 



 The abstract from the paper is:
 



*A diffusion probabilistic model (DPM), which constructs a forward diffusion process by gradually adding noise to data points and learns the reverse denoising process to generate new samples, has been shown to handle complex data distribution. Despite its recent success in image synthesis, applying DPMs to video generation is still challenging due to high-dimensional data spaces. Previous methods usually adopt a standard diffusion process, where frames in the same video clip are destroyed with independent noises, ignoring the content redundancy and temporal correlation. This work presents a decomposed diffusion process via resolving the per-frame noise into a base noise that is shared among all frames and a residual noise that varies along the time axis. The denoising pipeline employs two jointly-learned networks to match the noise decomposition accordingly. Experiments on various datasets confirm that our approach, termed as VideoFusion, surpasses both GAN-based and diffusion-based alternatives in high-quality video generation. We further show that our decomposed formulation can benefit from pre-trained image diffusion models and well-support text-conditioned video creation.* 




 You can find additional information about Text-to-Video on the
 [project page](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) 
 ,
 [original codebase](https://github.com/modelscope/modelscope/) 
 , and try it out in a
 [demo](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis) 
. Official checkpoints can be found at
 [damo-vilab](https://huggingface.co/damo-vilab) 
 and
 [cerspense](https://huggingface.co/cerspense) 
.
 


## Usage example



### 


 text-to-video-ms-1.7b



 Let’s start by generating a short video with the default length of 16 frames (2s at 8 fps):
 



```
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

prompt = "Spiderman is surfing"
video_frames = pipe(prompt).frames
video_path = export_to_video(video_frames)
video_path
```



 Diffusers supports different optimization techniques to improve the latency
and memory footprint of a pipeline. Since videos are often more memory-heavy than images,
we can enable CPU offloading and VAE slicing to keep the memory footprint at bay.
 



 Let’s generate a video of 8 seconds (64 frames) on the same GPU using CPU offloading and VAE slicing:
 



```
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.enable_model_cpu_offload()

# memory optimization
pipe.enable_vae_slicing()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=64).frames
video_path = export_to_video(video_frames)
video_path
```



 It just takes
 **7 GBs of GPU memory** 
 to generate the 64 video frames using PyTorch 2.0, “fp16” precision and the techniques mentioned above.
 



 We can also use a different scheduler easily, using the same method we’d use for Stable Diffusion:
 



```
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames)
video_path
```



 Here are some sample outputs:
 


|  |  |
| --- | --- |
| 	 An astronaut riding a horse.	 	An astronaut riding a horse.	 | 	 Darth vader surfing in waves.	 	Darth vader surfing in waves.	 |


### 


 cerspense/zeroscope\_v2\_576w & cerspense/zeroscope\_v2\_XL



 Zeroscope are watermark-free model and have been trained on specific sizes such as
 `576x320` 
 and
 `1024x576` 
.
One should first generate a video using the lower resolution checkpoint
 [`cerspense/zeroscope_v2_576w`](https://huggingface.co/cerspense/zeroscope_v2_576w)
 with
 [TextToVideoSDPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.TextToVideoSDPipeline) 
 ,
which can then be upscaled using
 [VideoToVideoSDPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.VideoToVideoSDPipeline) 
 and
 [`cerspense/zeroscope_v2_XL`](https://huggingface.co/cerspense/zeroscope_v2_XL)
.
 



```
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope\_v2\_576w", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=24).frames
video_path = export_to_video(video_frames)
video_path
```



 Now the video can be upscaled:
 



```
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope\_v2\_XL", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

video_frames = pipe(prompt, video=video, strength=0.6).frames
video_path = export_to_video(video_frames)
video_path
```



 Here are some sample outputs:
 


|  |
| --- |
| 	 Darth vader surfing in waves.	 	Darth vader surfing in waves.	 |


## TextToVideoSDPipeline




### 




 class
 

 diffusers.
 

 TextToVideoSDPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L79)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 tokenizer
 
 : CLIPTokenizer
 




 unet
 
 : UNet3DConditionModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 



 )
 


 Parameters
 




* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
Frozen text-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
A
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 to tokenize text.
* **unet** 
 (
 [UNet3DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet3d-cond#diffusers.UNet3DConditionModel) 
 ) —
A
 [UNet3DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet3d-cond#diffusers.UNet3DConditionModel) 
 to denoise the encoded video latents.
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


 Pipeline for text-to-video generation.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L515)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 height
 
 : typing.Optional[int] = None
 




 width
 
 : typing.Optional[int] = None
 




 num\_frames
 
 : int = 16
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 9.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




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
 
 : typing.Optional[str] = 'np'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
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
The prompt or prompts to guide image generation. If not defined, you need to pass
 `prompt_embeds` 
.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
The height in pixels of the generated video.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
The width in pixels of the generated video.
* **num\_frames** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 16) —
The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
amounts to 2 seconds of video.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 7.5) —
A higher guidance scale value encourages the model to generate images closely linked to the text
 `prompt` 
 at the expense of lower image quality. Guidance scale is enabled when
 `guidance_scale > 1` 
.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide what to not include in image generation. If not defined, you need to
pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (
 `guidance_scale < 1` 
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
Corresponds to parameter eta (η) from the
 [DDIM](https://arxiv.org/abs/2010.02502) 
 paper. Only applies
to the
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , and is ignored in other schedulers.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 to make
generation deterministic.
* **latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor is generated by sampling using the supplied random
 `generator` 
. Latents should be of shape
 `(batch_size, num_channel, num_frames, height, width)` 
.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
provided, text embeddings are generated from the
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
not provided,
 `negative_prompt_embeds` 
 are generated from the
 `negative_prompt` 
 input argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"np"` 
 ) —
The output format of the generated video. Choose between
 `torch.FloatTensor` 
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
 [TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
 instead
of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that calls every
 `callback_steps` 
 steps during inference. The function is called with the
following arguments:
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
 function is called. If not specified, the callback is called at
every step.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined in
 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
.
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
 

[TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
 is
returned, otherwise a
 `tuple` 
 is returned where the first element is a list with the generated frames.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import TextToVideoSDPipeline
>>> from diffusers.utils import export_to_video

>>> pipe = TextToVideoSDPipeline.from_pretrained(
...     "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
... )
>>> pipe.enable_model_cpu_offload()

>>> prompt = "Spiderman is surfing"
>>> video_frames = pipe(prompt).frames
>>> video_path = export_to_video(video_frames)
>>> video_path
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L511)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L129)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 disable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L146)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L488)



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




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L121)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L137)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L187)



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
 


## VideoToVideoSDPipeline




### 




 class
 

 diffusers.
 

 VideoToVideoSDPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L141)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 tokenizer
 
 : CLIPTokenizer
 




 unet
 
 : UNet3DConditionModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 



 )
 


 Parameters
 




* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
* **text\_encoder** 
 (
 `CLIPTextModel` 
 ) —
Frozen text-encoder (
 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 
 ).
* **tokenizer** 
 (
 `CLIPTokenizer` 
 ) —
A
 [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clip#transformers.CLIPTokenizer) 
 to tokenize text.
* **unet** 
 (
 [UNet3DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet3d-cond#diffusers.UNet3DConditionModel) 
 ) —
A
 [UNet3DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet3d-cond#diffusers.UNet3DConditionModel) 
 to denoise the encoded video latents.
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


 Pipeline for text-guided video-to-video generation.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L606)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 video
 
 : typing.Union[typing.List[numpy.ndarray], torch.FloatTensor] = None
 




 strength
 
 : float = 0.6
 




 num\_inference\_steps
 
 : int = 50
 




 guidance\_scale
 
 : float = 15.0
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




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
 
 : typing.Optional[str] = 'np'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : int = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 clip\_skip
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
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
The prompt or prompts to guide image generation. If not defined, you need to pass
 `prompt_embeds` 
.
* **video** 
 (
 `List[np.ndarray]` 
 or
 `torch.FloatTensor` 
 ) —
 `video` 
 frames or tensor representing a video batch to be used as the starting point for the process.
Can also accept video latents as
 `image` 
 , if passing latents directly, it will not be encoded again.
* **strength** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.8) —
Indicates extent to transform the reference
 `video` 
. Must be between 0 and 1.
 `video` 
 is used as a
starting point, adding more noise to it the larger the
 `strength` 
. The number of denoising steps
depends on the amount of noise initially added. When
 `strength` 
 is 1, added noise is maximum and the
denoising process runs for the full number of iterations specified in
 `num_inference_steps` 
. A value of
1 essentially ignores
 `video` 
.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 7.5) —
A higher guidance scale value encourages the model to generate images closely linked to the text
 `prompt` 
 at the expense of lower image quality. Guidance scale is enabled when
 `guidance_scale > 1` 
.
* **negative\_prompt** 
 (
 `str` 
 or
 `List[str]` 
 ,
 *optional* 
 ) —
The prompt or prompts to guide what to not include in video generation. If not defined, you need to
pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (
 `guidance_scale < 1` 
 ).
* **eta** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 0.0) —
Corresponds to parameter eta (η) from the
 [DDIM](https://arxiv.org/abs/2010.02502) 
 paper. Only applies
to the
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , and is ignored in other schedulers.
* **generator** 
 (
 `torch.Generator` 
 or
 `List[torch.Generator]` 
 ,
 *optional* 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 to make
generation deterministic.
* **latents** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor is generated by sampling using the supplied random
 `generator` 
. Latents should be of shape
 `(batch_size, num_channel, num_frames, height, width)` 
.
* **prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
provided, text embeddings are generated from the
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
not provided,
 `negative_prompt_embeds` 
 are generated from the
 `negative_prompt` 
 input argument.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"np"` 
 ) —
The output format of the generated video. Choose between
 `torch.FloatTensor` 
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
 [TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
 instead
of a plain tuple.
* **callback** 
 (
 `Callable` 
 ,
 *optional* 
 ) —
A function that calls every
 `callback_steps` 
 steps during inference. The function is called with the
following arguments:
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
 function is called. If not specified, the callback is called at
every step.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined in
 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
.
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
 

[TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [TextToVideoSDPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video#diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput) 
 is
returned, otherwise a
 `tuple` 
 is returned where the first element is a list with the generated frames.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import torch
>>> from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
>>> from diffusers.utils import export_to_video

>>> pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope\_v2\_576w", torch_dtype=torch.float16)
>>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
>>> pipe.to("cuda")

>>> prompt = "spiderman running in the desert"
>>> video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
>>> # safe low-res video
>>> video_path = export_to_video(video_frames, output_video_path="./video\_576\_spiderman.mp4")

>>> # let's offload the text-to-image model
>>> pipe.to("cpu")

>>> # and load the image-to-image model
>>> pipe = DiffusionPipeline.from_pretrained(
...     "cerspense/zeroscope\_v2\_XL", torch_dtype=torch.float16, revision="refs/pr/15"
... )
>>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
>>> pipe.enable_model_cpu_offload()

>>> # The VAE consumes A LOT of memory, let's make sure we run it in sliced mode
>>> pipe.vae.enable_slicing()

>>> # now let's upscale it
>>> video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

>>> # and denoise it
>>> video_frames = pipe(prompt, video=video, strength=0.6).frames
>>> video_path = export_to_video(video_frames, output_video_path="./video\_1024\_spiderman.mp4")
>>> video_path
```


#### 




 disable\_freeu




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L602)



 (
 

 )
 




 Disables the FreeU mechanism if enabled.
 




#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L191)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 disable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L208)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L579)



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




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L183)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 enable\_vae\_tiling




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L199)



 (
 

 )
 




 Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py#L249)



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
 


## TextToVideoSDPipelineOutput




### 




 class
 

 diffusers.pipelines.text\_to\_video\_synthesis.
 

 TextToVideoSDPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_output.py#L13)



 (
 


 frames
 
 : typing.Union[typing.List[numpy.ndarray], torch.FloatTensor]
 



 )
 


 Parameters
 




* **frames** 
 (
 `List[np.ndarray]` 
 or
 `torch.FloatTensor` 
 ) —
List of denoised frames (essentially images) as NumPy arrays of shape
 `(height, width, num_channels)` 
 or as
a
 `torch` 
 tensor. The length of the list denotes the video length (the number of frames).


 Output class for text-to-video pipelines.