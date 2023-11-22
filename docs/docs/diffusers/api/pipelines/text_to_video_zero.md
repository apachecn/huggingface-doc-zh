# Text2Video-Zero

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/text_to_video_zero>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/text_to_video_zero>



[Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://huggingface.co/papers/2303.13439) 
 is by
Levon Khachatryan,
Andranik Movsisyan,
Vahram Tadevosyan,
Roberto Henschel,
 [Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang) 
 , Shant Navasardyan,
 [Humphrey Shi](https://www.humphreyshi.com) 
.
 



 Text2Video-Zero enables zero-shot video generation using either:
 


1. A textual prompt
2. A prompt combined with guidance from poses or edges
3. Video Instruct-Pix2Pix (instruction-guided video editing)



 Results are temporally consistent and closely follow the guidance and textual prompts.
 



![teaser-img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2v_zero_teaser.png)




 The abstract from the paper is:
 



*Recent text-to-video generation approaches rely on computationally heavy training and require large-scale video datasets. In this paper, we introduce a new task of zero-shot text-to-video generation and propose a low-cost approach (without any training or optimization) by leveraging the power of existing text-to-image synthesis methods (e.g., Stable Diffusion), making them suitable for the video domain.
Our key modifications include (i) enriching the latent codes of the generated frames with motion dynamics to keep the global scene and the background time consistent; and (ii) reprogramming frame-level self-attention using a new cross-frame attention of each frame on the first frame, to preserve the context, appearance, and identity of the foreground object.
Experiments show that this leads to low overhead, yet high-quality and remarkably consistent video generation. Moreover, our approach is not limited to text-to-video synthesis but is also applicable to other tasks such as conditional and content-specialized video generation, and Video Instruct-Pix2Pix, i.e., instruction-guided video editing.
As experiments show, our method performs comparably or sometimes better than recent approaches, despite not being trained on additional video data.* 




 You can find additional information about Text-to-Video Zero on the
 [project page](https://text2video-zero.github.io/) 
 ,
 [paper](https://arxiv.org/abs/2303.13439) 
 , and
 [original codebase](https://github.com/Picsart-AI-Research/Text2Video-Zero) 
.
 


## Usage example



### 


 Text-To-Video



 To generate a video from prompt, run the following python command
 



```
import torch
import imageio
from diffusers import TextToVideoZeroPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A panda is playing guitar on times square"
result = pipe(prompt=prompt).images
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video.mp4", result, fps=4)
```



 You can change these parameters in the pipeline call:
 


* Motion field strength (see the
 [paper](https://arxiv.org/abs/2303.13439) 
 , Sect. 3.3.1):
	+ `motion_field_strength_x` 
	 and
	 `motion_field_strength_y` 
	. Default:
	 `motion_field_strength_x=12` 
	 ,
	 `motion_field_strength_y=12`
* `T` 
 and
 `T'` 
 (see the
 [paper](https://arxiv.org/abs/2303.13439) 
 , Sect. 3.3.1)
	+ `t0` 
	 and
	 `t1` 
	 in the range
	 `{0,..., num_inference_steps}` 
	. Default:
	 `t0=45` 
	 ,
	 `t1=48`
* Video length:
	+ `video_length` 
	 , the number of frames video\_length to be generated. Default:
	 `video_length=8`



 We an also generate longer videos by doing the processing in a chunk-by-chunk manner:
 



```
import torch
import imageio
from diffusers import TextToVideoZeroPipeline
import numpy as np

model_id = "runwayml/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
seed = 0
video_length = 8
chunk_size = 4
prompt = "A panda is playing guitar on times square"

# Generate the video chunk-by-chunk
result = []
chunk_ids = np.arange(0, video_length, chunk_size - 1)
generator = torch.Generator(device="cuda")
for i in range(len(chunk_ids)):
    print(f"Processing chunk {i + 1} / {len(chunk\_ids)}")
    ch_start = chunk_ids[i]
    ch_end = video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
    # Attach the first frame for Cross Frame Attention
    frame_ids = [0] + list(range(ch_start, ch_end))
    # Fix the seed for the temporal consistency
    generator.manual_seed(seed)
    output = pipe(prompt=prompt, video_length=len(frame_ids), generator=generator, frame_ids=frame_ids)
    result.append(output.images[1:])

# Concatenate chunks and save
result = np.concatenate(result)
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video.mp4", result, fps=4)
```


### 


 Text-To-Video with Pose Control



 To generate a video from prompt with additional pose control
 


1. Download a demo video
 



```
from huggingface_hub import hf_hub_download

filename = "\_\_assets\_\_/poses\_skeleton\_gifs/dance1\_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
```


2. Read video containing extracted pose images
 



```
from PIL import Image
import imageio

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```



 To extract pose from actual video, read
 [ControlNet documentation](./stable_diffusion/controlnet) 
.
3. Run
 `StableDiffusionControlNetPipeline` 
 with our custom attention processor
 



```
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipe(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```


### 


 Text-To-Video with Edge Control



 To generate a video from prompt with additional pose control,
follow the steps described above for pose-guided generation using
 [Canny edge ControlNet model](https://huggingface.co/lllyasviel/sd-controlnet-canny) 
.
 


### 


 Video Instruct-Pix2Pix



 To perform text-guided video editing (with
 [InstructPix2Pix](./stable_diffusion/pix2pix) 
 ):
 


1. Download a demo video
 



```
from huggingface_hub import hf_hub_download

filename = "\_\_assets\_\_/pix2pix video/camel.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
```
2. Read video from path
 



```
from PIL import Image
import imageio

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
video = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```
3. Run
 `StableDiffusionInstructPix2PixPipeline` 
 with our custom attention processor
 



```
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

prompt = "make it Van Gogh Starry Night style"
result = pipe(prompt=[prompt] * len(video), image=video).images
imageio.mimsave("edited\_video.mp4", result, fps=4)
```


### 


 DreamBooth specialization



 Methods
 **Text-To-Video** 
 ,
 **Text-To-Video with Pose Control** 
 and
 **Text-To-Video with Edge Control** 
 can run with custom
 [DreamBooth](../training/dreambooth) 
 models, as shown below for
 [Canny edge ControlNet model](https://huggingface.co/lllyasviel/sd-controlnet-canny) 
 and
 [Avatar style DreamBooth](https://huggingface.co/PAIR/text2video-zero-controlnet-canny-avatar) 
 model
 


1. Download a demo video
 



```
from huggingface_hub import hf_hub_download

filename = "\_\_assets\_\_/canny\_videos\_mp4/girl\_turning.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
```
2. Read video from path
 



```
from PIL import Image
import imageio

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
canny_edges = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```
3. Run
 `StableDiffusionControlNetPipeline` 
 with custom trained DreamBooth model
 



```
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

# set model id to custom model
model_id = "PAIR/text2video-zero-controlnet-canny-avatar"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(canny_edges), 1, 1, 1)

prompt = "oil painting of a beautiful girl avatar style"
result = pipe(prompt=[prompt] * len(canny_edges), image=canny_edges, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```



 You can filter out some available DreamBooth-trained models with
 [this link](https://huggingface.co/models?search=dreambooth) 
.
 


## TextToVideoZeroPipeline




### 




 class
 

 diffusers.
 

 TextToVideoZeroPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py#L274)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : CLIPTextModel
 




 tokenizer
 
 : CLIPTokenizer
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 safety\_checker
 
 : StableDiffusionSafetyChecker
 




 feature\_extractor
 
 : CLIPImageProcessor
 




 requires\_safety\_checker
 
 : bool = True
 



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
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
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
* **safety\_checker** 
 (
 `StableDiffusionSafetyChecker` 
 ) —
Classification module that estimates whether generated images could be considered offensive or harmful.
Please refer to the
 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
 for more details
about a model’s potential harms.
* **feature\_extractor** 
 (
 `CLIPImageProcessor` 
 ) —
A
 `CLIPImageProcessor` 
 to extract features from generated images; used as inputs to the
 `safety_checker` 
.


 Pipeline for zero-shot text-to-video generation using Stable Diffusion.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py#L421)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]]
 




 video\_length
 
 : typing.Optional[int] = 8
 




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
 




 num\_videos\_per\_prompt
 
 : typing.Optional[int] = 1
 




 eta
 
 : float = 0.0
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 latents
 
 : typing.Optional[torch.FloatTensor] = None
 




 motion\_field\_strength\_x
 
 : float = 12
 




 motion\_field\_strength\_y
 
 : float = 12
 




 output\_type
 
 : typing.Optional[str] = 'tensor'
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : typing.Optional[int] = 1
 




 t0
 
 : int = 44
 




 t1
 
 : int = 47
 




 frame\_ids
 
 : typing.Optional[typing.List[int]] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[TextToVideoPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video_zero#diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoPipelineOutput) 


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
* **video\_length** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 8) —
The number of generated video frames.
* **height** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
The height in pixels of the generated image.
* **width** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size * self.vae_scale_factor` 
 ) —
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
* **num\_videos\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of videos to generate per prompt.
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
.
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"numpy"` 
 ) —
The output format of the generated video. Choose between
 `"latent"` 
 and
 `"numpy"` 
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
 [TextToVideoPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video_zero#diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoPipelineOutput) 
 instead of
a plain tuple.
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
* **motion\_field\_strength\_x** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 12) —
Strength of motion in generated video along x-axis. See the
 [paper](https://arxiv.org/abs/2303.13439) 
 ,
Sect. 3.3.1.
* **motion\_field\_strength\_y** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 12) —
Strength of motion in generated video along y-axis. See the
 [paper](https://arxiv.org/abs/2303.13439) 
 ,
Sect. 3.3.1.
* **t0** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 44) —
Timestep t0. Should be in the range [0, num\_inference\_steps - 1]. See the
 [paper](https://arxiv.org/abs/2303.13439) 
 , Sect. 3.3.1.
* **t1** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 47) —
Timestep t0. Should be in the range [t0 + 1, num\_inference\_steps - 1]. See the
 [paper](https://arxiv.org/abs/2303.13439) 
 , Sect. 3.3.1.
* **frame\_ids** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Indexes of the frames that are being generated. This is used when generating longer videos
chunk-by-chunk.




 Returns
 




 export const metadata = 'undefined';
 

[TextToVideoPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/text_to_video_zero#diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.TextToVideoPipelineOutput) 




 export const metadata = 'undefined';
 




 The output contains a
 `ndarray` 
 of the generated video, when
 `output_type` 
 !=
 `"latent"` 
 , otherwise a
latent code of generated videos and a list of
 `bool` 
 s indicating whether the corresponding generated
video contains “not-safe-for-work” (nsfw) content..
 



 The call function to the pipeline for generation.
 




#### 




 backward\_loop




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py#L346)



 (
 


 latents
 


 timesteps
 


 prompt\_embeds
 


 guidance\_scale
 


 callback
 


 callback\_steps
 


 num\_warmup\_steps
 


 extra\_step\_kwargs
 


 cross\_attention\_kwargs
 
 = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

 latents
 




 Parameters
 




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
extra\_step\_kwargs —
Extra\_step\_kwargs.
cross\_attention\_kwargs —
A kwargs dictionary that if specified is passed along to the
 `AttentionProcessor` 
 as defined in
 [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
.
num\_warmup\_steps —
number of warmup steps.




 Returns
 




 export const metadata = 'undefined';
 

 latents
 



 export const metadata = 'undefined';
 




 Latents of backward process output at time timesteps[-1].
 



 Perform backward process given list of time steps.
 




#### 




 forward\_loop




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py#L322)



 (
 


 x\_t0
 


 t0
 


 t1
 


 generator
 




 )
 

 →
 



 export const metadata = 'undefined';
 

 x\_t1
 




 Parameters
 




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




 Returns
 




 export const metadata = 'undefined';
 

 x\_t1
 



 export const metadata = 'undefined';
 




 Forward process applied to x\_t0 from time t0 to t1.
 



 Perform DDPM forward process from time t0 to t1. This is the same as adding noise with corresponding variance.
 


## TextToVideoPipelineOutput




### 




 class
 

 diffusers.pipelines.text\_to\_video\_synthesis.pipeline\_text\_to\_video\_zero.
 

 TextToVideoPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py#L174)



 (
 


 images
 
 : typing.Union[typing.List[PIL.Image.Image], numpy.ndarray]
 




 nsfw\_content\_detected
 
 : typing.Optional[typing.List[bool]]
 



 )
 


 Parameters
 




* **images** 
 (
 `[List[PIL.Image.Image]` 
 ,
 `np.ndarray` 
 ]) —
List of denoised PIL images of length
 `batch_size` 
 or NumPy array of shape
 `(batch_size, height, width, num_channels)` 
.
* **nsfw\_content\_detected** 
 (
 `[List[bool]]` 
 ) —
List indicating whether the corresponding generated image contains “not-safe-for-work” (nsfw) content or
 `None` 
 if safety checking could not be performed.


 Output class for zero-shot text-to-video pipeline.