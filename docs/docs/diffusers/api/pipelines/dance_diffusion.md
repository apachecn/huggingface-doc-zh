# Dance Diffusion

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/dance_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/dance_diffusion>



[Dance Diffusion](https://github.com/Harmonai-org/sample-generator) 
 is by Zach Evans.
 



 Dance Diffusion is the first in a suite of generative audio tools for producers and musicians released by
 [Harmonai](https://github.com/Harmonai-org) 
.
 



 The original codebase of this implementation can be found at
 [Harmonai-org](https://github.com/Harmonai-org/sample-generator) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## DanceDiffusionPipeline




### 




 class
 

 diffusers.
 

 DanceDiffusionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/dance_diffusion/pipeline_dance_diffusion.py#L28)



 (
 


 unet
 


 scheduler
 




 )
 


 Parameters
 




* **unet** 
 (
 [UNet1DModel](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.UNet1DModel) 
 ) —
A
 `UNet1DModel` 
 to denoise the encoded audio.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded audio latents. Can be one of
 [IPNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ipndm#diffusers.IPNDMScheduler) 
.


 Pipeline for audio generation.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/dance_diffusion/pipeline_dance_diffusion.py#L48)



 (
 


 batch\_size
 
 : int = 1
 




 num\_inference\_steps
 
 : int = 100
 




 generator
 
 : typing.Union[torch.\_C.Generator, typing.List[torch.\_C.Generator], NoneType] = None
 




 audio\_length\_in\_s
 
 : typing.Optional[float] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[AudioPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/dance_diffusion#diffusers.AudioPipelineOutput) 
 or
 `tuple` 


 Parameters
 




* **batch\_size** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of audio samples to generate.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 50) —
The number of denoising steps. More denoising steps usually lead to a higher-quality audio sample at
the expense of slower inference.
* **generator** 
 (
 `torch.Generator` 
 ,
 *optional* 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 to make
generation deterministic.
* **audio\_length\_in\_s** 
 (
 `float` 
 ,
 *optional* 
 , defaults to
 `self.unet.config.sample_size/self.unet.config.sample_rate` 
 ) —
The length of the generated audio sample in seconds.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [AudioPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/dance_diffusion#diffusers.AudioPipelineOutput) 
 instead of a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

[AudioPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/dance_diffusion#diffusers.AudioPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [AudioPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/dance_diffusion#diffusers.AudioPipelineOutput) 
 is returned, otherwise a
 `tuple` 
 is
returned where the first element is a list with the generated audio.
 



 The call function to the pipeline for generation.
 


 Example:
 



```
from diffusers import DiffusionPipeline
from scipy.io.wavfile import write

model_id = "harmonai/maestro-150k"
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")

audios = pipe(audio_length_in_s=4.0).audios

# To save locally
for i, audio in enumerate(audios):
    write(f"maestro\_test\_{i}.wav", pipe.unet.sample_rate, audio.transpose())

# To dislay in google colab
import IPython.display as ipd

for audio in audios:
    display(ipd.Audio(audio, rate=pipe.unet.sample_rate))
```


## AudioPipelineOutput




### 




 class
 

 diffusers.
 

 AudioPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L124)



 (
 


 audios
 
 : ndarray
 



 )
 


 Parameters
 




* **audios** 
 (
 `np.ndarray` 
 ) —
List of denoised audio samples of a NumPy array of shape
 `(batch_size, num_channels, sample_rate)` 
.


 Output class for audio pipelines.