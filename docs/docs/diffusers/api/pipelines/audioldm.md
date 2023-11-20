# AudioLDM

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/audioldm>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/audioldm>



 AudioLDM was proposed in
 [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://huggingface.co/papers/2301.12503) 
 by Haohe Liu et al. Inspired by
 [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) 
 , AudioLDM
is a text-to-audio
 *latent diffusion model (LDM)* 
 that learns continuous audio representations from
 [CLAP](https://huggingface.co/docs/transformers/main/model_doc/clap) 
 latents. AudioLDM takes a text prompt as input and predicts the corresponding audio. It can generate text-conditional
sound effects, human speech and music.
 



 The abstract from the paper is:
 



*Text-to-audio (TTA) system has recently gained attention for its ability to synthesize general audio based on text descriptions. However, previous studies in TTA have limited generation quality with high computational costs. In this study, we propose AudioLDM, a TTA system that is built on a latent space to learn the continuous audio representations from contrastive language-audio pretraining (CLAP) latents. The pretrained CLAP models enable us to train LDMs with audio embedding while providing text embedding as a condition during sampling. By learning the latent representations of audio signals and their compositions without modeling the cross-modal relationship, AudioLDM is advantageous in both generation quality and computational efficiency. Trained on AudioCaps with a single GPU, AudioLDM achieves state-of-the-art TTA performance measured by both objective and subjective metrics (e.g., frechet distance). Moreover, AudioLDM is the first TTA system that enables various text-guided audio manipulations (e.g., style transfer) in a zero-shot fashion. Our implementation and demos are available at
 <https://audioldm.github.io>
.* 




 The original codebase can be found at
 [haoheliu/AudioLDM](https://github.com/haoheliu/AudioLDM) 
.
 


## Tips




 When constructing a prompt, keep in mind:
 


* Descriptive prompt inputs work best; you can use adjectives to describe the sound (for example, “high quality” or “clear”) and make the prompt context specific (for example, “water stream in a forest” instead of “stream”).
* It’s best to use general terms like “cat” or “dog” instead of specific names or abstract objects the model may not be familiar with.



 During inference:
 


* The
 *quality* 
 of the predicted audio sample can be controlled by the
 `num_inference_steps` 
 argument; higher steps give higher quality audio at the expense of slower inference.
* The
 *length* 
 of the predicted audio sample can be controlled by varying the
 `audio_length_in_s` 
 argument.




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## AudioLDMPipeline




### 




 class
 

 diffusers.
 

 AudioLDMPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm/pipeline_audioldm.py#L52)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : ClapTextModelWithProjection
 




 tokenizer
 
 : typing.Union[transformers.models.roberta.tokenization\_roberta.RobertaTokenizer, transformers.models.roberta.tokenization\_roberta\_fast.RobertaTokenizerFast]
 




 unet
 
 : UNet2DConditionModel
 




 scheduler
 
 : KarrasDiffusionSchedulers
 




 vocoder
 
 : SpeechT5HifiGan
 



 )
 


 Parameters
 




* **vae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
* **text\_encoder** 
 (
 [ClapTextModelWithProjection](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clap#transformers.ClapTextModelWithProjection) 
 ) —
Frozen text-encoder (
 `ClapTextModelWithProjection` 
 , specifically the
 [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) 
 variant.
* **tokenizer** 
 (
 `PreTrainedTokenizer` 
 ) —
A
 [RobertaTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/roberta#transformers.RobertaTokenizer) 
 to tokenize text.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
A
 `UNet2DConditionModel` 
 to denoise the encoded audio latents.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded audio latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 ,
 [LMSDiscreteScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) 
 , or
 [PNDMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/pndm#diffusers.PNDMScheduler) 
.
* **vocoder** 
 (
 [SpeechT5HifiGan](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/speecht5#transformers.SpeechT5HifiGan) 
 ) —
Vocoder of class
 `SpeechT5HifiGan` 
.


 Pipeline for text-to-audio generation using AudioLDM.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm/pipeline_audioldm.py#L366)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 audio\_length\_in\_s
 
 : typing.Optional[float] = None
 




 num\_inference\_steps
 
 : int = 10
 




 guidance\_scale
 
 : float = 2.5
 




 negative\_prompt
 
 : typing.Union[str, typing.List[str], NoneType] = None
 




 num\_waveforms\_per\_prompt
 
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
 




 return\_dict
 
 : bool = True
 




 callback
 
 : typing.Union[typing.Callable[[int, int, torch.FloatTensor], NoneType], NoneType] = None
 




 callback\_steps
 
 : typing.Optional[int] = 1
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 output\_type
 
 : typing.Optional[str] = 'np'
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[AudioPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/dance_diffusion#diffusers.AudioPipelineOutput) 
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
The prompt or prompts to guide audio generation. If not defined, you need to pass
 `prompt_embeds` 
.
* **audio\_length\_in\_s** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 5.12) —
The length of the generated audio sample in seconds.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 10) —
The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 2.5) —
A higher guidance scale value encourages the model to generate audio that is closely linked to the text
 `prompt` 
 at the expense of lower sound quality. Guidance scale is enabled when
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
The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
pass
 `negative_prompt_embeds` 
 instead. Ignored when not using guidance (
 `guidance_scale < 1` 
 ).
* **num\_waveforms\_per\_prompt** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) —
The number of waveforms to generate per prompt.
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
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
tensor is generated by sampling using the supplied random
 `generator` 
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
* **output\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"np"` 
 ) —
The output format of the generated image. Choose between
 `"np"` 
 to return a NumPy
 `np.ndarray` 
 or
 `"pt"` 
 to return a PyTorch
 `torch.Tensor` 
 object.




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
 


 Examples:
 



```
>>> from diffusers import AudioLDMPipeline
>>> import torch
>>> import scipy

>>> repo_id = "cvssp/audioldm-s-full-v2"
>>> pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
>>> audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

>>> # save the audio sample as a .wav file
>>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
```


#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm/pipeline_audioldm.py#L107)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm/pipeline_audioldm.py#L99)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 


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