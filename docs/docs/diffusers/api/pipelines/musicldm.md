# MusicLDM

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/musicldm>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/musicldm>



 MusicLDM was proposed in
 [MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies](https://huggingface.co/papers/2308.01546) 
 by Ke Chen, Yusong Wu, Haohe Liu, Marianna Nezhurina, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
MusicLDM takes a text prompt as input and predicts the corresponding music sample.
 



 Inspired by
 [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) 
 and
 [AudioLDM](https://huggingface.co/docs/diffusers/api/pipelines/audioldm/overview) 
 ,
MusicLDM is a text-to-music
 *latent diffusion model (LDM)* 
 that learns continuous audio representations from
 [CLAP](https://huggingface.co/docs/transformers/main/model_doc/clap) 
 latents.
 



 MusicLDM is trained on a corpus of 466 hours of music data. Beat-synchronous data augmentation strategies are applied to
the music samples, both in the time domain and in the latent space. Using beat-synchronous data augmentation strategies
encourages the model to interpolate between the training samples, but stay within the domain of the training data. The
result is generated music that is more diverse while staying faithful to the corresponding style.
 



 The abstract of the paper is the following:
 



*In this paper, we present MusicLDM, a state-of-the-art text-to-music model that adapts Stable Diffusion and AudioLDM architectures to the music domain. We achieve this by retraining the contrastive language-audio pretraining model (CLAP) and the Hifi-GAN vocoder, as components of MusicLDM, on a collection of music data samples. Then, we leverage a beat tracking model and propose two different mixup strategies for data augmentation: beat-synchronous audio mixup and beat-synchronous latent mixup, to encourage the model to generate music more diverse while still staying faithful to the corresponding style.* 




 This pipeline was contributed by
 [sanchit-gandhi](https://huggingface.co/sanchit-gandhi) 
.
 


## Tips




 When constructing a prompt, keep in mind:
 


* Descriptive prompt inputs work best; use adjectives to describe the sound (for example, “high quality” or “clear”) and make the prompt context specific where possible (e.g. “melodic techno with a fast beat and synths” works better than “techno”).
* Using a
 *negative prompt* 
 can significantly improve the quality of the generated audio. Try using a negative prompt of “low quality, average quality”.



 During inference:
 


* The
 *quality* 
 of the generated audio sample can be controlled by the
 `num_inference_steps` 
 argument; higher steps give higher quality audio at the expense of slower inference.
* Multiple waveforms can be generated in one go: set
 `num_waveforms_per_prompt` 
 to a value greater than 1 to enable. Automatic scoring will be performed between the generated waveforms and prompt text, and the audios ranked from best to worst accordingly.
* The
 *length* 
 of the generated audio sample can be controlled by varying the
 `audio_length_in_s` 
 argument.




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## MusicLDMPipeline




### 




 class
 

 diffusers.
 

 MusicLDMPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/musicldm/pipeline_musicldm.py#L67)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : typing.Union[transformers.models.clap.modeling\_clap.ClapTextModelWithProjection, transformers.models.clap.modeling\_clap.ClapModel]
 




 tokenizer
 
 : typing.Union[transformers.models.roberta.tokenization\_roberta.RobertaTokenizer, transformers.models.roberta.tokenization\_roberta\_fast.RobertaTokenizerFast]
 




 feature\_extractor
 
 : typing.Optional[transformers.models.clap.feature\_extraction\_clap.ClapFeatureExtractor]
 




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
 [ClapModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clap#transformers.ClapModel) 
 ) —
Frozen text-audio embedding model (
 `ClapTextModel` 
 ), specifically the
 [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) 
 variant.
* **tokenizer** 
 (
 `PreTrainedTokenizer` 
 ) —
A
 [RobertaTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/roberta#transformers.RobertaTokenizer) 
 to tokenize text.
* **feature\_extractor** 
 (
 [ClapFeatureExtractor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clap#transformers.ClapFeatureExtractor) 
 ) —
Feature extractor to compute mel-spectrograms from audio waveforms.
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


 Pipeline for text-to-audio generation using MusicLDM.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/musicldm/pipeline_musicldm.py#L434)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 audio\_length\_in\_s
 
 : typing.Optional[float] = None
 




 num\_inference\_steps
 
 : int = 200
 




 guidance\_scale
 
 : float = 2.0
 




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
 , defaults to 10.24) —
The length of the generated audio sample in seconds.
* **num\_inference\_steps** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 200) —
The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
expense of slower inference.
* **guidance\_scale** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 2.0) —
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
The number of waveforms to generate per prompt. If
 `num_waveforms_per_prompt > 1` 
 , the text encoding
model is a joint text-audio model (
 [ClapModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clap#transformers.ClapModel) 
 ), and the tokenizer is a
 `[~transformers.ClapProcessor]` 
 , then automatic scoring will be performed between the generated outputs
and the input text. This scoring ranks the generated waveforms based on their cosine similarity to text
input in the joint text-audio embedding space.
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
The output format of the generated audio. Choose between
 `"np"` 
 to return a NumPy
 `np.ndarray` 
 or
 `"pt"` 
 to return a PyTorch
 `torch.Tensor` 
 object. Set to
 `"latent"` 
 to return the latent diffusion
model (LDM) output.




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
>>> from diffusers import MusicLDMPipeline
>>> import torch
>>> import scipy

>>> repo_id = "cvssp/audioldm-s-full-v2"
>>> pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/musicldm/pipeline_musicldm.py#L125)



 (
 

 )
 




 Disable sliced VAE decoding. If
 `enable_vae_slicing` 
 was previously enabled, this method will go back to
computing decoding in one step.
 




#### 




 enable\_model\_cpu\_offload




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/musicldm/pipeline_musicldm.py#L400)



 (
 


 gpu\_id
 
 = 0
 



 )
 




 Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
to
 `enable_sequential_cpu_offload` 
 , this method moves one whole model at a time to the GPU when its
 `forward` 
 method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
 `enable_sequential_cpu_offload` 
 , but performance is much better due to the iterative execution of the
 `unet` 
.
 




#### 




 enable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/musicldm/pipeline_musicldm.py#L117)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.