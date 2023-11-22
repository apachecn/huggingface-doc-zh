# AudioLDM 2

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/audioldm2>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/audioldm2>



 AudioLDM 2 was proposed in
 [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) 
 by Haohe Liu et al. AudioLDM 2 takes a text prompt as input and predicts the corresponding audio. It can generate
text-conditional sound effects, human speech and music.
 



 Inspired by
 [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) 
 , AudioLDM 2
is a text-to-audio
 *latent diffusion model (LDM)* 
 that learns continuous audio representations from text embeddings. Two
text encoder models are used to compute the text embeddings from a prompt input: the text-branch of
 [CLAP](https://huggingface.co/docs/transformers/main/en/model_doc/clap) 
 and the encoder of
 [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5) 
. These text embeddings
are then projected to a shared embedding space by an
 [AudioLDM2ProjectionModel](https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2#diffusers.AudioLDM2ProjectionModel) 
.
A
 [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2) 
*language model (LM)* 
 is used to auto-regressively
predict eight new embedding vectors, conditional on the projected CLAP and Flan-T5 embeddings. The generated embedding
vectors and Flan-T5 text embeddings are used as cross-attention conditioning in the LDM. The
 [UNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2UNet2DConditionModel) 
 of AudioLDM 2 is unique in the sense that it takes
 **two** 
 cross-attention embeddings, as opposed to one cross-attention
conditioning, as in most other LDMs.
 



 The abstract of the paper is the following:
 



*Although audio generation shares commonalities across different types of audio, such as speech, music, and sound effects, designing models for each type requires careful consideration of specific objectives and biases that can significantly differ from those of other types. To bring us closer to a unified perspective of audio generation, this paper proposes a framework that utilizes the same learning method for speech, music, and sound effect generation. Our framework introduces a general representation of audio, called language of audio (LOA). Any audio can be translated into LOA based on AudioMAE, a self-supervised pre-trained representation learning model. In the generation process, we translate any modalities into LOA by using a GPT-2 model, and we perform self-supervised audio generation learning with a latent diffusion model conditioned on LOA. The proposed framework naturally brings advantages such as in-context learning abilities and reusable self-supervised pretrained AudioMAE and latent diffusion models. Experiments on the major benchmarks of text-to-audio, text-to-music, and text-to-speech demonstrate new state-of-the-art or competitive performance to previous approaches.* 




 This pipeline was contributed by
 [sanchit-gandhi](https://huggingface.co/sanchit-gandhi) 
. The original codebase can be
found at
 [haoheliu/audioldm2](https://github.com/haoheliu/audioldm2) 
.
 


## Tips



### 


 Choosing a checkpoint



 AudioLDM2 comes in three variants. Two of these checkpoints are applicable to the general task of text-to-audio
generation. The third checkpoint is trained exclusively on text-to-music generation.
 



 All checkpoints share the same model size for the text encoders and VAE. They differ in the size and depth of the UNet.
See table below for details on the three checkpoints:
 


| 	 Checkpoint	  | 	 Task	  | 	 UNet Model Size	  | 	 Total Model Size	  | 	 Training Data /h	  |
| --- | --- | --- | --- | --- |
| [audioldm2](https://huggingface.co/cvssp/audioldm2)  | 	 Text-to-audio	  | 	 350M	  | 	 1.1B	  | 	 1150k	  |
| [audioldm2-large](https://huggingface.co/cvssp/audioldm2-large)  | 	 Text-to-audio	  | 	 750M	  | 	 1.5B	  | 	 1150k	  |
| [audioldm2-music](https://huggingface.co/cvssp/audioldm2-music)  | 	 Text-to-music	  | 	 350M	  | 	 1.1B	  | 	 665k	  |


### 


 Constructing a prompt


* Descriptive prompt inputs work best: use adjectives to describe the sound (e.g. “high quality” or “clear”) and make the prompt context specific (e.g. “water stream in a forest” instead of “stream”).
* It’s best to use general terms like “cat” or “dog” instead of specific names or abstract objects the model may not be familiar with.
* Using a
 **negative prompt** 
 can significantly improve the quality of the generated waveform, by guiding the generation away from terms that correspond to poor quality audio. Try using a negative prompt of “Low quality.”


### 


 Controlling inference


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


### 


 Evaluating generated waveforms:


* The quality of the generated waveforms can vary significantly based on the seed. Try generating with different seeds until you find a satisfactory generation
* Multiple waveforms can be generated in one go: set
 `num_waveforms_per_prompt` 
 to a value greater than 1. Automatic scoring will be performed between the generated waveforms and prompt text, and the audios ranked from best to worst accordingly.



 The following example demonstrates how to construct good music generation using the aforementioned tips:
 [example](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.example) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## AudioLDM2Pipeline




### 




 class
 

 diffusers.
 

 AudioLDM2Pipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L103)



 (
 


 vae
 
 : AutoencoderKL
 




 text\_encoder
 
 : ClapModel
 




 text\_encoder\_2
 
 : T5EncoderModel
 




 projection\_model
 
 : AudioLDM2ProjectionModel
 




 language\_model
 
 : GPT2Model
 




 tokenizer
 
 : typing.Union[transformers.models.roberta.tokenization\_roberta.RobertaTokenizer, transformers.models.roberta.tokenization\_roberta\_fast.RobertaTokenizerFast]
 




 tokenizer\_2
 
 : typing.Union[transformers.models.t5.tokenization\_t5.T5Tokenizer, transformers.models.t5.tokenization\_t5\_fast.T5TokenizerFast]
 




 feature\_extractor
 
 : ClapFeatureExtractor
 




 unet
 
 : AudioLDM2UNet2DConditionModel
 




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
First frozen text-encoder. AudioLDM2 uses the joint audio-text embedding model
 [CLAP](https://huggingface.co/docs/transformers/model_doc/clap#transformers.CLAPTextModelWithProjection) 
 ,
specifically the
 [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) 
 variant. The
text branch is used to encode the text prompt to a prompt embedding. The full audio-text model is used to
rank generated waveforms against the text prompt by computing similarity scores.
* **text\_encoder\_2** 
 (
 [T5EncoderModel](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/t5#transformers.T5EncoderModel) 
 ) —
Second frozen text-encoder. AudioLDM2 uses the encoder of
 [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel) 
 , specifically the
 [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) 
 variant.
* **projection\_model** 
 (
 [AudioLDM2ProjectionModel](/docs/diffusers/v0.23.0/en/api/pipelines/audioldm2#diffusers.AudioLDM2ProjectionModel) 
 ) —
A trained model used to linearly project the hidden-states from the first and second text encoder models
and insert learned SOS and EOS token embeddings. The projected hidden-states from the two text encoders are
concatenated to give the input to the language model.
* **language\_model** 
 (
 [GPT2Model](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/gpt2#transformers.GPT2Model) 
 ) —
An auto-regressive language model used to generate a sequence of hidden-states conditioned on the projected
outputs from the two text encoders.
* **tokenizer** 
 (
 [RobertaTokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/roberta#transformers.RobertaTokenizer) 
 ) —
Tokenizer to tokenize text for the first frozen text-encoder.
* **tokenizer\_2** 
 (
 [T5Tokenizer](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/mt5#transformers.T5Tokenizer) 
 ) —
Tokenizer to tokenize text for the second frozen text-encoder.
* **feature\_extractor** 
 (
 [ClapFeatureExtractor](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/clap#transformers.ClapFeatureExtractor) 
 ) —
Feature extractor to pre-process generated audio waveforms to log-mel spectrograms for automatic scoring.
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
 to convert the mel-spectrogram latents to the final audio waveform.


 Pipeline for text-to-audio generation using AudioLDM2.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L732)



 (
 


 prompt
 
 : typing.Union[str, typing.List[str]] = None
 




 audio\_length\_in\_s
 
 : typing.Optional[float] = None
 




 num\_inference\_steps
 
 : int = 200
 




 guidance\_scale
 
 : float = 3.5
 




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
 




 generated\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_generated\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 attention\_mask
 
 : typing.Optional[torch.LongTensor] = None
 




 negative\_attention\_mask
 
 : typing.Optional[torch.LongTensor] = None
 




 max\_new\_tokens
 
 : typing.Optional[int] = None
 




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
 

[StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
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
 , defaults to 3.5) —
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
 , then automatic
scoring is performed between the generated outputs and the text prompt. This scoring ranks the
generated waveforms based on their cosine similarity with the text input in the joint text-audio
embedding space.
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
Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for spectrogram
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
* **generated\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not provided, text embeddings will be generated from
 `prompt` 
 input
argument.
* **negative\_generated\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
inputs,
 *e.g.* 
 prompt weighting. If not provided, negative\_prompt\_embeds will be computed from
 `negative_prompt` 
 input argument.
* **attention\_mask** 
 (
 `torch.LongTensor` 
 ,
 *optional* 
 ) —
Pre-computed attention mask to be applied to the
 `prompt_embeds` 
. If not provided, attention mask will
be computed from
 `prompt` 
 input argument.
* **negative\_attention\_mask** 
 (
 `torch.LongTensor` 
 ,
 *optional* 
 ) —
Pre-computed attention mask to be applied to the
 `negative_prompt_embeds` 
. If not provided, attention
mask will be computed from
 `negative_prompt` 
 input argument.
* **max\_new\_tokens** 
 (
 `int` 
 ,
 *optional* 
 , defaults to None) —
Number of new tokens to generate with the GPT2 language model. If not provided, number of tokens will
be taken from the config of the model.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 instead of a
plain tuple.
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
 

[StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is
 `True` 
 ,
 [StableDiffusionPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) 
 is returned,
otherwise a
 `tuple` 
 is returned where the first element is a list with the generated audio.
 



 The call function to the pipeline for generation.
 


 Examples:
 



```
>>> import scipy
>>> import torch
>>> from diffusers import AudioLDM2Pipeline

>>> repo_id = "cvssp/audioldm2"
>>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> # define the prompts
>>> prompt = "The sound of a hammer hitting a wooden surface."
>>> negative_prompt = "Low quality."

>>> # set the seed for generator
>>> generator = torch.Generator("cuda").manual_seed(0)

>>> # run the generation
>>> audio = pipe(
...     prompt,
...     negative_prompt=negative_prompt,
...     num_inference_steps=200,
...     audio_length_in_s=10.0,
...     num_waveforms_per_prompt=3,
...     generator=generator,
... ).audios

>>> # save the best audio sample (index 0) as a .wav file
>>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
```


#### 




 disable\_vae\_slicing




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L185)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L192)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L177)



 (
 

 )
 




 Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 




#### 




 encode\_prompt




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L270)



 (
 


 prompt
 


 device
 


 num\_waveforms\_per\_prompt
 


 do\_classifier\_free\_guidance
 


 negative\_prompt
 
 = None
 




 prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 generated\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 negative\_generated\_prompt\_embeds
 
 : typing.Optional[torch.FloatTensor] = None
 




 attention\_mask
 
 : typing.Optional[torch.LongTensor] = None
 




 negative\_attention\_mask
 
 : typing.Optional[torch.LongTensor] = None
 




 max\_new\_tokens
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

 prompt\_embeds (
 `torch.FloatTensor` 
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
* **device** 
 (
 `torch.device` 
 ) —
torch device
* **num\_waveforms\_per\_prompt** 
 (
 `int` 
 ) —
number of waveforms that should be generated per prompt
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
The prompt or prompts not to guide the audio generation. If not defined, one has to pass
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
Pre-computed text embeddings from the Flan T5 model. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not provided, text embeddings will be computed from
 `prompt` 
 input argument.
* **negative\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-computed negative text embeddings from the Flan T5 model. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not provided, negative\_prompt\_embeds will be computed from
 `negative_prompt` 
 input argument.
* **generated\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
 *e.g.* 
 prompt weighting. If not provided, text embeddings will be generated from
 `prompt` 
 input
argument.
* **negative\_generated\_prompt\_embeds** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
inputs,
 *e.g.* 
 prompt weighting. If not provided, negative\_prompt\_embeds will be computed from
 `negative_prompt` 
 input argument.
* **attention\_mask** 
 (
 `torch.LongTensor` 
 ,
 *optional* 
 ) —
Pre-computed attention mask to be applied to the
 `prompt_embeds` 
. If not provided, attention mask will
be computed from
 `prompt` 
 input argument.
* **negative\_attention\_mask** 
 (
 `torch.LongTensor` 
 ,
 *optional* 
 ) —
Pre-computed attention mask to be applied to the
 `negative_prompt_embeds` 
. If not provided, attention
mask will be computed from
 `negative_prompt` 
 input argument.
* **max\_new\_tokens** 
 (
 `int` 
 ,
 *optional* 
 , defaults to None) —
The number of new tokens to generate with the GPT2 language model.




 Returns
 




 export const metadata = 'undefined';
 

 prompt\_embeds (
 `torch.FloatTensor` 
 )
 



 export const metadata = 'undefined';
 




 Text embeddings from the Flan T5 model.
attention\_mask (
 `torch.LongTensor` 
 ):
Attention mask to be applied to the
 `prompt_embeds` 
.
generated\_prompt\_embeds (
 `torch.FloatTensor` 
 ):
Text embeddings generated from the GPT2 langauge model.
 



 Encodes the prompt into text encoder hidden states.
 


 Example:
 



```
>>> import scipy
>>> import torch
>>> from diffusers import AudioLDM2Pipeline

>>> repo_id = "cvssp/audioldm2"
>>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> # Get text embedding vectors
>>> prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
...     device="cuda",
...     do_classifier_free_guidance=True,
... )

>>> # Pass text embeddings to pipeline for text-conditional audio generation
>>> audio = pipe(
...     prompt_embeds=prompt_embeds,
...     attention_mask=attention_mask,
...     generated_prompt_embeds=generated_prompt_embeds,
...     num_inference_steps=200,
...     audio_length_in_s=10.0,
... ).audios[0]

>>> # save generated audio sample
>>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
```


#### 




 generate\_language\_model




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L229)



 (
 


 inputs\_embeds
 
 : Tensor = None
 




 max\_new\_tokens
 
 : int = 8
 




 \*\*model\_kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`inputs_embeds (` 
 torch.FloatTensor
 `of shape` 
 (batch\_size, sequence\_length, hidden\_size)`)
 




 Parameters
 




* **inputs\_embeds** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, sequence_length, hidden_size)` 
 ) —
The sequence used as a prompt for the generation.
* **max\_new\_tokens** 
 (
 `int` 
 ) —
Number of new tokens to generate.
* **model\_kwargs** 
 (
 `Dict[str, Any]` 
 ,
 *optional* 
 ) —
Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the
 `forward` 
 function of the model.




 Returns
 




 export const metadata = 'undefined';
 

`inputs_embeds (` 
 torch.FloatTensor
 `of shape` 
 (batch\_size, sequence\_length, hidden\_size)`)
 



 export const metadata = 'undefined';
 




 The sequence of generated hidden-states.
 



 Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.
 


## AudioLDM2ProjectionModel




### 




 class
 

 diffusers.
 

 AudioLDM2ProjectionModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L82)



 (
 


 text\_encoder\_dim
 


 text\_encoder\_1\_dim
 


 langauge\_model\_dim
 




 )
 


 Parameters
 




* **text\_encoder\_dim** 
 (
 `int` 
 ) —
Dimensionality of the text embeddings from the first text encoder (CLAP).
* **text\_encoder\_1\_dim** 
 (
 `int` 
 ) —
Dimensionality of the text embeddings from the second text encoder (T5 or VITS).
* **langauge\_model\_dim** 
 (
 `int` 
 ) —
Dimensionality of the text embeddings from the language model (GPT2).


 A simple linear projection model to map two text embeddings to a shared latent space. It also inserts learned
embedding vectors at the start and end of each text embedding sequence respectively. Each variable appended with
 `_1` 
 refers to that corresponding to the second text encoder. Otherwise, it is from the first.
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L111)



 (
 


 hidden\_states
 
 : typing.Optional[torch.FloatTensor] = None
 




 hidden\_states\_1
 
 : typing.Optional[torch.FloatTensor] = None
 




 attention\_mask
 
 : typing.Optional[torch.LongTensor] = None
 




 attention\_mask\_1
 
 : typing.Optional[torch.LongTensor] = None
 



 )
 


## AudioLDM2UNet2DConditionModel




### 




 class
 

 diffusers.
 

 AudioLDM2UNet2DConditionModel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L148)



 (
 


 sample\_size
 
 : typing.Optional[int] = None
 




 in\_channels
 
 : int = 4
 




 out\_channels
 
 : int = 4
 




 flip\_sin\_to\_cos
 
 : bool = True
 




 freq\_shift
 
 : int = 0
 




 down\_block\_types
 
 : typing.Tuple[str] = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
 




 mid\_block\_type
 
 : typing.Optional[str] = 'UNetMidBlock2DCrossAttn'
 




 up\_block\_types
 
 : typing.Tuple[str] = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
 




 only\_cross\_attention
 
 : typing.Union[bool, typing.Tuple[bool]] = False
 




 block\_out\_channels
 
 : typing.Tuple[int] = (320, 640, 1280, 1280)
 




 layers\_per\_block
 
 : typing.Union[int, typing.Tuple[int]] = 2
 




 downsample\_padding
 
 : int = 1
 




 mid\_block\_scale\_factor
 
 : float = 1
 




 act\_fn
 
 : str = 'silu'
 




 norm\_num\_groups
 
 : typing.Optional[int] = 32
 




 norm\_eps
 
 : float = 1e-05
 




 cross\_attention\_dim
 
 : typing.Union[int, typing.Tuple[int]] = 1280
 




 transformer\_layers\_per\_block
 
 : typing.Union[int, typing.Tuple[int]] = 1
 




 attention\_head\_dim
 
 : typing.Union[int, typing.Tuple[int]] = 8
 




 num\_attention\_heads
 
 : typing.Union[int, typing.Tuple[int], NoneType] = None
 




 use\_linear\_projection
 
 : bool = False
 




 class\_embed\_type
 
 : typing.Optional[str] = None
 




 num\_class\_embeds
 
 : typing.Optional[int] = None
 




 upcast\_attention
 
 : bool = False
 




 resnet\_time\_scale\_shift
 
 : str = 'default'
 




 time\_embedding\_type
 
 : str = 'positional'
 




 time\_embedding\_dim
 
 : typing.Optional[int] = None
 




 time\_embedding\_act\_fn
 
 : typing.Optional[str] = None
 




 timestep\_post\_act
 
 : typing.Optional[str] = None
 




 time\_cond\_proj\_dim
 
 : typing.Optional[int] = None
 




 conv\_in\_kernel
 
 : int = 3
 




 conv\_out\_kernel
 
 : int = 3
 




 projection\_class\_embeddings\_input\_dim
 
 : typing.Optional[int] = None
 




 class\_embeddings\_concat
 
 : bool = False
 



 )
 


 Parameters
 




* **sample\_size** 
 (
 `int` 
 or
 `Tuple[int, int]` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Height and width of input/output sample.
* **in\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) — Number of channels in the input sample.
* **out\_channels** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 4) — Number of channels in the output.
* **flip\_sin\_to\_cos** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to flip the sin to cos in the time embedding.
* **freq\_shift** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 0) — The frequency shift to apply to the time embedding.
* **down\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")` 
 ) —
The tuple of downsample blocks to use.
* **mid\_block\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"UNetMidBlock2DCrossAttn"` 
 ) —
Block type for middle of UNet, it can only be
 `UNetMidBlock2DCrossAttn` 
 for AudioLDM2.
* **up\_block\_types** 
 (
 `Tuple[str]` 
 ,
 *optional* 
 , defaults to
 `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")` 
 ) —
The tuple of upsample blocks to use.
* **only\_cross\_attention** 
 (
 `bool` 
 or
 `Tuple[bool]` 
 ,
 *optional* 
 , default to
 `False` 
 ) —
Whether to include self-attention in the basic transformer blocks, see
 `BasicTransformerBlock` 
.
* **block\_out\_channels** 
 (
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to
 `(320, 640, 1280, 1280)` 
 ) —
The tuple of output channels for each block.
* **layers\_per\_block** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 2) — The number of layers per block.
* **downsample\_padding** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 1) — The padding to use for the downsampling convolution.
* **mid\_block\_scale\_factor** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1.0) — The scale factor to use for the mid block.
* **act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"silu"` 
 ) — The activation function to use.
* **norm\_num\_groups** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 32) — The number of groups to use for the normalization.
If
 `None` 
 , normalization and activation layers is skipped in post-processing.
* **norm\_eps** 
 (
 `float` 
 ,
 *optional* 
 , defaults to 1e-5) — The epsilon to use for the normalization.
* **cross\_attention\_dim** 
 (
 `int` 
 or
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to 1280) —
The dimension of the cross attention features.
* **transformer\_layers\_per\_block** 
 (
 `int` 
 or
 `Tuple[int]` 
 ,
 *optional* 
 , defaults to 1) —
The number of transformer blocks of type
 `BasicTransformerBlock` 
. Only relevant for
 `CrossAttnDownBlock2D` 
 ,
 `CrossAttnUpBlock2D` 
 ,
 `UNetMidBlock2DCrossAttn` 
.
* **attention\_head\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to 8) — The dimension of the attention heads.
* **num\_attention\_heads** 
 (
 `int` 
 ,
 *optional* 
 ) —
The number of attention heads. If not defined, defaults to
 `attention_head_dim`
* **resnet\_time\_scale\_shift** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `"default"` 
 ) — Time scale shift config
for ResNet blocks (see
 `ResnetBlock2D` 
 ). Choose from
 `default` 
 or
 `scale_shift` 
.
* **class\_embed\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The type of class embedding to use which is ultimately summed with the time embeddings. Choose from
 `None` 
 ,
 `"timestep"` 
 ,
 `"identity"` 
 ,
 `"projection"` 
 , or
 `"simple_projection"` 
.
* **num\_class\_embeds** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Input dimension of the learnable embedding matrix to be projected to
 `time_embed_dim` 
 , when performing
class conditioning with
 `class_embed_type` 
 equal to
 `None` 
.
* **time\_embedding\_type** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `positional` 
 ) —
The type of position embedding to use for timesteps. Choose from
 `positional` 
 or
 `fourier` 
.
* **time\_embedding\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
An optional override for the dimension of the projected time embedding.
* **time\_embedding\_act\_fn** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
Optional activation function to use only once on the time embeddings before they are passed to the rest of
the UNet. Choose from
 `silu` 
 ,
 `mish` 
 ,
 `gelu` 
 , and
 `swish` 
.
* **timestep\_post\_act** 
 (
 `str` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The second activation function to use in timestep embedding. Choose from
 `silu` 
 ,
 `mish` 
 and
 `gelu` 
.
* **time\_cond\_proj\_dim** 
 (
 `int` 
 ,
 *optional* 
 , defaults to
 `None` 
 ) —
The dimension of
 `cond_proj` 
 layer in the timestep embedding.
* **conv\_in\_kernel** 
 (
 `int` 
 ,
 *optional* 
 , default to
 `3` 
 ) — The kernel size of
 `conv_in` 
 layer.
* **conv\_out\_kernel** 
 (
 `int` 
 ,
 *optional* 
 , default to
 `3` 
 ) — The kernel size of
 `conv_out` 
 layer.
* **projection\_class\_embeddings\_input\_dim** 
 (
 `int` 
 ,
 *optional* 
 ) — The dimension of the
 `class_labels` 
 input when
 `class_embed_type="projection"` 
. Required when
 `class_embed_type="projection"` 
.
* **class\_embeddings\_concat** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) — Whether to concatenate the time
embeddings with the class embeddings.


 A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
shaped output. Compared to the vanilla
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 , this variant optionally includes an additional
self-attention layer in each Transformer block, as well as multiple cross-attention layers. It also allows for up
to two cross-attention embeddings,
 `encoder_hidden_states` 
 and
 `encoder_hidden_states_1` 
.
 



 This model inherits from
 [ModelMixin](/docs/diffusers/v0.23.0/en/api/models/overview#diffusers.ModelMixin) 
. Check the superclass documentation for it’s generic methods implemented
for all models (such as downloading or saving).
 



#### 




 forward




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L664)



 (
 


 sample
 
 : FloatTensor
 




 timestep
 
 : typing.Union[torch.Tensor, float, int]
 




 encoder\_hidden\_states
 
 : Tensor
 




 class\_labels
 
 : typing.Optional[torch.Tensor] = None
 




 timestep\_cond
 
 : typing.Optional[torch.Tensor] = None
 




 attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 cross\_attention\_kwargs
 
 : typing.Union[typing.Dict[str, typing.Any], NoneType] = None
 




 encoder\_attention\_mask
 
 : typing.Optional[torch.Tensor] = None
 




 return\_dict
 
 : bool = True
 




 encoder\_hidden\_states\_1
 
 : typing.Optional[torch.Tensor] = None
 




 encoder\_attention\_mask\_1
 
 : typing.Optional[torch.Tensor] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 or
 `tuple` 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) —
The noisy input tensor with the following shape
 `(batch, channel, height, width)` 
.
* **timestep** 
 (
 `torch.FloatTensor` 
 or
 `float` 
 or
 `int` 
 ) — The number of timesteps to denoise an input.
* **encoder\_hidden\_states** 
 (
 `torch.FloatTensor` 
 ) —
The encoder hidden states with shape
 `(batch, sequence_length, feature_dim)` 
.
* **encoder\_attention\_mask** 
 (
 `torch.Tensor` 
 ) —
A cross-attention mask of shape
 `(batch, sequence_length)` 
 is applied to
 `encoder_hidden_states` 
. If
 `True` 
 the mask is kept, otherwise if
 `False` 
 it is discarded. Mask will be converted into a bias,
which adds large negative values to the attention scores corresponding to “discard” tokens.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 instead of a plain
tuple.
* **cross\_attention\_kwargs** 
 (
 `dict` 
 ,
 *optional* 
 ) —
A kwargs dictionary that if specified is passed along to the
 `AttnProcessor` 
.
* **encoder\_hidden\_states\_1** 
 (
 `torch.FloatTensor` 
 ,
 *optional* 
 ) —
A second set of encoder hidden states with shape
 `(batch, sequence_length_2, feature_dim_2)` 
. Can be
used to condition the model on a different set of embeddings to
 `encoder_hidden_states` 
.
* **encoder\_attention\_mask\_1** 
 (
 `torch.Tensor` 
 ,
 *optional* 
 ) —
A cross-attention mask of shape
 `(batch, sequence_length_2)` 
 is applied to
 `encoder_hidden_states_1` 
.
If
 `True` 
 the mask is kept, otherwise if
 `False` 
 it is discarded. Mask will be converted into a bias,
which adds large negative values to the attention scores corresponding to “discard” tokens.




 Returns
 




 export const metadata = 'undefined';
 

[UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If
 `return_dict` 
 is True, an
 [UNet2DConditionOutput](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.models.unet_2d_condition.UNet2DConditionOutput) 
 is returned, otherwise
a
 `tuple` 
 is returned where the first element is the sample tensor.
 



 The
 [AudioLDM2UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/pipelines/audioldm2#diffusers.AudioLDM2UNet2DConditionModel) 
 forward method.
 


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