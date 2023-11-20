# Audio Diffusion

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/audio_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/audio_diffusion>



[Audio Diffusion](https://github.com/teticio/audio-diffusion) 
 is by Robert Dargavel Smith, and it leverages the recent advances in image generation from diffusion models by converting audio samples to and from Mel spectrogram images.
 



 The original codebase, training scripts and example notebooks can be found at
 [teticio/audio-diffusion](https://github.com/teticio/audio-diffusion) 
.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## AudioDiffusionPipeline




### 




 class
 

 diffusers.
 

 AudioDiffusionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/pipeline_audio_diffusion.py#L30)



 (
 


 vqvae
 
 : AutoencoderKL
 




 unet
 
 : UNet2DConditionModel
 




 mel
 
 : Mel
 




 scheduler
 
 : typing.Union[diffusers.schedulers.scheduling\_ddim.DDIMScheduler, diffusers.schedulers.scheduling\_ddpm.DDPMScheduler]
 



 )
 


 Parameters
 




* **vqae** 
 (
 [AutoencoderKL](/docs/diffusers/v0.23.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) 
 ) —
Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
* **unet** 
 (
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 ) —
A
 `UNet2DConditionModel` 
 to denoise the encoded image latents.
* **mel** 
 (
 [Mel](/docs/diffusers/v0.23.0/en/api/pipelines/audio_diffusion#diffusers.Mel) 
 ) —
Transform audio into a spectrogram.
* **scheduler** 
 (
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 or
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded image latents. Can be one of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 or
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
.


 Pipeline for audio diffusion.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).
 



#### 




 \_\_call\_\_




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/pipeline_audio_diffusion.py#L70)



 (
 


 batch\_size
 
 : int = 1
 




 audio\_file
 
 : str = None
 




 raw\_audio
 
 : ndarray = None
 




 slice
 
 : int = 0
 




 start\_step
 
 : int = 0
 




 steps
 
 : int = None
 




 generator
 
 : Generator = None
 




 mask\_start\_secs
 
 : float = 0
 




 mask\_end\_secs
 
 : float = 0
 




 step\_generator
 
 : Generator = None
 




 eta
 
 : float = 0
 




 noise
 
 : Tensor = None
 




 encoding
 
 : Tensor = None
 




 return\_dict
 
 = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`List[PIL Image]` 


 Parameters
 




* **batch\_size** 
 (
 `int` 
 ) —
Number of samples to generate.
* **audio\_file** 
 (
 `str` 
 ) —
An audio file that must be on disk due to
 [Librosa](https://librosa.org/) 
 limitation.
* **raw\_audio** 
 (
 `np.ndarray` 
 ) —
The raw audio file as a NumPy array.
* **slice** 
 (
 `int` 
 ) —
Slice number of audio to convert.
* **start\_step** 
 (int) —
Step to start diffusion from.
* **steps** 
 (
 `int` 
 ) —
Number of denoising steps (defaults to
 `50` 
 for DDIM and
 `1000` 
 for DDPM).
* **generator** 
 (
 `torch.Generator` 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 to make
generation deterministic.
* **mask\_start\_secs** 
 (
 `float` 
 ) —
Number of seconds of audio to mask (not generate) at start.
* **mask\_end\_secs** 
 (
 `float` 
 ) —
Number of seconds of audio to mask (not generate) at end.
* **step\_generator** 
 (
 `torch.Generator` 
 ) —
A
 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 used to denoise.
None
* **eta** 
 (
 `float` 
 ) —
Corresponds to parameter eta (η) from the
 [DDIM](https://arxiv.org/abs/2010.02502) 
 paper. Only applies
to the
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
 , and is ignored in other schedulers.
* **noise** 
 (
 `torch.Tensor` 
 ) —
A noise tensor of shape
 `(batch_size, 1, height, width)` 
 or
 `None` 
.
* **encoding** 
 (
 `torch.Tensor` 
 ) —
A tensor for
 [UNet2DConditionModel](/docs/diffusers/v0.23.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel) 
 of shape
 `(batch_size, seq_length, cross_attention_dim)` 
.
* **return\_dict** 
 (
 `bool` 
 ) —
Whether or not to return a
 [AudioPipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/dance_diffusion#diffusers.AudioPipelineOutput) 
 ,
 [ImagePipelineOutput](/docs/diffusers/v0.23.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) 
 or a plain tuple.




 Returns
 




 export const metadata = 'undefined';
 

`List[PIL Image]` 




 export const metadata = 'undefined';
 




 A list of Mel spectrograms (
 `float` 
 ,
 `List[np.ndarray]` 
 ) with the sample rate and raw audio.
 



 The call function to the pipeline for generation.
 



 Examples:
 


 For audio diffusion:
 



```
import torch
from IPython.display import Audio
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-256").to(device)

output = pipe()
display(output.images[0])
display(Audio(output.audios[0], rate=mel.get_sample_rate()))
```



 For latent audio diffusion:
 



```
import torch
from IPython.display import Audio
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("teticio/latent-audio-diffusion-256").to(device)

output = pipe()
display(output.images[0])
display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))
```



 For other tasks like variation, inpainting, outpainting, etc:
 



```
output = pipe(
    raw_audio=output.audios[0, 0],
    start_step=int(pipe.get_default_steps() / 2),
    mask_start_secs=1,
    mask_end_secs=1,
)
display(output.images[0])
display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))
```


#### 




 encode




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/pipeline_audio_diffusion.py#L270)



 (
 


 images
 
 : typing.List[PIL.Image.Image]
 




 steps
 
 : int = 50
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`np.ndarray` 


 Parameters
 




* **images** 
 (
 `List[PIL Image]` 
 ) —
List of images to encode.
* **steps** 
 (
 `int` 
 ) —
Number of encoding steps to perform (defaults to
 `50` 
 ).




 Returns
 




 export const metadata = 'undefined';
 

`np.ndarray` 




 export const metadata = 'undefined';
 




 A noise tensor of shape
 `(batch_size, 1, height, width)` 
.
 



 Reverse the denoising step process to recover a noisy image from the generated image.
 




#### 




 get\_default\_steps




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/pipeline_audio_diffusion.py#L61)



 (
 

 )
 

 →
 



 export const metadata = 'undefined';
 

`int` 



 Returns
 




 export const metadata = 'undefined';
 

`int` 




 export const metadata = 'undefined';
 




 The number of steps.
 



 Returns default number of steps recommended for inference.
 




#### 




 slerp




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/pipeline_audio_diffusion.py#L311)



 (
 


 x0
 
 : Tensor
 




 x1
 
 : Tensor
 




 alpha
 
 : float
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.Tensor` 


 Parameters
 




* **x0** 
 (
 `torch.Tensor` 
 ) —
The first tensor to interpolate between.
* **x1** 
 (
 `torch.Tensor` 
 ) —
Second tensor to interpolate between.
* **alpha** 
 (
 `float` 
 ) —
Interpolation between 0 and 1




 Returns
 




 export const metadata = 'undefined';
 

`torch.Tensor` 




 export const metadata = 'undefined';
 




 The interpolated tensor.
 



 Spherical Linear intERPolation.
 


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
 


## ImagePipelineOutput




### 




 class
 

 diffusers.
 

 ImagePipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L110)



 (
 


 images
 
 : typing.Union[typing.List[PIL.Image.Image], numpy.ndarray]
 



 )
 


 Parameters
 




* **images** 
 (
 `List[PIL.Image.Image]` 
 or
 `np.ndarray` 
 ) —
List of denoised PIL images of length
 `batch_size` 
 or NumPy array of shape
 `(batch_size, height, width, num_channels)` 
.


 Output class for image pipelines.
 


## Mel




### 




 class
 

 diffusers.
 

 Mel




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L37)



 (
 


 x\_res
 
 : int = 256
 




 y\_res
 
 : int = 256
 




 sample\_rate
 
 : int = 22050
 




 n\_fft
 
 : int = 2048
 




 hop\_length
 
 : int = 512
 




 top\_db
 
 : int = 80
 




 n\_iter
 
 : int = 32
 



 )
 


 Parameters
 




* **x\_res** 
 (
 `int` 
 ) —
x resolution of spectrogram (time).
* **y\_res** 
 (
 `int` 
 ) —
y resolution of spectrogram (frequency bins).
* **sample\_rate** 
 (
 `int` 
 ) —
Sample rate of audio.
* **n\_fft** 
 (
 `int` 
 ) —
Number of Fast Fourier Transforms.
* **hop\_length** 
 (
 `int` 
 ) —
Hop length (a higher number is recommended if
 `y_res` 
 < 256).
* **top\_db** 
 (
 `int` 
 ) —
Loudest decibel value.
* **n\_iter** 
 (
 `int` 
 ) —
Number of iterations for Griffin-Lim Mel inversion.


#### 




 audio\_slice\_to\_image




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L143)



 (
 


 slice
 
 : int
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`PIL Image` 


 Parameters
 




* **slice** 
 (
 `int` 
 ) —
Slice number of audio to convert (out of
 `get_number_of_slices()` 
 ).




 Returns
 




 export const metadata = 'undefined';
 

`PIL Image` 




 export const metadata = 'undefined';
 




 A grayscale image of
 `x_res x y_res` 
.
 



 Convert slice of audio to spectrogram.
 




#### 




 get\_audio\_slice




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L121)



 (
 


 slice
 
 : int = 0
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`np.ndarray` 


 Parameters
 




* **slice** 
 (
 `int` 
 ) —
Slice number of audio (out of
 `get_number_of_slices()` 
 ).




 Returns
 




 export const metadata = 'undefined';
 

`np.ndarray` 




 export const metadata = 'undefined';
 




 The audio slice as a NumPy array.
 



 Get slice of audio.
 




#### 




 get\_number\_of\_slices




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L112)



 (
 

 )
 

 →
 



 export const metadata = 'undefined';
 

`int` 



 Returns
 




 export const metadata = 'undefined';
 

`int` 




 export const metadata = 'undefined';
 




 Number of spectograms audio can be sliced into.
 



 Get number of slices in audio.
 




#### 




 get\_sample\_rate




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L134)



 (
 

 )
 

 →
 



 export const metadata = 'undefined';
 

`int` 



 Returns
 




 export const metadata = 'undefined';
 

`int` 




 export const metadata = 'undefined';
 




 Sample rate of audio.
 



 Get sample rate.
 




#### 




 image\_to\_audio




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L162)



 (
 


 image
 
 : Image
 



 )
 

 →
 



 export const metadata = 'undefined';
 

 audio (
 `np.ndarray` 
 )
 




 Parameters
 




* **image** 
 (
 `PIL Image` 
 ) —
An grayscale image of
 `x_res x y_res` 
.




 Returns
 




 export const metadata = 'undefined';
 

 audio (
 `np.ndarray` 
 )
 



 export const metadata = 'undefined';
 




 The audio as a NumPy array.
 



 Converts spectrogram to audio.
 




#### 




 load\_audio




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L94)



 (
 


 audio\_file
 
 : str = None
 




 raw\_audio
 
 : ndarray = None
 



 )
 


 Parameters
 




* **audio\_file** 
 (
 `str` 
 ) —
An audio file that must be on disk due to
 [Librosa](https://librosa.org/) 
 limitation.
* **raw\_audio** 
 (
 `np.ndarray` 
 ) —
The raw audio file as a NumPy array.


 Load audio.
 




#### 




 set\_resolution




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/audio_diffusion/mel.py#L80)



 (
 


 x\_res
 
 : int
 




 y\_res
 
 : int
 



 )
 


 Parameters
 




* **x\_res** 
 (
 `int` 
 ) —
x resolution of spectrogram (time).
* **y\_res** 
 (
 `int` 
 ) —
y resolution of spectrogram (frequency bins).


 Set resolution.