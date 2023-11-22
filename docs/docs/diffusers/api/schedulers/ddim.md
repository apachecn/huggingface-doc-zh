# DDIMScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/ddim>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/ddim>



[Denoising Diffusion Implicit Models](https://huggingface.co/papers/2010.02502) 
 (DDIM) by Jiaming Song, Chenlin Meng and Stefano Ermon.
 



 The abstract from the paper is:
 



*Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training,
yet they require simulating a Markov chain for many steps to produce a sample.
To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models
with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process.
We construct a class of non-Markovian diffusion processes that lead to the same training objective, but whose reverse process can be much faster to sample from.
We empirically demonstrate that DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs, allow us to trade off
computation for sample quality, and can perform semantically meaningful image interpolation directly in the latent space.* 




 The original codebase of this paper can be found at
 [ermongroup/ddim](https://github.com/ermongroup/ddim) 
 , and you can contact the author on
 [tsong.me](https://tsong.me/) 
.
 


## Tips




 The paper
 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) 
 claims that a mismatch between the training and inference settings leads to suboptimal inference generation results for Stable Diffusion. To fix this, the authors propose:
 




 🧪 This is an experimental feature!
 



1. rescale the noise schedule to enforce zero terminal signal-to-noise ratio (SNR)



```
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True)
```


2. train a model with
 `v_prediction` 
 (add the following argument to the
 [train\_text\_to\_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) 
 or
 [train\_text\_to\_image\_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) 
 scripts)



```
--prediction_type="v\_prediction"
```


3. change the sampler to always start from the last timestep



```
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
```


4. rescale classifier-free guidance to prevent over-exposure



```
image = pipeline(prompt, guidance_rescale=0.7).images[0]
```



 For example:
 



```
from diffusers import DiffusionPipeline, DDIMScheduler

pipe = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipe.to("cuda")

prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"
image = pipeline(prompt, guidance_rescale=0.7).images[0]
```


## DDIMScheduler




### 




 class
 

 diffusers.
 

 DDIMScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim.py#L131)



 (
 


 num\_train\_timesteps
 
 : int = 1000
 




 beta\_start
 
 : float = 0.0001
 




 beta\_end
 
 : float = 0.02
 




 beta\_schedule
 
 : str = 'linear'
 




 trained\_betas
 
 : typing.Union[numpy.ndarray, typing.List[float], NoneType] = None
 




 clip\_sample
 
 : bool = True
 




 set\_alpha\_to\_one
 
 : bool = True
 




 steps\_offset
 
 : int = 0
 




 prediction\_type
 
 : str = 'epsilon'
 




 thresholding
 
 : bool = False
 




 dynamic\_thresholding\_ratio
 
 : float = 0.995
 




 clip\_sample\_range
 
 : float = 1.0
 




 sample\_max\_value
 
 : float = 1.0
 




 timestep\_spacing
 
 : str = 'leading'
 




 rescale\_betas\_zero\_snr
 
 : bool = False
 



 )
 


 Parameters
 




* **num\_train\_timesteps** 
 (
 `int` 
 , defaults to 1000) —
The number of diffusion steps to train the model.
* **beta\_start** 
 (
 `float` 
 , defaults to 0.0001) —
The starting
 `beta` 
 value of inference.
* **beta\_end** 
 (
 `float` 
 , defaults to 0.02) —
The final
 `beta` 
 value.
* **beta\_schedule** 
 (
 `str` 
 , defaults to
 `"linear"` 
 ) —
The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
 `linear` 
 ,
 `scaled_linear` 
 , or
 `squaredcos_cap_v2` 
.
* **trained\_betas** 
 (
 `np.ndarray` 
 ,
 *optional* 
 ) —
Pass an array of betas directly to the constructor to bypass
 `beta_start` 
 and
 `beta_end` 
.
* **clip\_sample** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Clip the predicted sample for numerical stability.
* **clip\_sample\_range** 
 (
 `float` 
 , defaults to 1.0) —
The maximum magnitude for sample clipping. Valid only when
 `clip_sample=True` 
.
* **set\_alpha\_to\_one** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
there is no previous alpha. When this option is
 `True` 
 the previous alpha product is fixed to
 `1` 
 ,
otherwise it uses the alpha value at step 0.
* **steps\_offset** 
 (
 `int` 
 , defaults to 0) —
An offset added to the inference steps. You can use a combination of
 `offset=1` 
 and
 `set_alpha_to_one=False` 
 to make the last step use step 0 for the previous alpha product like in Stable
Diffusion.
* **prediction\_type** 
 (
 `str` 
 , defaults to
 `epsilon` 
 ,
 *optional* 
 ) —
Prediction type of the scheduler function; can be
 `epsilon` 
 (predicts the noise of the diffusion process),
 `sample` 
 (directly predicts the noisy sample
 `) or` 
 v\_prediction` (see section 2.4 of
 [Imagen
Video](https://imagen.research.google/video/paper.pdf) 
 paper).
* **thresholding** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Whether to use the “dynamic thresholding” method. This is unsuitable for latent-space diffusion models such
as Stable Diffusion.
* **dynamic\_thresholding\_ratio** 
 (
 `float` 
 , defaults to 0.995) —
The ratio for the dynamic thresholding method. Valid only when
 `thresholding=True` 
.
* **sample\_max\_value** 
 (
 `float` 
 , defaults to 1.0) —
The threshold value for dynamic thresholding. Valid only when
 `thresholding=True` 
.
* **timestep\_spacing** 
 (
 `str` 
 , defaults to
 `"leading"` 
 ) —
The way the timesteps should be scaled. Refer to Table 2 of the
 [Common Diffusion Noise Schedules and
Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) 
 for more information.
* **rescale\_betas\_zero\_snr** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
dark samples instead of limiting it to samples with medium brightness. Loosely related to
 [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506)
.


`DDIMScheduler` 
 extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
non-Markovian guidance.
 



 This model inherits from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
. Check the superclass documentation for the generic
methods the library implements for all schedulers such as loading and saving.
 



#### 




 scale\_model\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim.py#L240)



 (
 


 sample
 
 : FloatTensor
 




 timestep
 
 : typing.Optional[int] = None
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) —
The input sample.
* **timestep** 
 (
 `int` 
 ,
 *optional* 
 ) —
The current timestep in the diffusion chain.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 A scaled input sample.
 



 Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
current timestep.
 




#### 




 set\_timesteps




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim.py#L301)



 (
 


 num\_inference\_steps
 
 : int
 




 device
 
 : typing.Union[str, torch.device] = None
 



 )
 


 Parameters
 




* **num\_inference\_steps** 
 (
 `int` 
 ) —
The number of diffusion steps used when generating samples with a pre-trained model.


 Sets the discrete timesteps used for the diffusion chain (to be run before inference).
 




#### 




 step




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim.py#L346)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : int
 




 sample
 
 : FloatTensor
 




 eta
 
 : float = 0.0
 




 use\_clipped\_model\_output
 
 : bool = False
 




 generator
 
 = None
 




 variance\_noise
 
 : typing.Optional[torch.FloatTensor] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~schedulers.scheduling_utils.DDIMSchedulerOutput` 
 or
 `tuple` 


 Parameters
 




* **model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The direct output from learned diffusion model.
* **timestep** 
 (
 `float` 
 ) —
The current discrete timestep in the diffusion chain.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by the diffusion process.
* **eta** 
 (
 `float` 
 ) —
The weight of noise for added noise in diffusion step.
* **use\_clipped\_model\_output** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
If
 `True` 
 , computes “corrected”
 `model_output` 
 from the clipped predicted original sample. Necessary
because predicted original sample is clipped to [-1, 1] when
 `self.config.clip_sample` 
 is
 `True` 
. If no
clipping has happened, “corrected”
 `model_output` 
 would coincide with the one provided as input and
 `use_clipped_model_output` 
 has no effect.
* **generator** 
 (
 `torch.Generator` 
 ,
 *optional* 
 ) —
A random number generator.
* **variance\_noise** 
 (
 `torch.FloatTensor` 
 ) —
Alternative to generating noise with
 `generator` 
 by directly providing the noise for the variance
itself. Useful for methods such as
 `CycleDiffusion` 
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
 [DDIMSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.schedulers.scheduling_ddim.DDIMSchedulerOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

`~schedulers.scheduling_utils.DDIMSchedulerOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [DDIMSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.schedulers.scheduling_ddim.DDIMSchedulerOutput) 
 is returned, otherwise a
tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).
 


## DDIMSchedulerOutput




### 




 class
 

 diffusers.schedulers.scheduling\_ddim.
 

 DDIMSchedulerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim.py#L33)



 (
 


 prev\_sample
 
 : FloatTensor
 




 pred\_original\_sample
 
 : typing.Optional[torch.FloatTensor] = None
 



 )
 


 Parameters
 




* **prev\_sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 for images) —
Computed sample
 `(x_{t-1})` 
 of previous timestep.
 `prev_sample` 
 should be used as next model input in the
denoising loop.
* **pred\_original\_sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 for images) —
The predicted denoised sample
 `(x_{0})` 
 based on the model output from the current timestep.
 `pred_original_sample` 
 can be used to preview progress or for guidance.


 Output class for the scheduler’s
 `step` 
 function output.