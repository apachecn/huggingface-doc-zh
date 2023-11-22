# DDIMInverseScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/ddim_inverse>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/ddim_inverse>



`DDIMInverseScheduler` 
 is the inverted scheduler from
 [Denoising Diffusion Implicit Models](https://huggingface.co/papers/2010.02502) 
 (DDIM) by Jiaming Song, Chenlin Meng and Stefano Ermon.
The implementation is mostly based on the DDIM inversion definition from
 [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://huggingface.co/papers/2211.09794.pdf) 
.
 


## DDIMInverseScheduler




### 




 class
 

 diffusers.
 

 DDIMInverseScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim_inverse.py#L130)



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
 




 clip\_sample\_range
 
 : float = 1.0
 




 timestep\_spacing
 
 : str = 'leading'
 




 rescale\_betas\_zero\_snr
 
 : bool = False
 




 \*\*kwargs
 




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
 the previous alpha product is fixed to 0, otherwise
it uses the alpha value at step
 `num_train_timesteps - 1` 
.
* **steps\_offset** 
 (
 `int` 
 , defaults to 0) —
An offset added to the inference steps. You can use a combination of
 `offset=1` 
 and
 `set_alpha_to_one=False` 
 to make the last step use
 `num_train_timesteps - 1` 
 for the previous alpha
product.
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


`DDIMInverseScheduler` 
 is the reverse scheduler of
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
.
 



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim_inverse.py#L238)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim_inverse.py#L255)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddim_inverse.py#L293)



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
 




 variance\_noise
 
 : typing.Optional[torch.FloatTensor] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~schedulers.scheduling_ddim_inverse.DDIMInverseSchedulerOutput` 
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
 `~schedulers.scheduling_ddim_inverse.DDIMInverseSchedulerOutput` 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

`~schedulers.scheduling_ddim_inverse.DDIMInverseSchedulerOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 `~schedulers.scheduling_ddim_inverse.DDIMInverseSchedulerOutput` 
 is
returned, otherwise a tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).