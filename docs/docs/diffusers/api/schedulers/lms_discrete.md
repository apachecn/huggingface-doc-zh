# LMSDiscreteScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/lms_discrete>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete>



`LMSDiscreteScheduler` 
 is a linear multistep scheduler for discrete beta schedules. The scheduler is ported from and created by
 [Katherine Crowson](https://github.com/crowsonkb/) 
 , and the original implementation can be found at
 [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181) 
.
 


## LMSDiscreteScheduler




### 




 class
 

 diffusers.
 

 LMSDiscreteScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L92)



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
 




 use\_karras\_sigmas
 
 : typing.Optional[bool] = False
 




 prediction\_type
 
 : str = 'epsilon'
 




 timestep\_spacing
 
 : str = 'linspace'
 




 steps\_offset
 
 : int = 0
 



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
 or
 `scaled_linear` 
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
* **use\_karras\_sigmas** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If
 `True` 
 ,
the sigmas are determined according to a sequence of noise levels {σi}.
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
 `"linspace"` 
 ) —
The way the timesteps should be scaled. Refer to Table 2 of the
 [Common Diffusion Noise Schedules and
Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) 
 for more information.
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


 A linear multistep scheduler for discrete beta schedules.
 



 This model inherits from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
. Check the superclass documentation for the generic
methods the library implements for all schedulers such as loading and saving.
 



#### 




 get\_lms\_coefficient




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L215)



 (
 


 order
 


 t
 


 current\_order
 




 )
 


 Parameters
 




* **order** 
 () —
* **t** 
 () —
* **current\_order** 
 () —


 Compute the linear multistep coefficient.
 




#### 




 scale\_model\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L189)



 (
 


 sample
 
 : FloatTensor
 




 timestep
 
 : typing.Union[float, torch.FloatTensor]
 



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
 `float` 
 or
 `torch.FloatTensor` 
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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L237)



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
* **device** 
 (
 `str` 
 or
 `torch.device` 
 ,
 *optional* 
 ) —
The device to which the timesteps should be moved to. If
 `None` 
 , the timesteps are not moved.


 Sets the discrete timesteps used for the diffusion chain (to be run before inference).
 




#### 




 step




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L343)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : typing.Union[float, torch.FloatTensor]
 




 sample
 
 : FloatTensor
 




 order
 
 : int = 4
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[SchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/dpm_discrete#diffusers.schedulers.scheduling_utils.SchedulerOutput) 
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
 or
 `torch.FloatTensor` 
 ) —
The current discrete timestep in the diffusion chain.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by the diffusion process.
* **order** 
 (
 `int` 
 , defaults to 4) —
The order of the linear multistep method.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [SchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/dpm_discrete#diffusers.schedulers.scheduling_utils.SchedulerOutput) 
 or tuple.




 Returns
 




 export const metadata = 'undefined';
 

[SchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/dpm_discrete#diffusers.schedulers.scheduling_utils.SchedulerOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [SchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/dpm_discrete#diffusers.schedulers.scheduling_utils.SchedulerOutput) 
 is returned, otherwise a
tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).
 


## LMSDiscreteSchedulerOutput




### 




 class
 

 diffusers.schedulers.scheduling\_lms\_discrete.
 

 LMSDiscreteSchedulerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L30)



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