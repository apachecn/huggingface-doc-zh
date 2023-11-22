# DEISMultistepScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/deis>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/deis>



 Diffusion Exponential Integrator Sampler (DEIS) is proposed in
 [Fast Sampling of Diffusion Models with Exponential Integrator](https://huggingface.co/papers/2204.13902) 
 by Qinsheng Zhang and Yongxin Chen.
 `DEISMultistepScheduler` 
 is a fast high order solver for diffusion ordinary differential equations (ODEs).
 



 This implementation modifies the polynomial fitting formula in log-rho space instead of the original linear
 `t` 
 space in the DEIS paper. The modification enjoys closed-form coefficients for exponential multistep update instead of replying on the numerical solver.
 



 The abstract from the paper is:
 



*The past few years have witnessed the great success of Diffusion models~(DMs) in generating high-fidelity samples in generative modeling tasks. A major limitation of the DM is its notoriously slow sampling procedure which normally requires hundreds to thousands of time discretization steps of the learned diffusion process to reach the desired accuracy. Our goal is to develop a fast sampling method for DMs with a much less number of steps while retaining high sample quality. To this end, we systematically analyze the sampling procedure in DMs and identify key factors that affect the sample quality, among which the method of discretization is most crucial. By carefully examining the learned diffusion process, we propose Diffusion Exponential Integrator Sampler~(DEIS). It is based on the Exponential Integrator designed for discretizing ordinary differential equations (ODEs) and leverages a semilinear structure of the learned diffusion process to reduce the discretization error. The proposed method can be applied to any DMs and can generate high-fidelity samples in as few as 10 steps. In our experiments, it takes about 3 minutes on one A6000 GPU to generate 50k images from CIFAR10. Moreover, by directly using pre-trained DMs, we achieve the state-of-art sampling performance when the number of score function evaluation~(NFE) is limited, e.g., 4.17 FID with 10 NFEs, 3.37 FID, and 9.74 IS with only 15 NFEs on CIFAR10. Code is available at
 [this https URL](https://github.com/qsh-zh/deis) 
.* 




 The original codebase can be found at
 [qsh-zh/deis](https://github.com/qsh-zh/deis) 
.
 


## Tips




 It is recommended to set
 `solver_order` 
 to 2 or 3, while
 `solver_order=1` 
 is equivalent to
 [DDIMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddim#diffusers.DDIMScheduler) 
.
 



 Dynamic thresholding from
 [Imagen](https://huggingface.co/papers/2205.11487) 
 is supported, and for pixel-space
diffusion models, you can set
 `thresholding=True` 
 to use the dynamic thresholding.
 


## DEISMultistepScheduler




### 




 class
 

 diffusers.
 

 DEISMultistepScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L74)



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
 
 : typing.Optional[numpy.ndarray] = None
 




 solver\_order
 
 : int = 2
 




 prediction\_type
 
 : str = 'epsilon'
 




 thresholding
 
 : bool = False
 




 dynamic\_thresholding\_ratio
 
 : float = 0.995
 




 sample\_max\_value
 
 : float = 1.0
 




 algorithm\_type
 
 : str = 'deis'
 




 solver\_type
 
 : str = 'logrho'
 




 lower\_order\_final
 
 : bool = True
 




 use\_karras\_sigmas
 
 : typing.Optional[bool] = False
 




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
* **solver\_order** 
 (
 `int` 
 , defaults to 2) —
The DEIS order which can be
 `1` 
 or
 `2` 
 or
 `3` 
. It is recommended to use
 `solver_order=2` 
 for guided
sampling, and
 `solver_order=3` 
 for unconditional sampling.
* **prediction\_type** 
 (
 `str` 
 , defaults to
 `epsilon` 
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
* **algorithm\_type** 
 (
 `str` 
 , defaults to
 `deis` 
 ) —
The algorithm type for the solver.
* **lower\_order\_final** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps.
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


`DEISMultistepScheduler` 
 is a fast high order solver for diffusion ordinary differential equations (ODEs).
 



 This model inherits from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
. Check the superclass documentation for the generic
methods the library implements for all schedulers such as loading and saving.
 



#### 




 convert\_model\_output




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L338)



 (
 


 model\_output
 
 : FloatTensor
 




 \*args
 


 sample
 
 : FloatTensor = None
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The direct output from the learned diffusion model.
* **timestep** 
 (
 `int` 
 ) —
The current discrete timestep in the diffusion chain.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by the diffusion process.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 The converted model output.
 



 Convert the model output to the corresponding type the DEIS algorithm needs.
 




#### 




 deis\_first\_order\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L395)



 (
 


 model\_output
 
 : FloatTensor
 




 \*args
 


 sample
 
 : FloatTensor = None
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The direct output from the learned diffusion model.
* **timestep** 
 (
 `int` 
 ) —
The current discrete timestep in the diffusion chain.
* **prev\_timestep** 
 (
 `int` 
 ) —
The previous discrete timestep in the diffusion chain.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by the diffusion process.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 The sample tensor at the previous timestep.
 



 One step for the first-order DEIS (equivalent to DDIM).
 




#### 




 multistep\_deis\_second\_order\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L453)



 (
 


 model\_output\_list
 
 : typing.List[torch.FloatTensor]
 




 \*args
 


 sample
 
 : FloatTensor = None
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **model\_output\_list** 
 (
 `List[torch.FloatTensor]` 
 ) —
The direct outputs from learned diffusion model at current and latter timesteps.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by the diffusion process.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 The sample tensor at the previous timestep.
 



 One step for the second-order multistep DEIS.
 




#### 




 multistep\_deis\_third\_order\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L522)



 (
 


 model\_output\_list
 
 : typing.List[torch.FloatTensor]
 




 \*args
 


 sample
 
 : FloatTensor = None
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **model\_output\_list** 
 (
 `List[torch.FloatTensor]` 
 ) —
The direct outputs from learned diffusion model at current and latter timesteps.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by diffusion process.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 The sample tensor at the previous timestep.
 



 One step for the third-order multistep DEIS.
 




#### 




 scale\_model\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L694)



 (
 


 sample
 
 : FloatTensor
 




 \*args
 


 \*\*kwargs
 




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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L199)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_deis_multistep.py#L629)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : int
 




 sample
 
 : FloatTensor
 




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
 ) —
The current discrete timestep in the diffusion chain.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by the diffusion process.
* **return\_dict** 
 (
 `bool` 
 ) —
Whether or not to return a
 [SchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/dpm_discrete#diffusers.schedulers.scheduling_utils.SchedulerOutput) 
 or
 `tuple` 
.




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
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
the multistep DEIS.
 


## SchedulerOutput




### 




 class
 

 diffusers.schedulers.scheduling\_utils.
 

 SchedulerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_utils.py#L50)



 (
 


 prev\_sample
 
 : FloatTensor
 



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


 Base class for the output of a scheduler’s
 `step` 
 function.