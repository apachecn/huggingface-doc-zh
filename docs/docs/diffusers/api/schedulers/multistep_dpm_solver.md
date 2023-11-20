# DPMSolverMultistepScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/multistep_dpm_solver>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver>



`DPMSolverMultistep` 
 is a multistep scheduler from
 [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927) 
 and
 [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095) 
 by Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu.
 



 DPMSolver (and the improved version DPMSolver++) is a fast dedicated high-order solver for diffusion ODEs with convergence order guarantee. Empirically, DPMSolver sampling with only 20 steps can generate high-quality
samples, and it can generate quite good samples even in 10 steps.
 


## Tips




 It is recommended to set
 `solver_order` 
 to 2 for guide sampling, and
 `solver_order=3` 
 for unconditional sampling.
 



 Dynamic thresholding from Imagen (
 <https://huggingface.co/papers/2205.11487>
 ) is supported, and for pixel-space
diffusion models, you can set both
 `algorithm_type="dpmsolver++"` 
 and
 `thresholding=True` 
 to use the dynamic
thresholding. This thresholding method is unsuitable for latent-space diffusion models such as
Stable Diffusion.
 



 The SDE variant of DPMSolver and DPM-Solver++ is also supported, but only for the first and second-order solvers. This is a fast SDE solver for the reverse diffusion SDE. It is recommended to use the second-order
 `sde-dpmsolver++` 
.
 


## DPMSolverMultistepScheduler




### 




 class
 

 diffusers.
 

 DPMSolverMultistepScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L74)



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
 
 : str = 'dpmsolver++'
 




 solver\_type
 
 : str = 'midpoint'
 




 lower\_order\_final
 
 : bool = True
 




 euler\_at\_final
 
 : bool = False
 




 use\_karras\_sigmas
 
 : typing.Optional[bool] = False
 




 use\_lu\_lambdas
 
 : typing.Optional[bool] = False
 




 lambda\_min\_clipped
 
 : float = -inf
 




 variance\_type
 
 : typing.Optional[str] = None
 




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
The DPMSolver order which can be
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
 and
 `algorithm_type="dpmsolver++"` 
.
* **algorithm\_type** 
 (
 `str` 
 , defaults to
 `dpmsolver++` 
 ) —
Algorithm type for the solver; can be
 `dpmsolver` 
 ,
 `dpmsolver++` 
 ,
 `sde-dpmsolver` 
 or
 `sde-dpmsolver++` 
. The
 `dpmsolver` 
 type implements the algorithms in the
 [DPMSolver](https://huggingface.co/papers/2206.00927) 
 paper, and the
 `dpmsolver++` 
 type implements the algorithms in the
 [DPMSolver++](https://huggingface.co/papers/2211.01095) 
 paper. It is recommended to use
 `dpmsolver++` 
 or
 `sde-dpmsolver++` 
 with
 `solver_order=2` 
 for guided sampling like in Stable Diffusion.
* **solver\_type** 
 (
 `str` 
 , defaults to
 `midpoint` 
 ) —
Solver type for the second-order solver; can be
 `midpoint` 
 or
 `heun` 
. The solver type slightly affects the
sample quality, especially for a small number of steps. It is recommended to use
 `midpoint` 
 solvers.
* **lower\_order\_final** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
* **euler\_at\_final** 
 (
 `bool` 
 , defaults to
 `False` 
 ) —
Whether to use Euler’s method in the final step. It is a trade-off between numerical stability and detail
richness. This can stabilize the sampling of the SDE variant of DPMSolver for small number of inference
steps, but sometimes may result in blurring.
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
* **use\_lu\_lambdas** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `False` 
 ) —
Whether to use the uniform-logSNR for step sizes proposed by Lu’s DPM-Solver in the noise schedule during
the sampling process. If
 `True` 
 , the sigmas and time steps are determined according to a sequence of
 `lambda(t)` 
.
* **lambda\_min\_clipped** 
 (
 `float` 
 , defaults to
 `-inf` 
 ) —
Clipping threshold for the minimum value of
 `lambda(t)` 
 for numerical stability. This is critical for the
cosine (
 `squaredcos_cap_v2` 
 ) noise schedule.
* **variance\_type** 
 (
 `str` 
 ,
 *optional* 
 ) —
Set to “learned” or “learned\_range” for diffusion models that predict variance. If set, the model’s output
contains the predicted Gaussian variance.
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


`DPMSolverMultistepScheduler` 
 is a fast dedicated high-order solver for diffusion ODEs.
 



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L386)



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
 



 Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
integral of the data prediction model.
 




 The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
prediction and data prediction models.
 


#### 




 dpm\_solver\_first\_order\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L485)



 (
 


 model\_output
 
 : FloatTensor
 




 \*args
 


 sample
 
 : FloatTensor = None
 




 noise
 
 : typing.Optional[torch.FloatTensor] = None
 




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
 



 One step for the first-order DPMSolver (equivalent to DDIM).
 




#### 




 multistep\_dpm\_solver\_second\_order\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L554)



 (
 


 model\_output\_list
 
 : typing.List[torch.FloatTensor]
 




 \*args
 


 sample
 
 : FloatTensor = None
 




 noise
 
 : typing.Optional[torch.FloatTensor] = None
 




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
 



 One step for the second-order multistep DPMSolver.
 




#### 




 multistep\_dpm\_solver\_third\_order\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L677)



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
 



 One step for the third-order multistep DPMSolver.
 




#### 




 scale\_model\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L857)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L226)



 (
 


 num\_inference\_steps
 
 : int = None
 




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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L781)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : int
 




 sample
 
 : FloatTensor
 




 generator
 
 = None
 




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
 `int` 
 ) —
The current discrete timestep in the diffusion chain.
* **sample** 
 (
 `torch.FloatTensor` 
 ) —
A current instance of a sample created by the diffusion process.
* **generator** 
 (
 `torch.Generator` 
 ,
 *optional* 
 ) —
A random number generator.
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
the multistep DPMSolver.
 


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