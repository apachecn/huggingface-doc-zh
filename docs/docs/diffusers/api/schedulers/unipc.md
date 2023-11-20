# UniPCMultistepScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/unipc>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/unipc>



`UniPCMultistepScheduler` 
 is a training-free framework designed for fast sampling of diffusion models. It was introduced in
 [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://huggingface.co/papers/2302.04867) 
 by Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu.
 



 It consists of a corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders.
UniPC is by design model-agnostic, supporting pixel-space/latent-space DPMs on unconditional/conditional sampling. It can also be applied to both noise prediction and data prediction models. The corrector UniC can be also applied after any off-the-shelf solvers to increase the order of accuracy.
 



 The abstract from the paper is:
 



*Diffusion probabilistic models (DPMs) have demonstrated a very promising ability in high-resolution image synthesis. However, sampling from a pre-trained DPM usually requires hundreds of model evaluations, which is computationally expensive. Despite recent progress in designing high-order solvers for DPMs, there still exists room for further speedup, especially in extremely few steps (e.g., 5~10 steps). Inspired by the predictor-corrector for ODE solvers, we develop a unified corrector (UniC) that can be applied after any existing DPM sampler to increase the order of accuracy without extra model evaluations, and derive a unified predictor (UniP) that supports arbitrary order as a byproduct. Combining UniP and UniC, we propose a unified predictor-corrector framework called UniPC for the fast sampling of DPMs, which has a unified analytical form for any order and can significantly improve the sampling quality over previous methods. We evaluate our methods through extensive experiments including both unconditional and conditional sampling using pixel-space and latent-space DPMs. Our UniPC can achieve 3.87 FID on CIFAR10 (unconditional) and 7.51 FID on ImageNet 256times256 (conditional) with only 10 function evaluations. Code is available at
 <https://github.com/wl-zhao/UniPC>*
.
 



 The original codebase can be found at
 [wl-zhao/UniPC](https://github.com/wl-zhao/UniPC) 
.
 


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
 `predict_x0=True` 
 and
 `thresholding=True` 
 to use dynamic thresholding. This thresholding method is unsuitable for latent-space diffusion models such as Stable Diffusion.
 


## UniPCMultistepScheduler




### 




 class
 

 diffusers.
 

 UniPCMultistepScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_unipc_multistep.py#L74)



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
 




 predict\_x0
 
 : bool = True
 




 solver\_type
 
 : str = 'bh2'
 




 lower\_order\_final
 
 : bool = True
 




 disable\_corrector
 
 : typing.List[int] = []
 




 solver\_p
 
 : SchedulerMixin = None
 




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
 , default
 `2` 
 ) —
The UniPC order which can be any positive integer. The effective order of accuracy is
 `solver_order + 1` 
 due to the UniC. It is recommended to use
 `solver_order=2` 
 for guided sampling, and
 `solver_order=3` 
 for
unconditional sampling.
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
 `predict_x0=True` 
.
* **predict\_x0** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to use the updating algorithm on the predicted x0.
* **solver\_type** 
 (
 `str` 
 , default
 `bh2` 
 ) —
Solver type for UniPC. It is recommended to use
 `bh1` 
 for unconditional sampling when steps < 10, and
 `bh2` 
 otherwise.
* **lower\_order\_final** 
 (
 `bool` 
 , default
 `True` 
 ) —
Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
* **disable\_corrector** 
 (
 `list` 
 , default
 `[]` 
 ) —
Decides which step to disable the corrector to mitigate the misalignment between
 `epsilon_theta(x_t, c)` 
 and
 `epsilon_theta(x_t^c, c)` 
 which can influence convergence for a large guidance scale. Corrector is
usually disabled during the first few steps.
* **solver\_p** 
 (
 `SchedulerMixin` 
 , default
 `None` 
 ) —
Any other scheduler that if specified, the algorithm becomes
 `solver_p + UniC` 
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


`UniPCMultistepScheduler` 
 is a training-free framework designed for the fast sampling of diffusion models.
 



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_unipc_multistep.py#L352)



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
 



 Convert the model output to the corresponding type the UniPC algorithm needs.
 




#### 




 multistep\_uni\_c\_bh\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_unipc_multistep.py#L551)



 (
 


 this\_model\_output
 
 : FloatTensor
 




 \*args
 


 last\_sample
 
 : FloatTensor = None
 




 this\_sample
 
 : FloatTensor = None
 




 order
 
 : int = None
 




 \*\*kwargs
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 


 Parameters
 




* **this\_model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The model outputs at
 `x_t` 
.
* **this\_timestep** 
 (
 `int` 
 ) —
The current timestep
 `t` 
.
* **last\_sample** 
 (
 `torch.FloatTensor` 
 ) —
The generated sample before the last predictor
 `x_{t-1}` 
.
* **this\_sample** 
 (
 `torch.FloatTensor` 
 ) —
The generated sample after the last predictor
 `x_{t}` 
.
* **order** 
 (
 `int` 
 ) —
The
 `p` 
 of UniC-p at this step. The effective order of accuracy should be
 `order + 1` 
.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 The corrected sample tensor at the current timestep.
 



 One step for the UniC (B(h) version).
 




#### 




 multistep\_uni\_p\_bh\_update




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_unipc_multistep.py#L422)



 (
 


 model\_output
 
 : FloatTensor
 




 \*args
 


 sample
 
 : FloatTensor = None
 




 order
 
 : int = None
 




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
The direct output from the learned diffusion model at the current timestep.
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
* **order** 
 (
 `int` 
 ) —
The order of UniP at this timestep (corresponds to the
 *p* 
 in UniPC-p).




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 




 export const metadata = 'undefined';
 




 The sample tensor at the previous timestep.
 



 One step for the UniP (B(h) version). Alternatively,
 `self.solver_p` 
 is used if is specified.
 




#### 




 scale\_model\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_unipc_multistep.py#L788)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_unipc_multistep.py#L210)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_unipc_multistep.py#L707)



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
 `int` 
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
the multistep UniPC.
 


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