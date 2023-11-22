# ScoreSdeVeScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/score_sde_ve>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/score_sde_ve>



`ScoreSdeVeScheduler` 
 is a variance exploding stochastic differential equation (SDE) scheduler. It was introduced in the
 [Score-Based Generative Modeling through Stochastic Differential Equations](https://huggingface.co/papers/2011.13456) 
 paper by Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole.
 



 The abstract from the paper is:
 



*Creating noise from data is easy; creating data from noise is generative modeling. We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise. Crucially, the reverse-time SDE depends only on the time-dependent gradient field (ka, score) of the perturbed data distribution. By leveraging advances in score-based generative modeling, we can accurately estimate these scores with neural networks, and use numerical SDE solvers to generate samples. We show that this framework encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities. In particular, we introduce a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE. We also derive an equivalent neural ODE that samples from the same distribution as the SDE, but additionally enables exact likelihood computation, and improved sampling efficiency. In addition, we provide a new way to solve inverse problems with score-based models, as demonstrated with experiments on class-conditional generation, image inpainting, and colorization. Combined with multiple architectural improvements, we achieve record-breaking performance for unconditional image generation on CIFAR-10 with an Inception score of 9.89 and FID of 2.20, a competitive likelihood of 2.99 bits/dim, and demonstrate high fidelity generation of 1024 x 1024 images for the first time from a score-based generative model* 
.
 


## ScoreSdeVeScheduler




### 




 class
 

 diffusers.
 

 ScoreSdeVeScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_ve.py#L46)



 (
 


 num\_train\_timesteps
 
 : int = 2000
 




 snr
 
 : float = 0.15
 




 sigma\_min
 
 : float = 0.01
 




 sigma\_max
 
 : float = 1348.0
 




 sampling\_eps
 
 : float = 1e-05
 




 correct\_steps
 
 : int = 1
 



 )
 


 Parameters
 




* **num\_train\_timesteps** 
 (
 `int` 
 , defaults to 1000) —
The number of diffusion steps to train the model.
* **snr** 
 (
 `float` 
 , defaults to 0.15) —
A coefficient weighting the step from the
 `model_output` 
 sample (from the network) to the random noise.
* **sigma\_min** 
 (
 `float` 
 , defaults to 0.01) —
The initial noise scale for the sigma sequence in the sampling procedure. The minimum sigma should mirror
the distribution of the data.
* **sigma\_max** 
 (
 `float` 
 , defaults to 1348.0) —
The maximum value used for the range of continuous timesteps passed into the model.
* **sampling\_eps** 
 (
 `float` 
 , defaults to 1e-5) —
The end value of sampling where timesteps decrease progressively from 1 to epsilon.
* **correct\_steps** 
 (
 `int` 
 , defaults to 1) —
The number of correction steps performed on a produced sample.


`ScoreSdeVeScheduler` 
 is a variance exploding stochastic differential equation (SDE) scheduler.
 



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_ve.py#L89)



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




 set\_sigmas




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_ve.py#L125)



 (
 


 num\_inference\_steps
 
 : int
 




 sigma\_min
 
 : float = None
 




 sigma\_max
 
 : float = None
 




 sampling\_eps
 
 : float = None
 



 )
 


 Parameters
 




* **num\_inference\_steps** 
 (
 `int` 
 ) —
The number of diffusion steps used when generating samples with a pre-trained model.
* **sigma\_min** 
 (
 `float` 
 , optional) —
The initial noise scale value (overrides value given during scheduler instantiation).
* **sigma\_max** 
 (
 `float` 
 , optional) —
The final noise scale value (overrides value given during scheduler instantiation).
* **sampling\_eps** 
 (
 `float` 
 , optional) —
The final timestep value (overrides value given during scheduler instantiation).


 Sets the noise scales used for the diffusion chain (to be run before inference). The sigmas control the weight
of the
 `drift` 
 and
 `diffusion` 
 components of the sample update.
 




#### 




 set\_timesteps




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_ve.py#L106)



 (
 


 num\_inference\_steps
 
 : int
 




 sampling\_eps
 
 : float = None
 




 device
 
 : typing.Union[str, torch.device] = None
 



 )
 


 Parameters
 




* **num\_inference\_steps** 
 (
 `int` 
 ) —
The number of diffusion steps used when generating samples with a pre-trained model.
* **sampling\_eps** 
 (
 `float` 
 ,
 *optional* 
 ) —
The final timestep value (overrides value given during scheduler instantiation).
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


 Sets the continuous timesteps used for the diffusion chain (to be run before inference).
 




#### 




 step\_correct




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_ve.py#L228)



 (
 


 model\_output
 
 : FloatTensor
 




 sample
 
 : FloatTensor
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
 or
 `tuple` 


 Parameters
 




* **model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The direct output from learned diffusion model.
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
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

[SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
 is returned, otherwise a tuple
is returned where the first element is the sample tensor.
 



 Correct the predicted sample based on the
 `model_output` 
 of the network. This is often run repeatedly after
making the prediction for the previous timestep.
 




#### 




 step\_pred




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_ve.py#L160)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : int
 




 sample
 
 : FloatTensor
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
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
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

[SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [SdeVeOutput](/docs/diffusers/v0.23.0/en/api/schedulers/score_sde_ve#diffusers.schedulers.scheduling_sde_ve.SdeVeOutput) 
 is returned, otherwise a tuple
is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).
 


## SdeVeOutput




### 




 class
 

 diffusers.schedulers.scheduling\_sde\_ve.
 

 SdeVeOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_ve.py#L30)



 (
 


 prev\_sample
 
 : FloatTensor
 




 prev\_sample\_mean
 
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
* **prev\_sample\_mean** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 for images) —
Mean averaged
 `prev_sample` 
 over previous timesteps.


 Output class for the scheduler’s
 `step` 
 function output.