# KarrasVeScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/stochastic_karras_ve>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/stochastic_karras_ve>



`KarrasVeScheduler` 
 is a stochastic sampler tailored o variance-expanding (VE) models. It is based on the
 [Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364) 
 and
 [Score-based generative modeling through stochastic differential equations](https://huggingface.co/papers/2011.13456) 
 papers.
 


## KarrasVeScheduler




### 




 class
 

 diffusers.
 

 KarrasVeScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_karras_ve.py#L49)



 (
 


 sigma\_min
 
 : float = 0.02
 




 sigma\_max
 
 : float = 100
 




 s\_noise
 
 : float = 1.007
 




 s\_churn
 
 : float = 80
 




 s\_min
 
 : float = 0.05
 




 s\_max
 
 : float = 50
 



 )
 


 Parameters
 




* **sigma\_min** 
 (
 `float` 
 , defaults to 0.02) —
The minimum noise magnitude.
* **sigma\_max** 
 (
 `float` 
 , defaults to 100) —
The maximum noise magnitude.
* **s\_noise** 
 (
 `float` 
 , defaults to 1.007) —
The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
1.011].
* **s\_churn** 
 (
 `float` 
 , defaults to 80) —
The parameter controlling the overall amount of stochasticity. A reasonable range is [0, 100].
* **s\_min** 
 (
 `float` 
 , defaults to 0.05) —
The start value of the sigma range to add noise (enable stochasticity). A reasonable range is [0, 10].
* **s\_max** 
 (
 `float` 
 , defaults to 50) —
The end value of the sigma range to add noise. A reasonable range is [0.2, 80].


 A stochastic scheduler tailored to variance-expanding models.
 



 This model inherits from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
. Check the superclass documentation for the generic
methods the library implements for all schedulers such as loading and saving.
 




 For more details on the parameters, see
 [Appendix E](https://arxiv.org/abs/2206.00364) 
. The grid search values used
to find the optimal
 `{s_noise, s_churn, s_min, s_max}` 
 for a specific model are described in Table 5 of the paper.
 




#### 




 add\_noise\_to\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_karras_ve.py#L138)



 (
 


 sample
 
 : FloatTensor
 




 sigma
 
 : float
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 



 )
 


 Parameters
 




* **sample** 
 (
 `torch.FloatTensor` 
 ) —
The input sample.
* **sigma** 
 (
 `float` 
 ) —
* **generator** 
 (
 `torch.Generator` 
 ,
 *optional* 
 ) —
A random number generator.


 Explicit Langevin-like “churn” step of adding noise to the sample according to a
 `gamma_i ≥ 0` 
 to reach a
higher noise level
 `sigma_hat = sigma_i + gamma_i*sigma_i` 
.
 




#### 




 scale\_model\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_karras_ve.py#L99)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_karras_ve.py#L116)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_karras_ve.py#L164)



 (
 


 model\_output
 
 : FloatTensor
 




 sigma\_hat
 
 : float
 




 sigma\_prev
 
 : float
 




 sample\_hat
 
 : FloatTensor
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput` 
 or
 `tuple` 


 Parameters
 




* **model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The direct output from learned diffusion model.
* **sigma\_hat** 
 (
 `float` 
 ) —
* **sigma\_prev** 
 (
 `float` 
 ) —
* **sample\_hat** 
 (
 `torch.FloatTensor` 
 ) —
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 `~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput` 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput` 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 `~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput` 
 is returned,
otherwise a tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).
 




#### 




 step\_correct




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_karras_ve.py#L203)



 (
 


 model\_output
 
 : FloatTensor
 




 sigma\_hat
 
 : float
 




 sigma\_prev
 
 : float
 




 sample\_hat
 
 : FloatTensor
 




 sample\_prev
 
 : FloatTensor
 




 derivative
 
 : FloatTensor
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

 prev\_sample (TODO)
 




 Parameters
 




* **model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The direct output from learned diffusion model.
* **sigma\_hat** 
 (
 `float` 
 ) — TODO
* **sigma\_prev** 
 (
 `float` 
 ) — TODO
* **sample\_hat** 
 (
 `torch.FloatTensor` 
 ) — TODO
* **sample\_prev** 
 (
 `torch.FloatTensor` 
 ) — TODO
* **derivative** 
 (
 `torch.FloatTensor` 
 ) — TODO
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [DDPMSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

 prev\_sample (TODO)
 



 export const metadata = 'undefined';
 




 updated sample in the diffusion chain. derivative (TODO): TODO
 



 Corrects the predicted sample based on the
 `model_output` 
 of the network.
 


## KarrasVeOutput




### 




 class
 

 diffusers.schedulers.scheduling\_karras\_ve.
 

 KarrasVeOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_karras_ve.py#L29)



 (
 


 prev\_sample
 
 : FloatTensor
 




 derivative
 
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
Computed sample (x\_{t-1}) of previous timestep.
 `prev_sample` 
 should be used as next model input in the
denoising loop.
* **derivative** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 for images) —
Derivative of predicted original image sample (x\_0).
* **pred\_original\_sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 for images) —
The predicted denoised sample (x\_{0}) based on the model output from the current timestep.
 `pred_original_sample` 
 can be used to preview progress or for guidance.


 Output class for the scheduler’s step function output.