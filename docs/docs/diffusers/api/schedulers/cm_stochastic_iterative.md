# CMStochasticIterativeScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/cm_stochastic_iterative>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/cm_stochastic_iterative>



[Consistency Models](https://huggingface.co/papers/2303.01469) 
 by Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever introduced a multistep and onestep scheduler (Algorithm 1) that is capable of generating good samples in one or a small number of steps.
 



 The abstract from the paper is:
 



*Diffusion models have made significant breakthroughs in image, audio, and video generation, but they depend on an iterative generation process that causes slow sampling speed and caps their potential for real-time applications. To overcome this limitation, we propose consistency models, a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either as a way to distill pre-trained diffusion models, or as standalone generative models. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step generation. For example, we achieve the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64 for one-step generation. When trained as standalone generative models, consistency models also outperform single-step, non-adversarial generative models on standard benchmarks like CIFAR-10, ImageNet 64x64 and LSUN 256x256.* 




 The original codebase can be found at
 [openai/consistency\_models](https://github.com/openai/consistency_models) 
.
 


## CMStochasticIterativeScheduler




### 




 class
 

 diffusers.
 

 CMStochasticIterativeScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_consistency_models.py#L44)



 (
 


 num\_train\_timesteps
 
 : int = 40
 




 sigma\_min
 
 : float = 0.002
 




 sigma\_max
 
 : float = 80.0
 




 sigma\_data
 
 : float = 0.5
 




 s\_noise
 
 : float = 1.0
 




 rho
 
 : float = 7.0
 




 clip\_denoised
 
 : bool = True
 



 )
 


 Parameters
 




* **num\_train\_timesteps** 
 (
 `int` 
 , defaults to 40) —
The number of diffusion steps to train the model.
* **sigma\_min** 
 (
 `float` 
 , defaults to 0.002) —
Minimum noise magnitude in the sigma schedule. Defaults to 0.002 from the original implementation.
* **sigma\_max** 
 (
 `float` 
 , defaults to 80.0) —
Maximum noise magnitude in the sigma schedule. Defaults to 80.0 from the original implementation.
* **sigma\_data** 
 (
 `float` 
 , defaults to 0.5) —
The standard deviation of the data distribution from the EDM
 [paper](https://huggingface.co/papers/2206.00364) 
. Defaults to 0.5 from the original implementation.
* **s\_noise** 
 (
 `float` 
 , defaults to 1.0) —
The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
1.011]. Defaults to 1.0 from the original implementation.
* **rho** 
 (
 `float` 
 , defaults to 7.0) —
The parameter for calculating the Karras sigma schedule from the EDM
 [paper](https://huggingface.co/papers/2206.00364) 
. Defaults to 7.0 from the original implementation.
* **clip\_denoised** 
 (
 `bool` 
 , defaults to
 `True` 
 ) —
Whether to clip the denoised outputs to
 `(-1, 1)` 
.
* **timesteps** 
 (
 `List` 
 or
 `np.ndarray` 
 or
 `torch.Tensor` 
 ,
 *optional* 
 ) —
An explicit timestep schedule that can be optionally specified. The timesteps are expected to be in
increasing order.


 Multistep and onestep sampling for consistency models.
 



 This model inherits from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
. Check the superclass documentation for the generic
methods the library implements for all schedulers such as loading and saving.
 



#### 




 get\_scalings\_for\_boundary\_condition




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_consistency_models.py#L254)



 (
 


 sigma
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`tuple` 


 Parameters
 




* **sigma** 
 (
 `torch.FloatTensor` 
 ) —
The current sigma in the Karras sigma schedule.




 Returns
 




 export const metadata = 'undefined';
 

`tuple` 




 export const metadata = 'undefined';
 




 A two-element tuple where
 `c_skip` 
 (which weights the current sample) is the first element and
 `c_out` 
 (which weights the consistency model output) is the second element.
 



 Gets the scalings used in the consistency model parameterization (from Appendix C of the
 [paper](https://huggingface.co/papers/2303.01469) 
 ) to enforce boundary condition.
 




`epsilon` 
 in the equations for
 `c_skip` 
 and
 `c_out` 
 is set to
 `sigma_min` 
.
 


#### 




 scale\_model\_input




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_consistency_models.py#L116)



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
 



 Scales the consistency model input by
 `(sigma**2 + sigma_data**2) ** 0.5` 
.
 




#### 




 set\_timesteps




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_consistency_models.py#L162)



 (
 


 num\_inference\_steps
 
 : typing.Optional[int] = None
 




 device
 
 : typing.Union[str, torch.device] = None
 




 timesteps
 
 : typing.Optional[typing.List[int]] = None
 



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
* **timesteps** 
 (
 `List[int]` 
 ,
 *optional* 
 ) —
Custom timesteps used to support arbitrary spacing between timesteps. If
 `None` 
 , then the default
timestep spacing strategy of equal spacing between timesteps is used. If
 `timesteps` 
 is passed,
 `num_inference_steps` 
 must be
 `None` 
.


 Sets the timesteps used for the diffusion chain (to be run before inference).
 




#### 




 sigma\_to\_t




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_consistency_models.py#L143)



 (
 


 sigmas
 
 : typing.Union[float, numpy.ndarray]
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`float` 
 or
 `np.ndarray` 


 Parameters
 




* **sigmas** 
 (
 `float` 
 or
 `np.ndarray` 
 ) —
A single Karras sigma or an array of Karras sigmas.




 Returns
 




 export const metadata = 'undefined';
 

`float` 
 or
 `np.ndarray` 




 export const metadata = 'undefined';
 




 A scaled input timestep or scaled input timestep array.
 



 Gets scaled timesteps from the Karras sigmas for input to the consistency model.
 




#### 




 step




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_consistency_models.py#L299)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : typing.Union[float, torch.FloatTensor]
 




 sample
 
 : FloatTensor
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[CMStochasticIterativeSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/cm_stochastic_iterative#diffusers.schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput) 
 or
 `tuple` 


 Parameters
 




* **model\_output** 
 (
 `torch.FloatTensor` 
 ) —
The direct output from the learned diffusion model.
* **timestep** 
 (
 `float` 
 ) —
The current timestep in the diffusion chain.
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
 [CMStochasticIterativeSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/cm_stochastic_iterative#diffusers.schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

[CMStochasticIterativeSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/cm_stochastic_iterative#diffusers.schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [CMStochasticIterativeSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/cm_stochastic_iterative#diffusers.schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput) 
 is returned,
otherwise a tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).
 


## CMStochasticIterativeSchedulerOutput




### 




 class
 

 diffusers.schedulers.scheduling\_consistency\_models.
 

 CMStochasticIterativeSchedulerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_consistency_models.py#L31)



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


 Output class for the scheduler’s
 `step` 
 function.