# DDPMScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/ddpm>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/ddpm>



[Denoising Diffusion Probabilistic Models](https://huggingface.co/papers/2006.11239) 
 (DDPM) by Jonathan Ho, Ajay Jain and Pieter Abbeel proposes a diffusion based model of the same name. In the context of the 🤗 Diffusers library, DDPM refers to the discrete denoising scheduler from the paper as well as the pipeline.
 



 The abstract from the paper is:
 



*We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN.* 


## DDPMScheduler




### 




 class
 

 diffusers.
 

 DDPMScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddpm.py#L92)



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
 




 variance\_type
 
 : str = 'fixed\_small'
 




 clip\_sample
 
 : bool = True
 




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
* **variance\_type** 
 (
 `str` 
 , defaults to
 `"fixed_small"` 
 ) —
Clip the variance when adding noise to the denoised sample. Choose from
 `fixed_small` 
 ,
 `fixed_small_log` 
 ,
 `fixed_large` 
 ,
 `fixed_large_log` 
 ,
 `learned` 
 or
 `learned_range` 
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


`DDPMScheduler` 
 explores the connections between denoising score matching and Langevin dynamics sampling.
 



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddpm.py#L190)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddpm.py#L207)



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
The number of diffusion steps used when generating samples with a pre-trained model. If used,
 `timesteps` 
 must be
 `None` 
.
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


 Sets the discrete timesteps used for the diffusion chain (to be run before inference).
 




#### 




 step




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddpm.py#L355)



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
 

[DDPMSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput) 
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
 [DDPMSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

[DDPMSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [DDPMSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput) 
 is returned, otherwise a
tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).
 


## DDPMSchedulerOutput




### 




 class
 

 diffusers.schedulers.scheduling\_ddpm.
 

 DDPMSchedulerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_ddpm.py#L31)



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