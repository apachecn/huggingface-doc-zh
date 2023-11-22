# RePaintScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/repaint>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/repaint>



`RePaintScheduler` 
 is a DDPM-based inpainting scheduler for unsupervised inpainting with extreme masks. It is designed to be used with the
 [RePaintPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/repaint#diffusers.RePaintPipeline) 
 , and it is based on the paper
 [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://huggingface.co/papers/2201.09865) 
 by Andreas Lugmayr et al.
 



 The abstract from the paper is:
 



*Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask. Most existing approaches train for a certain distribution of masks, which limits their generalization capabilities to unseen mask types. Furthermore, training with pixel-wise and perceptual losses often leads to simple textural extensions towards the missing areas instead of semantically meaningful generation. In this work, we propose RePaint: A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach that is applicable to even extreme masks. We employ a pretrained unconditional DDPM as the generative prior. To condition the generation process, we only alter the reverse diffusion iterations by sampling the unmasked regions using the given image information. Since this technique does not modify or condition the original DDPM network itself, the model produces high-quality and diverse output images for any inpainting form. We validate our method for both faces and general-purpose image inpainting using standard and extreme masks. RePaint outperforms state-of-the-art Autoregressive, and GAN approaches for at least five out of six mask distributions. Github Repository: git.io/RePaint* 
.
 



 The original implementation can be found at
 [andreas128/RePaint](https://github.com/andreas128/) 
.
 


## RePaintScheduler




### 




 class
 

 diffusers.
 

 RePaintScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_repaint.py#L91)



 (
 


 num\_train\_timesteps
 
 : int = 1000
 




 beta\_start
 
 : float = 0.0001
 




 beta\_end
 
 : float = 0.02
 




 beta\_schedule
 
 : str = 'linear'
 




 eta
 
 : float = 0.0
 




 trained\_betas
 
 : typing.Optional[numpy.ndarray] = None
 




 clip\_sample
 
 : bool = True
 



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
 ,
 `squaredcos_cap_v2` 
 , or
 `sigmoid` 
.
* **eta** 
 (
 `float` 
 ) —
The weight of noise for added noise in diffusion step. If its value is between 0.0 and 1.0 it corresponds
to the DDIM scheduler, and if its value is between -0.0 and 1.0 it corresponds to the DDPM scheduler.
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
Clip the predicted sample between -1 and 1 for numerical stability.


`RePaintScheduler` 
 is a scheduler for DDPM inpainting inside a given mask.
 



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_repaint.py#L165)



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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_repaint.py#L182)



 (
 


 num\_inference\_steps
 
 : int
 




 jump\_length
 
 : int = 10
 




 jump\_n\_sample
 
 : int = 10
 




 device
 
 : typing.Union[str, torch.device] = None
 



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
* **jump\_length** 
 (
 `int` 
 , defaults to 10) —
The number of steps taken forward in time before going backward in time for a single jump (“j” in
RePaint paper). Take a look at Figure 9 and 10 in the paper.
* **jump\_n\_sample** 
 (
 `int` 
 , defaults to 10) —
The number of times to make a forward time jump for a given chosen time sample. Take a look at Figure 9
and 10 in the paper.
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
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_repaint.py#L248)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : int
 




 sample
 
 : FloatTensor
 




 original\_image
 
 : FloatTensor
 




 mask
 
 : FloatTensor
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[RePaintSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/repaint#diffusers.schedulers.scheduling_repaint.RePaintSchedulerOutput) 
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
* **original\_image** 
 (
 `torch.FloatTensor` 
 ) —
The original image to inpaint on.
* **mask** 
 (
 `torch.FloatTensor` 
 ) —
The mask where a value of 0.0 indicates which part of the original image to inpaint.
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
 [RePaintSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/repaint#diffusers.schedulers.scheduling_repaint.RePaintSchedulerOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

[RePaintSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/repaint#diffusers.schedulers.scheduling_repaint.RePaintSchedulerOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [RePaintSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/repaint#diffusers.schedulers.scheduling_repaint.RePaintSchedulerOutput) 
 is returned,
otherwise a tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).
 


## RePaintSchedulerOutput




### 




 class
 

 diffusers.schedulers.scheduling\_repaint.
 

 RePaintSchedulerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_repaint.py#L29)



 (
 


 prev\_sample
 
 : FloatTensor
 




 pred\_original\_sample
 
 : FloatTensor
 



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
* **pred\_original\_sample** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch_size, num_channels, height, width)` 
 for images) —
The predicted denoised sample (x\_{0}) based on the model output from
the current timestep.
 `pred_original_sample` 
 can be used to preview progress or for guidance.


 Output class for the scheduler’s step function output.