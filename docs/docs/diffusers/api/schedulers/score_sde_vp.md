# ScoreSdeVpScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/score_sde_vp>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/score_sde_vp>



`ScoreSdeVpScheduler` 
 is a variance preserving stochastic differential equation (SDE) scheduler. It was introduced in the
 [Score-Based Generative Modeling through Stochastic Differential Equations](https://huggingface.co/papers/2011.13456) 
 paper by Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole.
 



 The abstract from the paper is:
 



*Creating noise from data is easy; creating data from noise is generative modeling. We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise. Crucially, the reverse-time SDE depends only on the time-dependent gradient field (ka, score) of the perturbed data distribution. By leveraging advances in score-based generative modeling, we can accurately estimate these scores with neural networks, and use numerical SDE solvers to generate samples. We show that this framework encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities. In particular, we introduce a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE. We also derive an equivalent neural ODE that samples from the same distribution as the SDE, but additionally enables exact likelihood computation, and improved sampling efficiency. In addition, we provide a new way to solve inverse problems with score-based models, as demonstrated with experiments on class-conditional generation, image inpainting, and colorization. Combined with multiple architectural improvements, we achieve record-breaking performance for unconditional image generation on CIFAR-10 with an Inception score of 9.89 and FID of 2.20, a competitive likelihood of 2.99 bits/dim, and demonstrate high fidelity generation of 1024 x 1024 images for the first time from a score-based generative model* 
.
 




 🚧 This scheduler is under construction!
 


## ScoreSdeVpScheduler




### 




 class
 

 diffusers.schedulers.
 

 ScoreSdeVpScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_vp.py#L27)



 (
 


 num\_train\_timesteps
 
 = 2000
 




 beta\_min
 
 = 0.1
 




 beta\_max
 
 = 20
 




 sampling\_eps
 
 = 0.001
 



 )
 


 Parameters
 




* **num\_train\_timesteps** 
 (
 `int` 
 , defaults to 2000) —
The number of diffusion steps to train the model.
* **beta\_min** 
 (
 `int` 
 , defaults to 0.1) —
* **beta\_max** 
 (
 `int` 
 , defaults to 20) —
* **sampling\_eps** 
 (
 `int` 
 , defaults to 1e-3) —
The end value of sampling where timesteps decrease progressively from 1 to epsilon.


`ScoreSdeVpScheduler` 
 is a variance preserving stochastic differential equation (SDE) scheduler.
 



 This model inherits from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
. Check the superclass documentation for the generic
methods the library implements for all schedulers such as loading and saving.
 



#### 




 set\_timesteps




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_vp.py#L51)



 (
 


 num\_inference\_steps
 


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


 Sets the continuous timesteps used for the diffusion chain (to be run before inference).
 




#### 




 step\_pred




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_sde_vp.py#L63)



 (
 


 score
 


 x
 


 t
 


 generator
 
 = None
 



 )
 


 Parameters
 




* **score** 
 () —
* **x** 
 () —
* **t** 
 () —
* **generator** 
 (
 `torch.Generator` 
 ,
 *optional* 
 ) —
A random number generator.


 Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
process from the learned model outputs (most often the predicted noise).