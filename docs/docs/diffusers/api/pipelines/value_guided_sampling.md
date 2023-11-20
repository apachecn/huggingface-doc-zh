# Value-guided planning

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/value_guided_sampling>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/value_guided_sampling>




 🧪 This is an experimental pipeline for reinforcement learning!
 




 This pipeline is based on the
 [Planning with Diffusion for Flexible Behavior Synthesis](https://huggingface.co/papers/2205.09991) 
 paper by Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine.
 



 The abstract from the paper is:
 



*Model-based reinforcement learning methods often use learning only for the purpose of estimating an approximate dynamics model, offloading the rest of the decision-making work to classical trajectory optimizers. While conceptually simple, this combination has a number of empirical shortcomings, suggesting that learned models may not be well-suited to standard trajectory optimization. In this paper, we consider what it would look like to fold as much of the trajectory optimization pipeline as possible into the modeling problem, such that sampling from the model and planning with it become nearly identical. The core of our technical approach lies in a diffusion probabilistic model that plans by iteratively denoising trajectories. We show how classifier-guided sampling and image inpainting can be reinterpreted as coherent planning strategies, explore the unusual and useful properties of diffusion-based planning methods, and demonstrate the effectiveness of our framework in control settings that emphasize long-horizon decision-making and test-time flexibility* 
.
 



 You can find additional information about the model on the
 [project page](https://diffusion-planning.github.io/) 
 , the
 [original codebase](https://github.com/jannerm/diffuser) 
 , or try it out in a demo
 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/reinforcement_learning_with_diffusers.ipynb) 
.
 



 The script to run the model is available
 [here](https://github.com/huggingface/diffusers/tree/main/examples/reinforcement_learning) 
.
 


## ValueGuidedRLPipeline




### 




 class
 

 diffusers.experimental.
 

 ValueGuidedRLPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/experimental/rl/value_guided_sampling.py#L25)



 (
 


 value\_function
 
 : UNet1DModel
 




 unet
 
 : UNet1DModel
 




 scheduler
 
 : DDPMScheduler
 




 env
 




 )
 


 Parameters
 




* **value\_function** 
 (
 [UNet1DModel](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.UNet1DModel) 
 ) —
A specialized UNet for fine-tuning trajectories base on reward.
* **unet** 
 (
 [UNet1DModel](/docs/diffusers/v0.23.0/en/api/models/unet#diffusers.UNet1DModel) 
 ) —
UNet architecture to denoise the encoded trajectories.
* **scheduler** 
 (
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 ) —
A scheduler to be used in combination with
 `unet` 
 to denoise the encoded trajectories. Default for this
application is
 [DDPMScheduler](/docs/diffusers/v0.23.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler) 
.
* **env** 
 () —
An environment following the OpenAI gym API to act in. For now only Hopper has pretrained models.


 Pipeline for value-guided sampling from a diffusion model trained to predict sequences of states.
 



 This model inherits from
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) 
. Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).