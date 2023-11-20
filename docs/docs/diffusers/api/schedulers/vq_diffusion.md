# VQDiffusionScheduler

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/schedulers/vq_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/schedulers/vq_diffusion>



`VQDiffusionScheduler` 
 converts the transformer model’s output into a sample for the unnoised image at the previous diffusion timestep. It was introduced in
 [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://huggingface.co/papers/2111.14822) 
 by Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo.
 



 The abstract from the paper is:
 



*We present the vector quantized diffusion (VQ-Diffusion) model for text-to-image generation. This method is based on a vector quantized variational autoencoder (VQ-VAE) whose latent space is modeled by a conditional variant of the recently developed Denoising Diffusion Probabilistic Model (DDPM). We find that this latent-space method is well-suited for text-to-image generation tasks because it not only eliminates the unidirectional bias with existing methods but also allows us to incorporate a mask-and-replace diffusion strategy to avoid the accumulation of errors, which is a serious problem with existing methods. Our experiments show that the VQ-Diffusion produces significantly better text-to-image generation results when compared with conventional autoregressive (AR) models with similar numbers of parameters. Compared with previous GAN-based text-to-image methods, our VQ-Diffusion can handle more complex scenes and improve the synthesized image quality by a large margin. Finally, we show that the image generation computation in our method can be made highly efficient by reparameterization. With traditional AR methods, the text-to-image generation time increases linearly with the output image resolution and hence is quite time consuming even for normal size images. The VQ-Diffusion allows us to achieve a better trade-off between quality and speed. Our experiments indicate that the VQ-Diffusion model with the reparameterization is fifteen times faster than traditional AR methods while achieving a better image quality.* 


## VQDiffusionScheduler




### 




 class
 

 diffusers.
 

 VQDiffusionScheduler




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_vq_diffusion.py#L106)



 (
 


 num\_vec\_classes
 
 : int
 




 num\_train\_timesteps
 
 : int = 100
 




 alpha\_cum\_start
 
 : float = 0.99999
 




 alpha\_cum\_end
 
 : float = 9e-06
 




 gamma\_cum\_start
 
 : float = 9e-06
 




 gamma\_cum\_end
 
 : float = 0.99999
 



 )
 


 Parameters
 




* **num\_vec\_classes** 
 (
 `int` 
 ) —
The number of classes of the vector embeddings of the latent pixels. Includes the class for the masked
latent pixel.
* **num\_train\_timesteps** 
 (
 `int` 
 , defaults to 100) —
The number of diffusion steps to train the model.
* **alpha\_cum\_start** 
 (
 `float` 
 , defaults to 0.99999) —
The starting cumulative alpha value.
* **alpha\_cum\_end** 
 (
 `float` 
 , defaults to 0.00009) —
The ending cumulative alpha value.
* **gamma\_cum\_start** 
 (
 `float` 
 , defaults to 0.00009) —
The starting cumulative gamma value.
* **gamma\_cum\_end** 
 (
 `float` 
 , defaults to 0.99999) —
The ending cumulative gamma value.


 A scheduler for vector quantized diffusion.
 



 This model inherits from
 [SchedulerMixin](/docs/diffusers/v0.23.0/en/api/schedulers/overview#diffusers.SchedulerMixin) 
 and
 [ConfigMixin](/docs/diffusers/v0.23.0/en/api/configuration#diffusers.ConfigMixin) 
. Check the superclass documentation for the generic
methods the library implements for all schedulers such as loading and saving.
 



#### 




 log\_Q\_t\_transitioning\_to\_known\_class




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_vq_diffusion.py#L356)



 (
 


 t
 
 : torch.int32
 




 x\_t
 
 : LongTensor
 




 log\_onehot\_x\_t
 
 : FloatTensor
 




 cumulative
 
 : bool
 



 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 
 of shape
 `(batch size, num classes - 1, num latent pixels)` 


 Parameters
 




* **t** 
 (
 `torch.Long` 
 ) —
The timestep that determines which transition matrix is used.
* **x\_t** 
 (
 `torch.LongTensor` 
 of shape
 `(batch size, num latent pixels)` 
 ) —
The classes of each latent pixel at time
 `t` 
.
* **log\_onehot\_x\_t** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch size, num classes, num latent pixels)` 
 ) —
The log one-hot vectors of
 `x_t` 
.
* **cumulative** 
 (
 `bool` 
 ) —
If cumulative is
 `False` 
 , the single step transition matrix
 `t-1` 
 ->
 `t` 
 is used. If cumulative is
 `True` 
 , the cumulative transition matrix
 `0` 
 ->
 `t` 
 is used.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 
 of shape
 `(batch size, num classes - 1, num latent pixels)` 




 export const metadata = 'undefined';
 




 Each
 *column* 
 of the returned matrix is a
 *row* 
 of log probabilities of the complete probability
transition matrix.
 



 When non cumulative, returns
 `self.num_classes - 1` 
 rows because the initial latent pixel cannot be
masked.
 



 Where:
 


* `q_n` 
 is the probability distribution for the forward process of the
 `n` 
 th latent pixel.
* C\_0 is a class of a latent pixel embedding
* C\_k is the class of the masked latent pixel



 non-cumulative result (omitting logarithms):
 



 \_0(
 
 x\_t
 
 |
 
 x\_
 
 {
 
 t
 
 -1\} = C\_0)
 
...
 
 q
 
 \_n(
 
 x\_t
 
 |
 
 x\_
 
 {
 
 t
 
 -1\} = C\_0)
 
...
...
...
q
 
 \_0(
 
 x\_t
 
 |
 
 x\_
 
 {
 
 t
 
 -1\} = C\_k)
 

...
 
 q
 
 \_n(
 
 x\_t
 
 |
 
 x\_
 
 {
 
 t
 
 -1\} = C\_k)
 
 `}
 wrap={false}
 />
 
 cumulative result (omitting logarithms):
 



 \_0\_cumulative(
 
 x\_t
 
 |
 
 x\_0
 
 = C\_0)
 
...
 
 q
 
 \_n\_cumulative(
 
 x\_t
 
 |
 
 x\_0
 
 = C\_0)
 
...
...
...
q
 
 \_0\_cumulative(
 
 x\_t
 
 |
 
 x\_0
 
 = C\_{
 
 k
 
 -1\})
 

...
 
 q
 
 \_n\_cumulative(
 
 x\_t
 
 |
 
 x\_0
 
 = C\_{
 
 k
 
 -1\})
 
 `}
 wrap={false}
 />
 


 Calculates the log probabilities of the rows from the (cumulative or non-cumulative) transition matrix for each
latent pixel in
 `x_t` 
.
 




#### 




 q\_posterior




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_vq_diffusion.py#L245)



 (
 


 log\_p\_x\_0
 


 x\_t
 


 t
 




 )
 

 →
 



 export const metadata = 'undefined';
 

`torch.FloatTensor` 
 of shape
 `(batch size, num classes, num latent pixels)` 


 Parameters
 




* **log\_p\_x\_0** 
 (
 `torch.FloatTensor` 
 of shape
 `(batch size, num classes - 1, num latent pixels)` 
 ) —
The log probabilities for the predicted classes of the initial latent pixels. Does not include a
prediction for the masked class as the initial unnoised image cannot be masked.
* **x\_t** 
 (
 `torch.LongTensor` 
 of shape
 `(batch size, num latent pixels)` 
 ) —
The classes of each latent pixel at time
 `t` 
.
* **t** 
 (
 `torch.Long` 
 ) —
The timestep that determines which transition matrix is used.




 Returns
 




 export const metadata = 'undefined';
 

`torch.FloatTensor` 
 of shape
 `(batch size, num classes, num latent pixels)` 




 export const metadata = 'undefined';
 




 The log probabilities for the predicted classes of the image at timestep
 `t-1` 
.
 


 Calculates the log probabilities for the predicted classes of the image at timestep
 `t-1` 
 :
 



```
p(x\_{t-1} | x_t) = sum( q(x\_t | x\_{t-1}) * q(x\_{t-1} | x\_0) * p(x_0) / q(x\_t | x\_0) )
```


#### 




 set\_timesteps




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_vq_diffusion.py#L178)



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
The device to which the timesteps and diffusion process parameters (alpha, beta, gamma) should be moved
to.


 Sets the discrete timesteps used for the diffusion chain (to be run before inference).
 




#### 




 step




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_vq_diffusion.py#L200)



 (
 


 model\_output
 
 : FloatTensor
 




 timestep
 
 : torch.int64
 




 sample
 
 : LongTensor
 




 generator
 
 : typing.Optional[torch.\_C.Generator] = None
 




 return\_dict
 
 : bool = True
 



 )
 

 →
 



 export const metadata = 'undefined';
 

[VQDiffusionSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/vq_diffusion#diffusers.schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput) 
 or
 `tuple` 


 Parameters
 




* **t** 
 (
 `torch.long` 
 ) —
The timestep that determines which transition matrices are used.
* **x\_t** 
 (
 `torch.LongTensor` 
 of shape
 `(batch size, num latent pixels)` 
 ) —
The classes of each latent pixel at time
 `t` 
.
* **generator** 
 (
 `torch.Generator` 
 , or
 `None` 
 ) —
A random number generator for the noise applied to
 `p(x_{t-1} | x_t)` 
 before it is sampled from.
* **return\_dict** 
 (
 `bool` 
 ,
 *optional* 
 , defaults to
 `True` 
 ) —
Whether or not to return a
 [VQDiffusionSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/vq_diffusion#diffusers.schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput) 
 or
 `tuple` 
.




 Returns
 




 export const metadata = 'undefined';
 

[VQDiffusionSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/vq_diffusion#diffusers.schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput) 
 or
 `tuple` 




 export const metadata = 'undefined';
 




 If return\_dict is
 `True` 
 ,
 [VQDiffusionSchedulerOutput](/docs/diffusers/v0.23.0/en/api/schedulers/vq_diffusion#diffusers.schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput) 
 is
returned, otherwise a tuple is returned where the first element is the sample tensor.
 



 Predict the sample from the previous timestep by the reverse transition distribution. See
 [q\_posterior()](/docs/diffusers/v0.23.0/en/api/schedulers/vq_diffusion#diffusers.VQDiffusionScheduler.q_posterior) 
 for more details about how the distribution is computer.
 


## VQDiffusionSchedulerOutput




### 




 class
 

 diffusers.schedulers.scheduling\_vq\_diffusion.
 

 VQDiffusionSchedulerOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/schedulers/scheduling_vq_diffusion.py#L28)



 (
 


 prev\_sample
 
 : LongTensor
 



 )
 


 Parameters
 




* **prev\_sample** 
 (
 `torch.LongTensor` 
 of shape
 `(batch size, num latent pixels)` 
 ) —
Computed sample x\_{t-1} of previous timestep.
 `prev_sample` 
 should be used as next model input in the
denoising loop.


 Output class for the scheduler’s step function output.