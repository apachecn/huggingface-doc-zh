# Spectrogram Diffusion

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/api/pipelines/spectrogram_diffusion>
>
> 原始地址：<https://huggingface.co/docs/diffusers/api/pipelines/spectrogram_diffusion>



[Spectrogram Diffusion](https://huggingface.co/papers/2206.05408) 
 is by Curtis Hawthorne, Ian Simon, Adam Roberts, Neil Zeghidour, Josh Gardner, Ethan Manilow, and Jesse Engel.
 



*An ideal music synthesizer should be both interactive and expressive, generating high-fidelity audio in realtime for arbitrary combinations of instruments and notes. Recent neural synthesizers have exhibited a tradeoff between domain-specific models that offer detailed control of only specific instruments, or raw waveform models that can train on any music but with minimal control and slow generation. In this work, we focus on a middle ground of neural synthesizers that can generate audio from MIDI sequences with arbitrary combinations of instruments in realtime. This enables training on a wide range of transcription datasets with a single model, which in turn offers note-level control of composition and instrumentation across a wide range of instruments. We use a simple two-stage process: MIDI to spectrograms with an encoder-decoder Transformer, then spectrograms to audio with a generative adversarial network (GAN) spectrogram inverter. We compare training the decoder as an autoregressive model and as a Denoising Diffusion Probabilistic Model (DDPM) and find that the DDPM approach is superior both qualitatively and as measured by audio reconstruction and Fréchet distance metrics. Given the interactivity and generality of this approach, we find this to be a promising first step towards interactive and expressive neural synthesis for arbitrary combinations of instruments and notes.* 




 The original codebase can be found at
 [magenta/music-spectrogram-diffusion](https://github.com/magenta/music-spectrogram-diffusion) 
.
 



![img](https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/architecture.png)




 As depicted above the model takes as input a MIDI file and tokenizes it into a sequence of 5 second intervals. Each tokenized interval then together with positional encodings is passed through the Note Encoder and its representation is concatenated with the previous window’s generated spectrogram representation obtained via the Context Encoder. For the initial 5 second window this is set to zero. The resulting context is then used as conditioning to sample the denoised Spectrogram from the MIDI window and we concatenate this spectrogram to the final output as well as use it for the context of the next MIDI window. The process repeats till we have gone over all the MIDI inputs. Finally a MelGAN decoder converts the potentially long spectrogram to audio which is the final result of this pipeline.
 




 Make sure to check out the Schedulers
 [guide](../../using-diffusers/schedulers) 
 to learn how to explore the tradeoff between scheduler speed and quality, and see the
 [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) 
 section to learn how to efficiently load the same components into multiple pipelines.
 


## SpectrogramDiffusionPipeline




### 




 class
 

 diffusers.
 

 SpectrogramDiffusionPipeline




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/utils/dummy_transformers_and_torch_and_note_seq_objects.py#L5)



 (
 


 \*args
 


 \*\*kwargs
 




 )
 




#### 




 \_\_call\_\_




 (
 


 \*args
 


 \*\*kwargs
 




 )
 




 Call self as a function.
 


## AudioPipelineOutput




### 




 class
 

 diffusers.
 

 AudioPipelineOutput




[<
 

 source
 

 >](https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/pipeline_utils.py#L124)



 (
 


 audios
 
 : ndarray
 



 )
 


 Parameters
 




* **audios** 
 (
 `np.ndarray` 
 ) —
List of denoised audio samples of a NumPy array of shape
 `(batch_size, num_channels, sample_rate)` 
.


 Output class for audio pipelines.