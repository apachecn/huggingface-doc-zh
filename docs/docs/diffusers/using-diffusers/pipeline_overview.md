# 概述

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/using-diffusers/pipeline_overview>
>
> 原始地址：<https://huggingface.co/docs/diffusers/using-diffusers/pipeline_overview>


管道是一种端到端的类，通过将独立训练的模型和调度程序捆绑在一起，提供了一种快速、简单的方法来使用扩散系统进行推理。模型和调度程序的某些组合定义了特定的管道类型，例如
 [StableDiffusionXLPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)
 或者
 [StableDiffusionControlNetPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline)
 ，具有特定的能力。所有管道类型都继承自基础管道
 [DiffusionPipeline](/docs/diffusers/v0.23.0/en/api/pipelines/overview#diffusers.DiffusionPipeline)
 班级;通过任何检查点，它会自动检测管道类型并加载必要的组件。


本节演示如何使用特定管道，例如 Stable Diffusion XL、ControlNet 和 DiffEdit。您还将学习如何使用稳定扩散模型的精炼版本来加速推理、如何创建可重现的管道以及如何使用和贡献社区管道。