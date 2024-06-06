# [](#reinforcement-learning-training-with-ddpo)使用 DDPO 进行强化学习训练

> 译者：[疾风兔X](https://github.com/jifnegtu)
>
> 项目地址：<https://huggingface.apachecn.org/docs/diffusers/training/ddpo>
>
> 原始地址：<https://huggingface.co/docs/diffusers/training/ddpo>



您可以通过使用 🤗 TRL 库和 🤗 Diffusers 进行强化学习来精细调整奖励函数的 Stable Diffusion。这是通过Black等人在[《利用强化学习训练扩散模型》（Training Diffusion Models with Reinforcement Learning）](https://arxiv.org/abs/2305.13301)中引入的去噪扩散策略优化（DDPO）算法完成的，该算法在🤗 TRL中使用[DDPOTrainer](https://huggingface.co/docs/trl/v0.8.6/en/trainer#trl.DDPOTrainer)实现。

有关更多信息，请查看 [DDPOTrainer](https://huggingface.co/docs/trl/v0.8.6/en/trainer#trl.DDPOTrainer)的API参考文档 和 [使用TRL与DDPO微调Stable Diffusion模型（Finetune Stable Diffusion Models with DDPO via TRL）](https://huggingface.co/blog/trl-ddpo) 博客文章。