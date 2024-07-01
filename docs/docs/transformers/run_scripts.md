# 使用脚本训练

> 译者：[BrightLi](https://github.com/brightli)
>
> 项目地址：<https://huggingface.apachecn.org/docs/transformers/run_scripts>
>
> 原始地址：<https://huggingface.co/docs/transformers/run_scripts>

使用脚本训练
除了 🤗 Transformers [笔记本](https://huggingface.co/docs/transformers/notebooks)之外，还有示例脚本演示了如何使用 [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch)、[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow)或[JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax)为任务训练模型。

你还会发现我们在我们的[研究项目](https://github.com/huggingface/transformers/tree/main/examples/research_projects)中使用的脚本和主要是社区贡献的[遗留示例](https://github.com/huggingface/transformers/tree/main/examples/legacy)。这些脚本不是积极维护的，并且需要 🤗 Transformers 的特定版本，这很可能与库的最新版本不兼容。

示例脚本并不期望在每个问题上都能即插即用，你可能需要根据你要解决的问题来调整脚本。为了帮助你做到这一点，大多数脚本都完全展示了数据是如何预处理的，允许你根据用例需要进行编辑。

对于你想在示例脚本中实现的任何功能，请在提交 Pull Request 之前在[论坛](https://discuss.huggingface.co/)上或在[问题](https://github.com/huggingface/transformers/issues)中讨论。虽然我们欢迎修复错误，但我们不太可能合并添加更多功能但牺牲可读性的 Pull Request。

本指南将展示如何在[PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)和[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization)中运行一个示例摘要训练脚本。除非另有说明，所有示例都应该可以在两个框架上运行。

设置

为了成功运行最新版本的示例脚本，你必须在一个新的虚拟环境中从源代码**安装 🤗 Transformers**

```shell
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

对于旧版本的示例脚本，点击下面的切换按钮：
旧版本 🤗 Transformers 的示例
然后，将你当前的 🤗 Transformers 克隆切换到特定版本，例如 v3.5.1：

```shell
git checkout tags/v3.5.1
```

在设置了正确的库版本后，导航到你选择的示例文件夹并安装示例特定的要求：

```shell
pip install -r requirements.txt
```

**运行脚本**

Pytorch

示例脚本从 🤗 [Datasets](https://huggingface.co/docs/datasets/)库下载并预处理一个数据集。然后，脚本使用[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)在一个支持摘要的架构上微调数据集。以下示例展示了如何在 [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)数据集上微调[T5-small](https://huggingface.co/google-t5/t5-small)。由于 T5 的训练方式，T5 模型需要额外的 source_prefix 参数。这个提示让 T5 知道这是一个摘要任务。

```shell
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

TensorFlow

示例脚本从 🤗[Datasets](https://huggingface.co/docs/datasets/)库下载并预处理一个数据集。然后，脚本使用 Keras 在一个支持摘要的架构上微调数据集。以下示例展示了如何在[CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)数据集上微调[T5-small](https://huggingface.co/google-t5/t5-small)。由于 T5 的训练方式，T5 模型需要额外的 source_prefix 参数。这个提示让 T5 知道这是一个摘要任务。

```shell
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```

**分布式训练和混合精度**

Trainer 支持分布式训练和混合精度，这意味着你也可以在脚本中使用它。要启用这两个功能：

* 添加 fp16 参数以启用混合精度。
* 使用 nproc_per_node 参数设置要使用的 GPU 数量。

```shell
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

TensorFlow 脚本利用[MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)进行分布式训练，你不需要向训练脚本添加任何额外的参数。如果多个 GPU 可用，TensorFlow 脚本将默认使用它们。

**在 TPU 上运行脚本**

Pytorch

张量处理单元 (TPUs) 是专门为加速性能而设计的。PyTorch 通过[XLA](https://www.tensorflow.org/xla)深度学习编译器支持 TPU（在[这里](https://github.com/pytorch/xla/blob/master/README.md)查看更多详细信息）。要使用 TPU，启动 xla_spawn.py 脚本并使用 num_cores 参数设置你想使用的 TPU 核心数。

```shell
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

TensorFlow

张量处理单元 (TPUs) 是专门为加速性能而设计的。TensorFlow 脚本利用 [TPUStrategy](https://www.tensorflow.org/guide/distributed_training#tpustrategy)在 TPU 上进行训练。要使用 TPU，将 TPU 资源的名称传递给 tpu 参数。

```shell
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```

**使用 🤗 Accelerate 运行脚本**

🤗 [Accelerate](https://huggingface.co/docs/accelerate)是一个仅限 PyTorch 的库，它提供了一种统一的方法来在几种类型的设置上训练模型（仅限 CPU、多个 GPU、TPU），同时保持对 PyTorch 训练循环的完全可见性。如果你还没有安装 🤗 Accelerate，请确保安装了它：

>注意：由于 Accelerate 正在快速发展，必须安装加速的 git 版本才能运行脚本
>
>```shell
>pip install git+https://github.com/huggingface/accelerate
>```

你需要使用 run_summarization_no_trainer.py 脚本，而不是 run_summarization.py 脚本。支持 🤗 Accelerate 的脚本会在文件夹中有一个 task_no_trainer.py 文件。首先运行以下命令来创建并保存配置文件：

```shell
accelerate config
```

测试你的设置以确保其配置正确：

```shell
accelerate test
```

现在你已经准备好启动训练了：

```shell
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

**使用自定义数据集**

只要数据集是 CSV 或 JSON Line 文件，摘要脚本就支持自定义数据集。当你使用自己的数据集时，你需要指定几个额外的参数：

* train_file 和 validation_file 指定你的训练和验证文件的路径。
* text_column 是要总结的输入文本。
* summary_column 是要输出的目标文本。

使用自定义数据集的摘要脚本将如下所示：

```shell
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

**测试脚本**

在提交可能需要数小时才能完成的整个数据集之前，通常最好在较少的数据集示例上运行你的脚本以确保一切按预期工作。使用以下参数将数据集截断到最大样本数：

* max_train_samples
* max_eval_samples
* max_predict_samples

```shell
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

并非所有示例脚本都支持 max_predict_samples 参数。如果你不确定你的脚本是否支持此参数，添加 -h 参数以检查：

```shell
examples/pytorch/summarization/run_summarization.py -h
```

**从检查点恢复训练**

另一个有用的选项是能够从前一个检查点恢复训练。这将确保如果你的训练被中断，你可以从中断的地方继续，而不必重新开始。有两种方法可以从检查点恢复训练。
第一种方法是使用 output_dir previous_output_dir 参数从前一个检查点恢复训练，该检查点存储在 output_dir 中。在这种情况下，你应该移除 overwrite_output_dir：

```shell
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir previous_output_dir \
    --predict_with_generate
```

第二种方法是使用 resume_from_checkpoint path_to_specific_checkpoint 参数从特定检查点文件夹恢复训练。

```shell
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate
```

**分享你的模型**

所有脚本都可以将你的最终模型上传到[Model Hub](https://huggingface.co/models)。在开始之前，请确保你已经登录 Hugging Face：

```shell
huggingface-cli login
```

然后在脚本中添加 push_to_hub 参数。此参数将使用你的 Hugging Face 用户名和 output_dir 中指定的文件夹名称创建一个仓库。
要为你的仓库指定一个具体名称，请使用 push_to_hub_model_id 参数添加它。该仓库将自动列在你的命名空间下。
以下示例展示了如何上传具有特定仓库名称的模型：

```shell
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```