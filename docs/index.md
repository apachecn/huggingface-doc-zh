<!-- Row Highlight Javascript -->
<script type="text/javascript">
	window.onload=function(){
	var tfrow = document.getElementById('tfhover').rows.length;
	var tbRow=[];
	for (var i=1;i<tfrow;i++) {
		tbRow[i]=document.getElementById('tfhover').rows[i];
		tbRow[i].onmouseover = function(){
		  this.style.backgroundColor = '#f3f8aa';
		};
		tbRow[i].onmouseout = function() {
		  this.style.backgroundColor = '#ffffff';
		};
	}
};
</script>

<style type="text/css">
table.tftable {font-size:12px;color:#333333;width:100%;border-width: 1px;border-color: #ebab3a;border-collapse: collapse;}
table.tftable th {font-size:36px;background-color:#e6983b;border-width: 1px;padding: 8px;border-style: solid;border-color: #ebab3a;text-align:center; height:120px;}
table.tftable tr {background-color:#ffffff;}
table.tftable td {font-size:12px;border-width: 1px;padding: 8px;border-style: solid;border-color: #ebab3a;text-align:center;}
</style>


<table id='tfhover' class='tftable' border='1'>
    <tr><th colspan='3'>Hugging Face - 文档库(中文版)</th></tr>
<tr>
    <td rowspan='2'>
        <h2><a href='https://huggingface.apachecn.org/docs/hub' target='_blank'>Hub</a></h2>
        <p>拥抱面线上的主机基于GIT的模型，数据集和空间。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/transformers' target='_blank'>Transformers</a></h2>
        <p>Pytorch，Tensorflow和Jax的最先进的ML。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/diffusers' target='_blank'>Diffusers</a></h2>
        <p>Pytorch中图像和音频产生的最新扩散模型。</p>
    </td>
</tr><tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/datasets' target='_blank'>Datasets</a></h2>
        <p>访问并共享数据集，以获取计算机视觉，音频和NLP任务。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.orghttps://www.gradio.app/docs/' target='_blank'>Gradio</a></h2>
        <p>在仅几行Python中构建机器学习演示和其他Web应用程序。</p>
    </td>
</tr><tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/huggingface_hub' target='_blank'>Hub Python Library</a></h2>
        <p>HF集线器的客户库库：从Python运行时管理存储库。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/huggingface.js' target='_blank'>Huggingface.js</a></h2>
        <p>JS库的集合与拥抱面交互，其中包括TS类型。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/transformers.js' target='_blank'>Transformers.js</a></h2>
        <p>社区图书馆可以从浏览器中的变压器中运行预算模型。</p>
    </td>
</tr><tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/api-inference' target='_blank'>Inference API</a></h2>
        <p>使用我们的免费推理API轻松地尝试超过200K模型。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/inference-endpoints' target='_blank'>Inference Endpoints</a></h2>
        <p>在专用的，完全管理的基础架构上轻松将模型放在生产中。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/peft' target='_blank'>PEFT</a></h2>
        <p>大型型号的参数有效的明迹方法</p>
    </td>
</tr><tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/accelerate' target='_blank'>Accelerate</a></h2>
        <p>轻松训练和使用具有多GPU，TPU，混合精液的Pytorch型号。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/optimum' target='_blank'>Optimum</a></h2>
        <p>使用易于使用的硬件优化工具对HF变压器进行快速训练和推断。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/optimum-neuron' target='_blank'>AWS Trainium & Inferentia</a></h2>
        <p>使用AWS Trainium和AWS推理训练和部署变压器和扩散器。</p>
    </td>
</tr><tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/tokenizers' target='_blank'>Tokenizers</a></h2>
        <p>快速引导者，针对研究和生产进行了优化。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/evaluate' target='_blank'>Evaluate</a></h2>
        <p>评估和报告模型性能更容易，更标准化。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/tasks' target='_blank'>Tasks</a></h2>
        <p>有关ML任务的所有事情：演示，用例，模型，数据集等等！</p>
    </td>
</tr><tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/datasets-server' target='_blank'>Datasets-server</a></h2>
        <p>API访问所有拥抱面轮数据集的内容，元数据和基本统计信息。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/trl' target='_blank'>TRL</a></h2>
        <p>通过强化学习训练变压器语言模型。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/sagemaker' target='_blank'>Amazon SageMaker</a></h2>
        <p>使用Amazon Sagemaker和拥抱Face DLC培训和部署变压器模型。</p>
    </td>
</tr><tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/timm' target='_blank'>timm</a></h2>
        <p>最先进的计算机视觉模型，层，优化器，培训/评估和实用程序。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/safetensors' target='_blank'>Safetensors</a></h2>
        <p>简单，安全的方法，可以安全，快速地存储和分发神经网络。</p>
    </td>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/text-generation-inference' target='_blank'>Text Generation Inference</a></h2>
        <p>工具包可供大型语言模型。</p>
    </td>
</tr>
<tr>
    <td rowspan='1'>
        <h2><a href='https://huggingface.apachecn.org/docs/autotrain' target='_blank'>AutoTrain</a></h2>
        <p>Autotrain API和UI</p>
    </td>
</tr>
</table>

## Hugging Face 中文文档

> 原文：[Hugging Face Documentation](https://huggingface.co/docs)
>
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)
>
> 计算机科学中仅存在两件难事：缓存失效和命名。——菲尔·卡尔顿

* [在线阅读(中文)](https://huggingface.apachecn.org) | [英文地址](https://huggingface.co/docs)
* [ApacheCN 中文翻译组 713436582](https://qm.qq.com/cgi-bin/qm/qr?k=5u_aAU-YlY3fH-m8meXTJzBEo2boQIUs&jump_from=webapi&authKey=CVZcReMt/vKdTXZBQ8ly+jWncXiSzzWOlrx5hybX5pSrKu6s0fvGX54+vHHlgYNt)
* [ApacheCN 组织资源](https://docs.apachecn.org/)

## 贡献者

* [@片刻小哥哥](https://github.com/jiangzhonglian)

## 赞助我们

<img src="http://data.apachecn.org/img/about/donate.jpg" alt="微信&支付宝" />
