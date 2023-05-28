## 写在前面

会总结部署的相关知识点以及代码，最晚每周更新一次。

github仓库更新需要整理格式比较慢，可以先看ai.oldpan.me中的一些更新，后续会同步到github仓库中。

## 部署之路
之前也写过不少关于部署的文章，但技术更迭很快，写出来的内容也不可能覆盖所有地方，而且很多小伙伴也会问我一些新的问题。干脆写一个比较全的总结版，挂到github和自己的博客上，之后有新想法了、新点子了，也会**随时更新**。大家想找学习路线看这一篇跟着学就对了。当然我也不是什么大佬，**难免存在纰漏，望大家及时提醒，我好及时纠正。**
> 尽管说是部署之路，重点是部署，但多多少少也要会一些模型训练和设计，只有理解模型的具体使用场景、使用要求，才能够使用更多方式去优化这个模型

整理了下，涉及到**训练、优化、推理、部署**这几块，暂时把我能想到的都列出来了，后续也会不断更新（立个flag）：

- 首先要有模型
- 模型转换那些事儿
- 模型优化那些事儿
- 了解很多推理框架
- 开始部署吧！
- 高性能计算/AI编译器

本文也仅仅是**总结**的作用，尽可能把东西都列全。每部分的细节，会总结相关文章再展开描述，可能是我写的，也可能是其他人优秀的文章。<br />附上之前的两篇文章：

- [老潘的思考（AI部署、方向、就业）](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247489106&idx=1&sn=b0c02cdfbf6cda89c3a6a1b9d36b6d8c&chksm=cf108e57f86707411fba716c3fbeb511930ea1e4932f4754c9e9a361fcf69d320c27b01dd31e&token=359604848&lang=zh_CN#rd)
- [老潘的AI部署以及工业落地学习之路](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247487312&idx=1&sn=54215ca063de0ef297dd0f5eb9d11007&chksm=cf109755f8671e43c0629820d56b45ff280c6733c0b9dff85136138a8a5914f9e8fa897e0b68&token=359604848&lang=zh_CN#rd)
## 学习建议
诚然，我们想学的东西很多。但很多方向我们不可能**每个都精通、像单独领域中的大佬一样**，因为我们每个人的时间精力问题，我们可以雨露均沾，但是不可能样样大神，学的再多，也是沧海一粟。<br />于是最好的办法就是**抓重点学习**，比如：

- 一些特别基础的知识，其他知识点都依赖它，这种基础是必须要学的，也是性价比最高的
- 项目中的要学习的一些知识点，自己之前没有接触过的，需要边学边做，这种从0到1是收获最快的
- 一些JD中的岗位需求，找人自己感兴趣并且想转方向的，有心去参考下

比如商汤的2023年HPC团队招聘[计划](https://zhuanlan.zhihu.com/p/595902377)，一些岗位需求也可以参照学习：<br />![部署相关要求](https://image.oldpan.me/部署相关要求.jpg)<br />很多需要学习的点，也一一列出来了，又或者你想学习高性能计算：<br />![高性能JD要求](https://image.oldpan.me/高性能JD要求.jpg)<br />这些JD已经把需要你的，**你需要的技能点都列出来了，自己按需点满即可**。<br />学习是自己的事儿，自己想干啥想学啥，最好有个**明确的目标**，不要瞎学不要浪费时间。

### 学习FAQ
#### 如果想做模型压缩、量化这些，需要先做算法相关的吗
肯定懂算法是最好的，毕竟模型压缩和量化是**对模型做事情**，搞算法的已经提前了解了模型的结构、op特点（训练时和推理时）、op种类、特殊结构等等。模型压缩、量化或者转换的时候都要经过这些op，难免会遇到很多问题，知道op细节对于解决这个问题肯定是有帮助的（比如BN在训练时和推理时表现不一样，转化的时候就需要注意），有很多坑。<br />不过都是可以慢慢学的，转换的时候再看也行，遇到不会的算子再去查阅。




## 首先要有模型
搞深度学习部署最重要的当然是**模型**，大部分的场景都是靠模型和算法共同实现的，这里的算法又包含逻辑部分（俗称if-else）、算法部分（各种各样与场景相关的算法，比如滤波算法blabla等）、图像处理部分（我这里主要指的是视觉场景，比如二值化、最大外接矩形等等）等等。当然这是我的经验，可能其他场景有不同的组合。<br />模型当然是**最重要**的，因为**模型效果或者速度不行**的话，用传统图像方法或者逻辑（if-else）捞case是比较困难的，在某些场景中模型可以直击要害，抓需求重点去学习。
> 举个极端的例子，比如你要识别一个人的动作，这个人跳起来然后落下来算一个，也就是无绳跳跃。如果训练一个检测模型，首先检测这个人，然后分两类，跳起来算一类，落下来算一类，产出一个分两类的检测模型，那这个问题就好解多了，逻辑只需要根据模型结果来判断即可。但如果没有这个模型的话， 你可能会通过检测人的模型+姿态检测+时序算法来判断这个人到底有没有跳跃一次了。

知道模型的重要性了，关于怎么收集数据怎么训练这里姑且先不谈了。涉及到部署的话，模型就不能只看精度了，了解些部署的算法工程师在设计模型的时候通常也会注意这一点，**实用好用（好训练好部署，比如YOLOv5、V7）**才是重点。<br />实际应用中的模型，往往有以下一些要求：

- 模型精度达到要求，当然这是前提
- 模型大小符合要求，部署的场景能放得下这个模型的权重信息
- 模型op符合要求，部署的平台库可以支持这个模型的op
- 模型的速度符合要求，这时候需要模型的op和设计共同决定

模型的设计需要考虑计算量、参数量、访存量、内存占用等等条件。可以参考这篇：[深度学习模型大小与模型推理速度的探讨](https://zhuanlan.zhihu.com/p/411522457)，内容很详实！<br />相关参考：

- [深度学习模型大小与模型推理速度的探讨](https://zhuanlan.zhihu.com/p/411522457)
### 自己造模型
有了想法，有了目标，有了**使用场景**，那就训练一个模型吧！
#### 如何设计模型
关于设计模型的思路，大部分人会直接参考GITHUB上找开源的模型结构，这个没问题，根据**场景**选对就行：

- 首先确定任务类型，分类、检测、分割、识别、NLP
- 根据任务特点选择（数据集分布，数据特点）
- 移动端和桌面端考虑的情况不同，比如移动端更看重计算量，而桌面端更看重访存。两个平台都有能用的backbone
- 待补充

如果自己对模型优化也有特定了解的话，可以针对**特定平台**设计一些对**硬件友好的结构**（这里举个例子，比如重参数化的RepVGG），一般来说大公司会有专门的高性能计算工程师和算法工程师对齐，当然小公司的话，都要干了555。
#### 模型结构
根据**经验和理论知识**来设计模型的结构（比如dcn系列、比如senet中的se模块、yolo中的fcos模块）；根据任务来设计模型；NAS搜索模型；<br />有很多优秀的模型块结构，属于即插即用的那种，这里收集一些常见的：

- DCNv2系列，已经比较成熟，
- [DCNv3](https://github.com/OpenGVLab/InternImage)（InternImage）
- 重参数化结构（[RepVGG](https://github.com/DingXiaoH/RepVGG)）
- 待补充
#### 训练框架
训练框架这里指的不是**Pytorch或者TensorFlow**，而是类似于mmlab中各种任务框架，比如**mmyolo、mmdetection这种**的。<br />你如果想自己训练一个检测模型，大概是要自己基于Pytorch去重新实现一些方法：数据处理、模型搭建、训练优化器、debug可视化等等。比较烦而且浪费时间，如果有这些已经完善的轮子你自己使用，个人觉着是比较舒服的，但是当然**有优劣**：

- 自己实现可能自定义性更好，但是比较费时间去造轮子
- 使用轮子，上手轮子可能也比较花时间，尤其是比较重的轮子，上手周期也挺长
- 使用现有的轮子，这个轮子不断更新的话，你也可以享受到这个新功能或者实用的功能，不用你自己实现或者移植

一些大的轮子可以参考：

- [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- [https://github.com/open-mmlab/mmyolo/](https://github.com/open-mmlab/mmyolo/tree/damo-yolo)

一些小的轮子可以参考：

- [https://github.com/xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)
- [https://github.com/open-mmlab/mmyolo/tree/damo-yolo](https://github.com/open-mmlab/mmyolo/tree/damo-yolo)

我个人之前使用的是`detectron2`，但是官方貌似不怎么维护了，因为我打算增加一些新功能，但是没有时间搞。后来发现mmlab2.0出来了，之前那会使用过mmdet1.0版本，感觉封装的太深了，训练的模型导出很费劲儿就没怎么用。后来2.0简单看了看上手稍微简单了些，打算切到mmlab系列，主要原因还是支持比较丰富，各种pr很多，有的模型不需要你移植就已经有人移过去了，省去很多造轮子的时间。<br />后面我也会比较下detectron2和mmdet2.0这俩框架。
#### 导出模型
自己的模型在训练完毕后，如果不是直接在训练框架中推理（好处就是模型精度和你eval的时候一致，而且不需要花时间转模型的格式，坏处就是模型运行速度不能保证，而且python环境比较依赖python解释器），大部分是要导出的。<br />导出的格式有很多，我也不一一列了，可以参考[Netron](https://github.com/lutzroeder/netron)模型Viewer的介绍：<br />![Netron支持的模型列表](https://image.oldpan.me/Netron支持的模型列表.jpg)<br />导出模型也就是模型转换，只不过因为是自己训练的，一般会从自己训练框架为起点进行转换，可以避免一些中间转换格式。<br />比如我经常使用Pytorch，如果训练完模型后，需要转为TensorRT的模型，正常来说可能会Pytorch->ONNX->TensorRT。但是这样会经过ONNX这个中间模型结构，可能会对最终模型精度产生不确定性，我可能也会直接从Pytorch转为TensorRT：

- torch2trt
- [torch.fx2trt](https://pytorch.org/docs/stable/fx.html?highlight=fx#module-torch.fx)
- [torchscript2trt](https://github.com/pytorch/TensorRT)

因为Pytorch支持比较好，所以能这样搞，支持不好的，可能还要走ONNX这个路子。
#### nas搜索
nas就是搜索一个在你平台上一个速度最优的模型结构，目前我只了解过DAMO相关的一个搜索，还未深度尝试：

- DAMO-backbone的搜索指南 [https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/scripts/damo-yolo/Tutorial_NAS_for_DAMO-YOLO_cn.md](https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/scripts/damo-yolo/Tutorial_NAS_for_DAMO-YOLO_cn.md)
### 别人的模型
直接拿**现成模型**来用的情况也很多见：

- 自己懒得训，或者没有数据
- 人家这个模型训得好，权重牛逼
- 你就是负责帮忙部署模型的，训练模型不管你的事儿

不知道大家有没有这种情况，在网上寻找他人训练好的合适的模型的时候，有一种探索的感觉，类似于开盲盒，一般会看模型在某个平台的benchmark速度或者某个任务的精度，以及模型本身的一些特点啥的，我比较喜欢mediapipe中的[MODEL CARD](https://drive.google.com/file/d/10WlcTvrQnR_R2TdTmKw0nkyRLqrwNkWU/preview)介绍模型的方法，详细介绍了该模型的输入输出，训练集，特点，模型性能啥的：<br />![mediapipe-blazepose-GHUM3D-model-card](https://image.oldpan.me/mediapipe-blazepose-GHUM3D-model-card.jpg)<br />个人建议，如果自己训练的模型想要被别人更好地使用，想开源模型贡献的话，也可以参考mediapipe写个类似的[model-card](https://arxiv.org/pdf/1810.03993.pdf)，介绍下你的模型哈。<br />回到正题，拿到模型的第一点，当然是观察**模型的结构**。模型结构主要是为了分析这个模型是基于哪个backbone、哪个框架训练的，可以知道模型的大概，心里有数：<br />![能看出来是什么模型结构吗](https://image.oldpan.me/能看出来是什么模型结构吗.jpg)<br />一般来说要看：

- 模型的输入输出大小、shape、精度类型
- 模型包含的op算子类型，op名称
- 模型的大小，权重精度
- 模型在哪个平台训练的
- 模型的权重和结构格式

一般经验丰富的算法工程师，大概看一下模型结构、模型op类型就知道这个模型坑不坑了。
### benchmark
benchmark分两种，一种是评测模型的精度指标，一些常见的指标：

- AP、mAP、AR
- F1 score：精确率和召回率的调和平均值
- AUC：ROC 曲线下的面积，表示正样本的分数高于负样本的分数的概率

不列了，不同任务模型评价的指标也不一样，明白这个意思就行。<br />另一种是benchmark模型的性能，在确认**模型精度**符合要求之后，接下来需要看模型的几个指标：

- 模型latency，充分预热后跑一次模型的耗时
- 模型吞吐量，1s内可以模型可以跑多少次
- 模型附带的预处理和后处理耗时，整体耗时
- 模型输入输出相关耗时，传输耗时

相关概念：
> Latency refers to the amount of time it takes for the model to make a prediction after receiving input. In other words, it's the time between the input and the corresponding output. Latency is an important metric for real-time applications where fast predictions are necessary, such as in self-driving cars or speech recognition systems.

> Throughput refers to the number of predictions that a model can make in a given amount of time. It's usually measured in predictions per second (PPS). Throughput is an important metric for large-scale applications where the model needs to process a large amount of data quickly, such as in image classification or natural language processing.

举个TensorRT的例子：
```bash
[01/08/2023-10:47:32] [I] Average on 10 runs - GPU latency: 4.45354 ms - Host latency: 5.05334 ms (enqueue 1.61294 ms)
[01/08/2023-10:47:32] [I] Average on 10 runs - GPU latency: 4.46018 ms - Host latency: 5.06121 ms (enqueue 1.61682 ms)
[01/08/2023-10:47:32] [I] Average on 10 runs - GPU latency: 4.47092 ms - Host latency: 5.07136 ms (enqueue 1.61714 ms)
[01/08/2023-10:47:32] [I] Average on 10 runs - GPU latency: 4.48318 ms - Host latency: 5.08337 ms (enqueue 1.61753 ms)
[01/08/2023-10:47:32] [I] Average on 10 runs - GPU latency: 4.49258 ms - Host latency: 5.09268 ms (enqueue 1.61719 ms)
[01/08/2023-10:47:32] [I] Average on 10 runs - GPU latency: 4.50391 ms - Host latency: 5.10193 ms (enqueue 1.43665 ms)
[01/08/2023-10:47:32] [I] 
[01/08/2023-10:47:32] [I] === Performance summary ===
[01/08/2023-10:47:32] [I] Throughput: 223.62 qps
[01/08/2023-10:47:32] [I] Latency: min = 5.00714 ms, max = 5.33484 ms, mean = 5.06326 ms, median = 5.05469 ms, percentile(90%) = 5.10596 ms, percentile(95%) = 5.12622 ms, percentile(99%) = 5.32755 ms
[01/08/2023-10:47:32] [I] Enqueue Time: min = 0.324463 ms, max = 1.77274 ms, mean = 1.61379 ms, median = 1.61826 ms, percentile(90%) = 1.64294 ms, percentile(95%) = 1.65076 ms, percentile(99%) = 1.66541 ms
[01/08/2023-10:47:32] [I] H2D Latency: min = 0.569824 ms, max = 0.60498 ms, mean = 0.58749 ms, median = 0.587158 ms, percentile(90%) = 0.591064 ms, percentile(95%) = 0.592346 ms, percentile(99%) = 0.599182 ms
[01/08/2023-10:47:32] [I] GPU Compute Time: min = 4.40759 ms, max = 4.73703 ms, mean = 4.46331 ms, median = 4.45447 ms, percentile(90%) = 4.50464 ms, percentile(95%) = 4.5282 ms, percentile(99%) = 4.72678 ms
[01/08/2023-10:47:32] [I] D2H Latency: min = 0.00585938 ms, max = 0.0175781 ms, mean = 0.0124573 ms, median = 0.0124512 ms, percentile(90%) = 0.013855 ms, percentile(95%) = 0.0141602 ms, percentile(99%) = 0.0152588 ms
[01/08/2023-10:47:32] [I] Total Host Walltime: 3.01404 s
[01/08/2023-10:47:32] [I] Total GPU Compute Time: 3.00827 s
[01/08/2023-10:47:32] [W] * GPU compute time is unstable, with coefficient of variance = 1.11717%.
```
benchmark的场景和方式也有很多。除了本地离线测试，云端的benchmark也是需要的，举个压测triton-inference-server的例子：
```bash
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 45000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client: 
    Request count: 32518
    Throughput: 200.723 infer/sec
    Avg latency: 4981 usec (standard deviation 204 usec)
    p50 latency: 4952 usec
    p90 latency: 5044 usec
    p95 latency: 5236 usec
    p99 latency: 5441 usec
    Avg HTTP time: 4978 usec (send/recv 193 usec + response wait 4785 usec)
  Server: 
    Inference count: 32518
    Execution count: 32518
    Successful request count: 32518
    Avg request latency: 3951 usec (overhead 46 usec + queue 31 usec + compute 3874 usec)

  Composing models: 
  centernet, version: 
      Inference count: 32518
      Execution count: 32518
      Successful request count: 32518
      Avg request latency: 3512 usec (overhead 14 usec + queue 11 usec + compute input 93 usec + compute infer 3370 usec + compute output 23 usec)

  centernet-postprocess, version: 
      Inference count: 32518
      Execution count: 32518
      Successful request count: 32518
      Avg request latency: 96 usec (overhead 15 usec + queue 8 usec + compute input 7 usec + compute infer 63 usec + compute output 2 usec)

  image-preprocess, version: 
      Inference count: 32518
      Execution count: 32518
      Successful request count: 32518
      Avg request latency: 340 usec (overhead 14 usec + queue 12 usec + compute input 234 usec + compute infer 79 usec + compute output 0 usec)

  Server Prometheus Metrics: 
    Avg GPU Utilization:
      GPU-6d31bfa8-5c82-a4ec-9598-ce41ea72b7d2 : 70.2407%
    Avg GPU Power Usage:
      GPU-6d31bfa8-5c82-a4ec-9598-ce41ea72b7d2 : 264.458 watts
    Max GPU Memory Usage:
      GPU-6d31bfa8-5c82-a4ec-9598-ce41ea72b7d2 : 1945108480 bytes
    Total GPU Memory:
      GPU-6d31bfa8-5c82-a4ec-9598-ce41ea72b7d2 : 10504634368 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 200.723 infer/sec, latency 4981 usec
```
benchmark模型的速度**还是很重要的**，平常我们可能习惯单压一个模型，也就是**一个请求接一个连续不断**请求这个模型然后看耗时以及qps，这样可以大概评估出一个模型的性能。但实际中要考虑很多种情况：

- 计算资源是不是这个模型独享，会不会有其他模型一起抢占
- 该模型1s中会有多少请求同时发过来
- 该模型请求的输入图像尺寸是否每次一致（如果模型是动态尺寸）

一般可以用很多压测工具对模型使用场景进行压测来观察模型的性能，但最好是实际去测一下跑一下，感受一下实际场景的请求量。
## 开始部署吧！
有了符合要求的模型、并且模型转换好之后，就可以集成到对应的框架中了。<br />本地部署其实考虑比较多就是**推理后端，或者说推理框架**。你要根据平台选择开源的或者自己研发的推理框架，**平台很多（PC端、移动端、服务器、边缘端侧、嵌入式），**一般来说有：

- 手机（苹果的ANE、高通的DSP、以及手机端GPU）
- 服务端（最出名的英伟达，也有很多国产优秀的GPU加速卡）
- 电脑（Intel的cpu，我们平时使用的电脑都可以跑）
- 智能硬件（比如智能学习笔、智能台灯、智能XXX）
- VR/AR/XR设备（一般都是高通平台，比如Oculus Quest2）
- 自动驾驶系统（英伟达和高通）
- 各种开发板（树莓派、rk3399、NVIDIA-Xavier、orin）
- 各种工业装置设备（各种五花八门的平台，不是很了解）

坑都不少，需要学习的也比较杂，毕竟在某一个平台部署，这个平台的相关知识相关信息也要理解，不过有一些经验是可以迁移的，因此经验也比较重要，什么AI部署、AI工程化、落地都是一个概念。能让模型在某个平台顺利跑起来就行。<br />确定好平台就可以确定推理框架了：

- CPU端：ONNXruntime、OpenVINO、libtorch、Caffe、NCNN
- GPU端：TensorRT、OpenPPL、AITemplate
- 通吃：TVM、OpenPPL
- 其他平台

当然，如果你自己有实力或者有时间的话，也可以自己写一个推理框架，能学到很多东西！<br />![自己学推理框架难不难](https://image.oldpan.me/自己学推理框架难不难.jpg)
### 本地部署
本地部署就是模型在本地跑，利用本地的计算资源。比如你的笔记本电脑有显卡，然后你想做一个可以检测手的软件，通过exe安装，安装好之后使用笔记本的**显卡或者CPU**去跑，这个模型会根据笔记本的硬件情况选择CPU还是GPU跑（当然你得自己写代码去检测有哪些硬件可以使用）。又比如说你有一个Ubuntu的服务器，插上一些摄像头完成一些工业项目，插上显卡在显卡上跑模型等等。<br />本地部署可以参考的项目：

- triton-inference-server-C-API

本地部署需要考虑的一些点：

- 客户端和服务的设计
- 资源分配，本地部署往往资源有限，如何分配模型优先级，模型内存显存占用是比较重要的问题
- 模型加密，本地部署的模型当然是放在本地，想要不被竞品发现模型的结构信息和权重，当然要进行加密，加密的方法很多，对模型文件流进行加密、对权重加密、对解密代码加密等等
### 服务器部署相关
服务器模型部署和本地部署有类似的地方，但涉及到的知识点更多，面也更广。很多知识点和模型已经没什么关系了，我们其实也不需要深入理解架构方面的东西，简单了解一下即可，架构方面可以由公司内部专门的op去处理。<br />首先，可部署的模型种类更多，你有了服务器，一般来说基本都是NVIDIA的，基本什么模型都能跑，基本什么服务都会搞上去，甚至python模型扔上去也行，当然性能不是最优的。除了模型的优化外，会涉及到模型的调度，自动扩缩，模型pack等等，大部分服务端相关的操作，不用我们参与，只需要提供模型或者前后处理代码即可，但有些我们是可以参与的。<br />![Chatgpt-相关提问](https://image.oldpan.me/Chatgpt-相关提问.jpg)<br />我个人参与过一些服务端部署，基于K8s那套（仅仅是了解并没有实际使用过），说说咱们模型端需要做的事儿吧：

- 模型首先需要线上支持，线上无非也就是运行一个服务端推理框架，类似于triton的，当然也可以用triton，你的模型基本都要进行转换才能用，最好不要原生的pth模型依赖pytorch环境去跑，最基本的转成onnx，或者其他格式，方便统一管理
- 模型速度当然是越快越好，在某些pipeline中慢的模型会导致木桶效应
- 线上有的模型更倾向于组batch，想想你的模型怎么设计可以使qps最高
- 当然也有要求dynamic输入的模型，设计时候也需要注意
- 模型的体积大小一般没什么限制，如果做model pack的话，多个模型放一张加速卡可能会有资源不够的情况
- 模型的输入输出IO可能需要注意下，模型输入是float或者uint8，对于客户端发送图片数据的大小就有要求了
- 模型是否稳定或者安全，也就是模型是否会core，这个其实主要看推理引擎，一般我们用的推理引擎问题不大（比如TensorRT，很多人用过），很有可能前后处理或者数据传输的时候有问题
- 未完待续
#### 要学习的东西

- flask 比较简单的模型部署 可以简单先尝试下
- workflow 可以基于此后端搭建自定义的http服务
- triton inference server 目前开源最好的TensorRT推理框架，支持在线和离线推理
- triton服务器客户端-python自己设计
- triton服务器客户端-C++自己设计
- 服务器预处理、后处理

相关参考：

- [https://www.nvidia.cn/on-demand/session/gtcfall22-a41246/](https://www.nvidia.cn/on-demand/session/gtcfall22-a41246/)
- Maximizing GPU utilization in Large-Scale Machine Learning Infrastructure
### 嵌入式部署
嵌入式部署和本地部署类似，不过嵌入式的**平台硬件**相比普通电脑或者插加速卡的工业机器要弱不少。<br />一般都是嵌入式板卡一类的，比较好的有英伟达的jetson orin系列，比较差的有单片机stm32这类。<br />![chatgpt相关提问](https://image.oldpan.me/chatgpt相关提问.jpg)<br />嵌入式或者说手机端部署可以说是差不多，特点就是计算资源比较紧张，功耗往往都给的很低，性能不强的同时要求很多模型需要达到实时，这个对算法工程师和算法优化工程师来说挑战不小，差不多可以达到扣QPS这个级别（为了提升模型的几个QPS努力几个月），不像服务端动不动就100多Q。嵌入式端简单转一个模型可能只有10来q，想要达到24q也就是实时的级别可能需要花不小的功夫。<br />举几个大家经常玩的嵌入式平台：

- 亲民的树莓派，社区资源很丰富，价格个人玩家买得起
- NVIDIA的jetson系列，nano、NX、Xavier，资源也很丰富，算力高，可玩性好
- 高通的嵌入式平台，算力其实还可以，但是资源很少，个人玩家很难玩
- INTEL的计算棒，好久没玩了
- 国产的地平线旭日X3派，群里都在说这个，支持也不错
- 还有一些我不认识的，待补充

特点：

- 硬件资源紧张，很多时候需要把资源切分给每个模型使用，比如有10个加速单元，分2个单元给一个模型去跑实时等等，资源划分会更细
- 算力特别弱的，模型的可优化型可选择性也不高，看看这点算力，直接使用厂商提供的框架训练数据就行
- 未完待续

参考：

- [https://www.zhihu.com/question/554488510/answer/2891840467](https://www.zhihu.com/question/554488510/answer/2891840467)
### 多模型混合部署
有时候会有**很多模型**部署到**同一张显卡**上、或者部署到同一个AI加速卡上的情况，而显卡或者说加速卡，资源都是有限的（多个卡同理）。
> 比如英伟达的显卡，有限的显存和计算资源（SM、Tensor core、DLA）；比如高通的AI处理器，有限的计算资源（HVX、HMX、NPU）；

这个在服务器级别和嵌入式级别都会有需求。一般来说，提供加速卡的厂商也会提供相应的工具供你搜索多模型部署的最优配置，这种在实际场景中经常会用到，因为不大可能就一个模型在跑。比如自动驾驶场景中，多路摄像头每路摄像头接一个模型去跑，比如检测红绿灯的、检测行人的blabla，而这些模型一般都会在一个加速卡上跑，<br />如何合理分配这些资源，或者说如何使多个模型比较好的运行在这些资源上也是需要考虑的。简单预估可以，比如一个模型单张卡压测200qps，另一个模型100qps，可以简单预估两个模型一起qps各缩减一半。不过其实**最好的是有工具**可以实际压测，通过排列组合各种配置去实际压测这些模型的各种配置，选择一个最大化两个模型QPS或者latency的方法。当然也可以通过硬件划分资源来执行每个模型能获取到的实际算力，再细致层面你可以通过控制每个op->kernel执行发到内核通过内核进行控制每个kernel的执行优先级来控制模型。<br />先回来最简单的在一张卡上或者有限资源内如何搜寻最优配置，最好还是有工具，没有工具只能自己造了。<br />闭源的在你使用硬件厂商提供的工具中就知道了，比如你使用高通的板子，高通会提供你很多软件库，其中有一个可以分析多个模型或者一个模型在特定资源下的最高qps以及最低latency供你参考。同理其他硬件厂商也是一样的。<br />开源的可以看看triton的model_analyzer：

- [https://github.com/triton-inference-server/model_analyzer](https://github.com/triton-inference-server/model_analyzer/tree/add-ensemble-feature)
### 模型加密
先说模型加密有没有这个需求，如果没有这个需求就没必要哈，模型加密主要看**竞对的能力以及有多大意愿破解你的模型**hh。<br />我知道的模型加密的几个方式：

- 模型文件直接加密，把模型的二进制文件直接加密，一般都会有加密算法，对模型的前xx字节进行加密，解密的时候对前xx字节进行解密即可，这种加密只在load时候做，会影响模型加载速度，但是不影响运行速度
- 模型结构不加密，模型权重加密，有的可以采用类似于模型压缩的方法把权重压缩了，推理时以自定义协议读取加载推理
- 也需要对解密模型的代码进行加密，代码混淆要做，不过这样会影响解密代码也就是load模型过程，会变慢，就看你代码混淆做到什么程度了
- 未完待续

很多推理框架也提供了方便模型加密的接口：

- [https://github.com/triton-inference-server/checksum_repository_agent](https://github.com/triton-inference-server/checksum_repository_agent)

PS：加密模型无非就是想让破解方**花点功夫**去破解你的模型，花多少功夫取决你花多少功夫去加密你的模型；照这样说修改模型结构也可以达到加密模型的目的，比如模型融合，TensorRT天然会实现对模型的op融合、权重合并，很多op（conv+bn+relu）转成TensorRT之后结构已经面目全非了，权重也融合的融合、去除的去除，想要把TensorRT模型权重提取出来也是挺难得
## 模型转换那些事儿
**模型转换**是很常见的事情，因为训练平台和部署平台大概率是不一样的，比如你用Pytorch训练，但是用TensorRT去部署。两者模型结构实现方式不一样，肯定会涉及到模型移植的问题，模型转换，又或是模型移植，都是苦力活。又或者你不训练模型，拿到的**是别人训练**的模型，如果想在自己的平台部署，你的平台和人家的模型不在一个平台，那就得模型转换了，因为你拿到的模型可能是各种格式的：

- Caffe
- ONNX
- Pytorch
- TFLITE
- NCNN、MNN
- 不列了太多了

还是可以参考Netron，看下有多少模型列表，之前有提到过：<br />![Netron支持的模型类型列表](https://image.oldpan.me/Netron支持的模型类型列表.jpg)
> 不过如果不换平台的话，可以不考虑模型转换。

模型转换是个经验活，见得模型多了，就知道每个模型结构的特点以及转换特点。转换起来也就得心应手，少踩很多坑了。<br />![ChatGPT的回答](https://image.oldpan.me/ChatGPT的回答.jpg)
### 常干的事儿
首先是看模型结构，在上文中“自己造模型”这一节中已有提及。不论是什么样的模型结构，模型肯定都是按照一定的规则组织的，模型结构和模型权重都在里头，怎么看各凭本事了。<br />总结一些和模型相关的例子，有更新会放这里：

- [AI部署系列：你知道模型权重的小秘密吗？？？](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247487770&idx=1&sn=749d1719844dcfe5f87dae2de65a8b6b&chksm=cf10891ff86700096be8eeb671004693adca26051c3e75628dedb87a95ed857aadc0502c10b4&token=1097456929&lang=zh_CN#rd)
- 转换带有单独cuda实现的模型
- 如何导出torchscript模型
- 借助libtorch在C++和python中进行debug
### 排查各种问题
转模型和开盲盒一样，不能保证模型转完精度一定没问题，要做很多检查。<br />![ChatGPT-转换难点](https://image.oldpan.me/ChatGPT-转换难点.jpg)
#### 模型转换后一般要做的事儿

- 转换后看一下模型的输入输出类型、维度、名称、数量啥的是否和之前一致
- 转换后首先跑一张训练时的图（或者一批图），看下新模型的输出和旧模型差多少，一定要保证最终输入到模型的tensor一致（也可以使用random输入或者ones输入测试，不过对于模型权重分布特殊的模型来说，对于这种输入可能评测不是很准确）
- 批量跑测试集测一下精度是否一致
- benchmark转换后模型的速度是否符合预期
#### 常见的问题

- 转换后模型精度问题，输出为nan、精度完全错乱
- 转换后模型batch=1正常，多batch结果错乱
- 转换后输入/输出类型维度啥的错误
- 转换后模型速度未达到预期
- 未完待续
## 模型优化那些事儿
模型优化也是老生常谈的事情，小到优化一个op，大概优化整个推理pipeline，可以干的事情很多。<br />之后会总结一些常用的优化技巧在这里，先埋坑。
### 合并、替换op算子
很多框架在导出的时候就会**自动合并一些操作**，比如torch.onnx在导出**conv+bn**的时候会将bn吸到前面的conv中，比较常见了。<br />但也有一些**可以合并的操作，**假如框架代码还没有实现该pattern则不会合并，不过我们也可以自己合并，这里需要经验了，需要我们熟知各种op的合并方式。<br />可以自己合并一些操作，比如下图中的**convTranspose+Add+BN**，是不常见的Squence的pattern，如果自己愿意的话，可以直接在ONNX中进行合并，把权重关系搞对就行。
### ![conv+add+bn可以合并](https://image.oldpan.me/conv+add+bn可以合并.jpg)
列举一些合并的例子，总之就是将多个op合成大op，节省计算量以及数据搬运的时间：

- conv/transposeConv+bn
- 多个支路的conv合并为group conv
- gemm + bias -> conv 
- 未完待续

![两个ocnv可以合并](https://image.oldpan.me/两个ocnv可以合并.jpg)多路合并![由多路conv合并为一个conv](https://image.oldpan.me/由多路conv合并为一个conv.jpg)<br />既可以融合一些算子，当然也可以替换一些算子：

- relu6替换为max(0,6)
- 未完待续
### 蒸馏、剪枝
剪枝和蒸馏都是模型层面的优化，如果是稀疏化剪枝则会涉及到硬件层面。<br />剪枝的目的就是将在精度不变的前提下减少模型的一些层或者通道或者权重数量等等，甚至可以剪枝filter。可以节省计算量并且减少模型的体积，对于大模型来说是很有用的，一般来说剪枝后的模型比未剪枝的同等size大小精度更高。<br />具体的可以参考这篇：t[o prune or not to prune exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)。<br />剪枝的方法可以看zomi总结的ppt：<br />![来自chenzomi12-DeepLearningSystem](https://image.oldpan.me/来自chenzomi12-DeepLearningSystem.jpg)<br />蒸馏可以使同结构的小模型精度提升接近大模型的精度。
#### 总结
剪枝类似于模型搜索，如果直接NAS的话，就没有必要剪枝了。
#### 参考

- [to prune or not to prune exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)
- DAMO-YOLO
### 量化
量化目前比较成熟，也有很好的教程（PPQ），很成熟的库（PPQ），新手入门建议直接看[PPQ](https://github.com/openppl-public/ppq)，跟着教程来大概就知道量化的很多细节了。<br />要学习的知识点有：

- 量化基本概念，与硬件的关系
- PTQ量化和QAT量化，两种量化还有很多方法。PTQ有  EasyQuant（EQ）、；QAT有[LSQ](https://arxiv.org/abs/1902.08153)(Learned Step Size Quantization)、[DSQ](https://arxiv.org/abs/1908.05033)(Differentiable Soft Quantization)
- 可以实施量化的框架，怎么使用

但是需要注意，不是所有模型、所有op都适合量化：

- 重参数量化 [https://tech.meituan.com/2022/09/22/yolov6-quantization-in-meituan.html](https://tech.meituan.com/2022/09/22/yolov6-quantization-in-meituan.html)
- 量化模型直接输入int8 进行测试

注意点：

- 有些模型量化后，虽然整体指标没有变化（某个评价标准，比如coco的mAP），但是实际使用中，发现之前的一些效果变差了，这种情况大多是调用模型的策略效果与这个模型的耦合度比较高了。举个例子，比如之前这个模型对小目标检测效果好，但是量化后，小目标检测效果差了（然而中目标效果好了）所以导致与小目标耦合度比较高的策略兼容度不高，导致算法的整体精度下降。这种情况就比较尴尬，你可以调整策略，或者重新量化模型，加上一些约束使其在某些场景下尽可能和原始模型表现一致，但这个需要时间去优化了，有较高的时间成本。
- 待补充
#### 参考

- [必看部署系列~懂你的神经网络量化教程：第一讲！](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247488318&idx=1&sn=048c1b78f3b2cb25c05abb115f20d6c6&chksm=cf108b3bf867022d1b214928102d65ed691c81955b59ca02bccdee92584ad9aa8e390e1d2978&token=1097456929&lang=zh_CN#rd)
- [量化番外篇——TensorRT-8的量化细节](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247488838&idx=1&sn=56107c468d5b683a574e6046af3a541f&chksm=cf108d43f8670455736a83546eb5ed81abc9194d7d4c2359af393f26e3bddd1379f777e35f35&token=1097456929&lang=zh_CN#rd)
- [实践torch.fx第二篇——基于FX的PTQ量化实操](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247489317&idx=1&sn=797e32276bd4f55948d992f455415943&chksm=cf108f20f8670636b27be2431d5a4fdef1689eaa672e59ffc7d6181d2afb13a9a050d9b6b4a5&token=1097456929&lang=zh_CN#rd)
- [https://github.com/openppl-public/ppq](https://github.com/openppl-public/ppq)
### 算子op优化
算子优化就是无底洞，没有尽头，要学习的东西也很多，之前我也**简单深入**地探索过这个领域，只不过也只是**浅尝辄止，**优化完就立马干其他活了。这里也不班门弄斧了，涉及高性能计算的一些入门问题，可以看下这个的回答：

- [想进大厂的高性能计算岗位需要做哪些准备？](https://www.zhihu.com/question/525995150/answer/2465200528)

大部分优化op的场景，很多原因是**原生op的实现可能不是最优**的，有很大的优化空间。<br />比如LayerNorm这个操作，Pytorch原生的实现比较慢，于是就有了优化空间：

- [CUDA优化之LayerNorm性能优化实践](https://zhuanlan.zhihu.com/p/443026261)

同理，很多CUDA实现的OP可能不是最优的，只有你有精力，就可以进行优化。也要考虑是这个优化值不值，在整个模型中的占比大不大，投入产出比怎么样blabla。<br />对于CUDA还好些，资料很多，很多op网上都有开源的不错的实现（尤其是gemm），抄抄改改就可以了。<br />不过对于一些没有CUDA那么火的平台或者语言，比如arm平台的neon或者npu，这些开源的算子实现少一些，大部分需要自己手写。分析模型哪些op慢之后，手动优化一下。<br />知乎上也有很多优化系列的教程，跟着一步一步来吧：

- [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)
- [深入浅出GPU优化系列：elementwise优化及CUDA工具链介绍](https://zhuanlan.zhihu.com/p/488601925)
- [[施工中] CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275)
- [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)

有几种可以自动生成op的框架：

- TVM
- triton（此triton非彼triton）
### 调优可视化工具
可视化工具很重要，无论是可视化模型还是可视化模型的推理时间线。可以省我们的很多时间，提升效率：

- 画模型图的工具，graphvis
- NVIDIA的nsight system和nsight compute
- pytorch的profiler
## 还要了解很多推理框架/AI编译器
本来这节的小标题是“还要会很多推理框架/AI编译器”，但感觉很不现实，人的精力有限，不可能都精通，选择自己喜欢的、对实际平台有相应提升的推理框架即可。<br />这里总结了一些AI编译器：[https://github.com/merrymercy/awesome-tensor-compilers](https://github.com/merrymercy/awesome-tensor-compilers)
### ONNX
ONNX不多说了，**搞部署避免不了的中间格式**，重要性可想而知。
> [Open Neural Network Exchange (ONNX)](https://onnx.ai/) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Currently we focus on the capabilities needed for inferencing (scoring).

需要了解ONNX模型机构组成，和protobuf的一些细节；能够根据要求修改ONNX模型；能够转换ONNX模型；同时也需要知道、了解ONNX的一些坑。
#### 修改ONNX模型
修改模型，如果直接使用官方的helper接口会比较麻烦，建议使用`onnx-graphsurgeon`或者尝试尝试[onnx-modifier](https://github.com/ZhangGe6/onnx-modifier)。
#### 调试ONNX模型
埋坑。可以看知乎中openmmlab相关文章。
#### ONNX学习相关建议
多找些ONNX的模型看看，转换转换，看看相关github的issue<br />也可以加入大老师的群，群里很活跃很多大佬，相关ONNX问题大老师会亲自回答
#### 相关文章：

- [https://github.com/daquexian/onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
- [https://github.com/ZhangGe6/onnx-modifier](https://github.com/ZhangGe6/onnx-modifier)
- [https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)
- [https://github.com/onnx/onnx](https://github.com/onnx/onnx)
- [模型部署入门教程（五）：ONNX 模型的修改与调试](https://zhuanlan.zhihu.com/p/516920606)
- [https://github.com/onnx/onnx/blob/main/docs/Operators.md](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
### TVM
占坑。

- 如何评测模型的速度
- 如何benchmark模型各层的错误
- 解析trtexec中的benchmark
- 解析TVM中的graph和VM
### TensorRT
Tensorrt不必多说，先占坑。

- python端口调用
- C++端口调用
- 多线程调用
- 各种模型转换TensorRT（ONNX、Pytorch、TensorFLow）
- 各种和TensorRT相关的转换库
#### 自定义插件
如果模型中保存TensorRT不支持的算子，就需要自己实现cuda操作并且集成到TensorRT中。<br />现在也有很多可以生成插件的工具：

- [https://github.com/NVIDIA-AI-IOT/tensorrt_plugin_generator](https://github.com/NVIDIA-AI-IOT/tensorrt_plugin_generator) 

相关资料：

- [https://github.com/NVIDIA/trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)  
### libtorch
简单好用的对Pytorch模型友好的C++推理库。

- torch.jit.trace
- torch.jit.script
- python导出libtorch模型，C++加载libtorch模型
### AITemplate
AITemplate借助Stable Diffusion火了一下，其加速Stable Diffusion的效果比TensorRT要好不少。<br />我自己测试了一个res50的模型，利用`TensorRT-8.5.1.7`和`AITemplate-0.1dev`转化后简单测试了下速度，精度都是FP16，显卡是A4000。

| <br /> | 4x3x224x224 | 8x3x224x224 |
| --- | --- | --- |
| TensorRT | 2.07885 ms | 3.49877 ms |
| AITemplate | 1.36401 ms | 2.38946 ms |

看来很有潜力。
### 还有很多，待续
慢慢补充吧。
## 高性能计算
高性能计算和之前提到的op优化是强相关的，之前提到的这里就不多提了。<br />根据自己的场景和爱好熟悉一门**并行语言**是比较好的，而对于我来说，CUDA用的比较多，所以就学习CUDA了。如果你不知道该学什么，CUDA也很适合作为入门的并行计算语言，资料很多，使用场景丰富，相关讨论也很多。<br />如果你工作中使用的平台不是NVIDIA，则需要学习对应平台的相关技术栈。<br />未完待续。
## 自己的库
自己部署过程这么多，也积累了很多代码脚本，找时间整理下，供大家参考学习。
## 后记
这篇文章会一直更新总结，因为公众号的限制只能一篇一篇发。所以这个清单会放到博客、GITHUB、或者其他有精力之后更新的平台（知乎啥的）继续写，自己学习总结用，也可以供大家参考学习。关于链接也会在之后发出来，有兴趣的同学可以看看，目标是总结一个学习“部署”的路线吧，平民版的！<br />趁着新的一年到了（貌似是2023年第一篇），也给自己立个小目标吧，2023年文章输出30篇以上，尝试尝试换种方式搞一些相关的东西（视频、软件），最近好玩的太多了。<br />后续的文章也不会一写一大坨了，写的短一些，分篇章感觉会好一些，当然质量还会保证。
