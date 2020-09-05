# <p align="right">关于本书</p>

***
本书的目的是利用PyTorch提供深度学习的基础，并在实际项目中展现它们。我们致力于解释深度学习的关键概念，并演示如何通过PyTorch将它们教给实践者。在这本书中，我们尝试提供能进一步探索的直觉概念，同时，我们会有选择地深剖细节，展示这些场景背后的道理。

《深度学习与PyTorch》并不是一本参考手册，相反，它是一本概念性的读物，可以让您独立地在线探索更高级的材料。因此，我们将重点关注 PyTorch 提供的子集功能。最值得注意的是递归神经网络，但是PyTorch API的其他部分也是如此。

## 本书的读者是谁？

本书面向正在或打算成为深度学习实践者并希望了解PyTorch的开发人员。我们假设我们的典型读者是计算机科学家、数据科学家，软件工程师，或者是相关课程的本科生或更高学历的学生。由于我们假定读者不具备深度学习的先验知识，因此本书前半部分的某些内容可能是有经验的从业者已经知道的重复概念。对于这类读者，我们希望本书能提供一个与您已知主题稍有不同的视角。

我们希望读者具备命令式和面向对象编程的基本知识。由于本书使用Python，您应该熟悉其语法和操作环境。了解如何在您选择的平台上安装Python包和运行脚本是先决条件。来自C++、Java、JavaScript、Ruby或其他类似语言的读者应该很容易上手，但在本书之外还需要做一些补充。同样，虽然不是硬性要求，熟悉NumPy也会很有帮助。我们还希望您能熟悉一些基本的线性代数，例如知道什么是矩阵和向量，什么是点积。

## 本书的组织结构：路线图

《深度学习与PyTorch》分为三个不同的部分。第1部分涵盖了基础知识，而第2部分在第1部分介绍的基本概念的基础上，加入更多高级概念，引导您完成一个端到端项目。简短的第3部分则以了解PyTorch提供的部署功能来结束本书。您可能会注意到各部分之间存在不同的语言和图片风格。尽管本书是无休止地协作规划、讨论和编辑的结果，但写作和绘制图片的工作却是由各位作者分担完成的。Luca主要负责第1部分，Eli负责第2部分<sup>2</sup>，Thomas则试图将第3部分的风格与前两部分相融合。与其追求最低限度的统一性，我们决定保留各部分特有的原貌。
___
<sup>2</sup> Eli和Thomas的一些艺术作品出现在了其他部分；如果您发现在某个章节中，风格改变了，不要感到震惊！
___

&nbsp;&nbsp;&nbsp;&nbsp;以下是各部分的分解，并简要介绍了各部分。

**PART 1**

在第1部分中，我们将迈出使用PyTorch的第一步，培养理解开发PyTorch项目所需的基本技能，并开始构建我们自己的项目。我们将介绍PyTorch API和一些令PyTorch成为库的幕后特性，并着手训练一个初始的分类模型。在第1部分结束时，我们将为处理一个真实的项目而做好准备。

第1章介绍了PyTorch作为一个库及其在深度学习革命中的地位，并触及了PyTorch有别于其他深度学习框架的地方。
第2章通过运行预训练网络的示例来展示PyTorch的运行情况，它演示了如何在PyTorch Hub中下载和运行模型。
第3章介绍了PyTorch的基本构件--张量，展示了它的API，并在介绍了一些幕后的实现细节。
第4章演示了如何将不同类型的数据表示为张量，以及深度学习模型对张量形状的预期。
第5章介绍了通过梯度下降进行学习的机制，以及PyTorch如何使其实现自动微分功能。
第6章展示了在PyTorch中使用`nn`和`optim`模块构建和训练回归神经网络的过程。
第7章在上一章的基础上，创建一个用于图像分类的全连接模型，并扩展PyTorch API的知识。
第8章介绍了卷积神经网络，并扩展了用于构建神经网络模型及其PyTorch实现的更多高级概念。

**PART 2**

在第2部分中，每一章都使我们更接近自动检测肺癌的综合解决方案。我们将以这一难题为动力，展示解决癌症筛查等大规模问题所需的实际方法。这是一个专注于清洁工程、故障排除和问题解决的大型项目。

第9章描述了我们将用于肺部肿瘤分类的端到端策略，从计算机断层扫描(CT)成像开始。
第10章使用标准PyTorch API加载人体注释数据以及来自CT扫描的图像，并将相关信息转换为张量。
第11章介绍了一个使用第10章中介绍的训练数据的一级分类模型，我们对该模型进行了训练，并收集了基本的性能指标。
我们还介绍了如何使用TensorBoard来监控训练。
第12章探索并实现了标准的性能指标，并使用这些指标来判别训练模型的缺陷。然后，我们通过使用数据平衡和数据增强的改进训练集来缓解这些缺陷。
第13章描述了分割，这是一种像素到像素的模型架构，我们使用它来生成覆盖整个CT扫描图像的可能肿瘤位置的热图。
这张热图可以用于在未经人类判别的CT扫描图片上寻找肿瘤位置。
第14章实现了最终的端到端项目：使用我们的新型分割模型对癌症患者进行诊断，然后进行分类。

**PART 3**

第3部分是有关部署的单章。 第15章概述了如何将PyTorch模型部署到简单的Web服务里，如何将其嵌入C++程序或将其移植到手机中。

## 关于代码

本书中的所有代码都是为Python 3.6或更高版本编写的。该书的代码可从Manning的网站(https://www.manning.com/books/deep-learning-with-pytorch) 和GitHub(https://github.com/deep-learning-withpytorch/dlwpt-code) 下载。版本3.6.8是撰写本文时的最新版本，也是我们用来测试本书中示例的版本。 例如：

```python
$ python
Python 3.6.8 (default, Jan 14 2019, 11:02:34)
[GCC 8.0.1 20180414 on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

在Bash提示符下输入的命令行以` $ `开头（例如，本示例中的`$ python`行）。固定宽度的内联代码看起来像`self`。

以`>>>`开头的代码块是Python交互式提示符下会话的记录。`>>>`字符不应视为输入；本书中不以`>>>`或者`...`开头的文本行都是输出。在某些情况下，会在`>>>`前插入额外的空行，以提高打印时的可读性。当您在交互提示符下实际输入文本时，不包括这些空行：

```bash
>>> print("Hello, world!") 
Hello, world!
                                <-----------这一空行在实际运行时不会出现
>>> print("Until next time...") 
Until next time...
```

我们还大量使用了Jupyter Notebook，如第1章第1.5.1节所述。 作为官方GitHub存储库的一部分，我们提供的notebook中的代码如下所示：

```python
# In[1]: 
print("Hello, world!") 

# Out[1]: 
Hello, world!

# In[2]: 
print("Until next time...") 

# Out[2]: 
Until next time...
```

几乎我们所有的示例笔记本都在第一个单元格中包含以下样板（在前几章中可能会遗漏一些行），之后，我们会将其略过：

```python
# In[1]: 
%matplotlib inline 
from matplotlib import pyplot as plt 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

torch.set_printoptions(edgeitems=2) 
torch.manual_seed(123)
```

此外，代码块是`.py`源文件的部分或全部。

**Listing 15.1 main.py:5, def man**
```python
def main(): 
    print("Hello, world!") 
    
if __name__ == '__main__': 
    main()
```

书中的许多代码示例都使用两个空格缩进。由于打印的限制，代码清单被限制为80个字符的行，这对于大量缩进的代码段来说可能是不切实际的。使用两个空格缩进有助于减轻原本会出现的过度换行。这本书的所有可供下载的代码(同样，在https://www.manning.com/books/deep-learningwith-pytorch 和 https://github.com/deep-learning-with-pytorch/dlwpt-code) 上使用一致的四个空格缩进)。以a_t后缀命名的变量是存储在CPU内存中的张量，_g是GPU内存中的张量，_a是NumPy数组。

本书中的许多代码样本都是以两格缩进的方式呈现的。由于印刷的局限性，代码列表每行被限制在80个字符内，这对于大量缩进的代码段来说可能不切实际。使用两个空格缩进有助于减轻原本会出现的过度换行问题。所有可供本书下载的代码(同样，在 https://www.manning.com/books/deep-learningwith-pytorch 和 https://github.com/deep-learning-with-pytorch/dlwpt-code)都使用了一致的四个空格缩进。以`_t`为后缀命名的变量是存储在CPU内存中的张量，`_g`是存储在GPU内存中的张量，`_a`是`NumPy`数组。

## 硬件和软件需求

第1部分被设计为不需要任何特定的计算资源。任何最新的计算机或在线计算资源都是足够的。同样，也不需要特定的操作系统。在第2部分中，我们预计完成更高级示例的完整培训运行将需要支持一个支持CUDA的GPU。第2部分中使用的设备默认参数均假设GPU具有8 GB的RAM(我们建议使用NVIDIA GTX 1070或更高版本)，但是如果您的硬件可用的RAM较少，可以调整这些参数。第2部分的癌症检测项目所需的原始数据下载量约为60 GB，系统上总共需要200 GB(至少)可用磁盘空间用于训练模型。幸运的是，在线计算服务最近开始免费提供GPU时间。我们将在相应的章节中更详细地讨论计算需求。

您需要Python 3.6或更高版本的版本；可在Python网站(https://www.python.org/downloads) 上找到相关说明。有关 PyTorch的安装信息，请参阅 PyTorch 官方网站 (https://pytorch.org/get-started/locally) 上的入门指南。我们建议Windows用户使用Anaconda或Miniconda (https://www.anaconda.com/distribution 或 https://docs.conda.io/en/latest/miniconda.html) 进行安装。其他操作系统如Linux通常有更多的可行选项，`Pip`是Python最常用的包管理器。我们提供了一个requirements.txt文件，`Pip`可以用它来处理Python的依赖需求。由于目前的苹果笔记本电脑不包含支持CUDA的GPU，因此PyTorch的macOS预编译包只支持CPU。当然，有经验的用户可以自由地以最符合您首选开发环境的方式安装软件包。

## liveBook论坛

购买《深度学习与PyTorch》包括免费访问由Manning Publications运营的私人网络论坛，您可以在该论坛上对本书发表评论，提出技术问题，并从作者和其他用户那里获得帮助。要访问该论坛，请访问 https://livebook.manning.com/#！/book/deep-learning-with-pytorch/discussion 。您可以在 https://livebook.manning.com/#！/discussion 了解更多关于Manning论坛和行为规则的信息。Manning对读者承诺提供一个场所，让个人读者之间和读者与作者之间可以进行有意义的对话。这不是对作者的任何具体参与量的承诺，他们对论坛的贡献仍然是自愿的（且无偿的）。我们建议您尝试向他们提出一些具有挑战性的问题，以免他们失去兴趣。只要该书还在印刷，就可以从出版商的网站上访问论坛并获得以前讨论的存档。

## 其他在线资源

虽然这本书没有假设读者需要具备深度学习的先验知识，但它没有包括深度学习的基础介绍。我们涵盖了基础知识，但我们的重点是熟练使用PyTorch库。我们鼓励感兴趣的读者在阅读本书之前、期间或之后建立起对深度学习的直观理解。为此，*Grokking Deep Learning* (www.manning.com/books/grokking-deep-learning) 是一本很好的资源，它可以帮助我们对深度神经网络的基础机制建立起强大的心理模型和直觉理解。要想获得全面的介绍和参考，我们建议您阅读Goodfellow et al.的*Deep Learning* (www.deeplearningbook.org)。当然，Manning Publications也具有大量的深度学习书目(www.manning.com/CATALOG#SECTION-83)，涵盖了该领域的各种主题。根据您的兴趣，它们中的许多都值得成为您阅读的下一本书。