#基于深度迁移学习对JPEG图像进行隐写分析
---
keywords：deep learning,transfer learning,steganalysis
---
##内容介绍
DCTR_matlab：使用MATLAB实现的残差图像的离散余弦变换算法。

SRNet：隐写分析残差网络实现，CNN网络中加入深度学习短连接（shortcut connections）的残差层。

log：保存tensorboard可视化查看日志文件，可在网页localhost：6060中打开，观察网络和张量的变化以及进行loss，acc曲线图的绘制等。

savemodel：训练过程中保存的训练模型。

conv.py：CNN网络的卷积层等部分，主要用来进行隐写特征提取。

dataload.py：对JPEG图像进行处理生成网络能够读取的格式。

dctr.py：对JPEG图像进行DCTR（残差图像的离散余弦变换）进行DCT基核滤波提取图像在变换域（频域空间）的隐写特征，以及使用高通滤波器进行空间域的特征提取。

distance：计算源域和目标域数据的KV核距离（高斯核和线性核）。

others.py：用来生成对JPEG图像进行滤波处理后得到的残差图像。

train.py：读取JPEG图像训练数据进行网络训练得到深层卷积网络进行隐写分析的网络模型。

test.py：读取测试数据并使用训练好的网络进行隐写分析性能测试。

steganography：使用不同的隐写算法对原始JPEG图像数据集进行预处理得到cover_stego（原始_隐写）图像数据对。

##图像数据集
原始数据集使用[BOSSbase_v1.01](http://dde.binghamton.edu/download/)数据集转化成10000张512×512大小的JPEG图像。

使用不同的隐写算法和隐写嵌入率生成的tfrecord文件保存在[tfrecord for stego](https://pan.baidu.com/s/1Nsd1pQG2NR77pk9eB-vaUQ)（提取码steg）中。

同上，SRNet中使用的tfrecord文件保存在[tfrecord for srnet](https://pan.baidu.com/s/1arkMdP2zQ-SvCdOKIWHB_Q)（提取码steg）中。