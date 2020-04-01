## 特征逆推图像

### 研究背景

#### ``ONNX``

很多优秀的视觉模型都是用caffe写的， 很多新的研究论文使用Pytorch写得，而更多的模型用TF写成，因此如果我们要测试它们就需要对应的框架环境，ONNX交换格式时我们能够在同一环境下进行不同模型的测试。

#### VGGNET结构py实现

13个卷积层 + 3个全连接层

##### 卷积层

- conv2d+bias
- batchnorm正则化
- relu激活

```
def conv_layer(self, bottom, name, stride = 1):
    with tf.variable_scope(name):
        filt = self.get_conv_filter(name)
        conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
        conv_biases = self.get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)
        mean = self.get_mean(name)
        variance = self.get_variance(name)
        offset = self.get_beta(name)
        scale = self.get_gamma(name)
        norm = tf.nn.batch_normalization(bias, mean, variance, offset, scale, 1e-20 )
        relu = tf.nn.relu(norm)
        return relu
```

#### HOG 方向梯度直方图

HOG+SVM是行人检测的主要方法

1. 主要思想：在一副图像中，局部目标的表象和形状能够被梯度或边缘方向密度分布很好描述（梯度的统计信息主要集中在边缘部分）
2. 具体实现方法：把图像分成很小的联通区域，称为细胞单元，然后采集细胞单元中各像素点的梯度或者边缘方向的直方图，然后把直方图组合起来构成特征描述器。
3. 优点：
   1. HOG在图像的局部单元上进行操作，所以对图像的几何和光学形变都能保持很好的不变性，这两种形变只会出现在更大的空间领域上。
   2. 在粗的空域抽象、精细方向抽样以及较强的局部光学归一化等条件下，只要行人大题能够保持直立姿势，可以忽略一些细微动作带来的影响。

#### SIFT 尺度不变特征转化

用来侦测与描述影像中的局部特征，它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量。

应用范围：物体辨识、机器人地图感知与导航、影像追踪、手势辨识等。

局部影像特征帮助辨识物体：

1. SIFT特征是基于物体上的一些局部外观特征兴趣点而与影像大小和旋转无关，对于光线、噪声、些微视角改变的容忍度页相当高。基于这个特性，高度显著且容易撷取，在庞大的特征数据库中，很容易辨识物体且鲜有误认。
2. 使用SIFT描述特征对部分物体遮蔽的侦测率页相当高，甚至只需要三个以上的SIFT物体特征就足以计算出位置与方位。辨识速度可以接近即时运算。



#### 作业的思路

VGGNET16已经准备好（参数已经训练好了）。

1. 使用原图构建一个VGG16计算图 --- bottom
2. 使用noise构建一个VGG16计算图
3. 指定计算图的某一层，比如conv3_1, 目的就是看在这一层神经网络学到了什么内容。
4. bottom在构建的时候只初始化一次，不会更新，作为noise学习的target；使用欧几里得距离来计算误差【损失函数】
5. 使用ADAM作为优化器，对损失函数进行优化。

#### 记录TF优化器内置方法

- ##### compute_gradients(loss, val_list) 

用来计算loss对val_list中每一项的偏导

- ##### apply_gradients(grads)

将compute_gradients返回的值作为输入参数对variable进行更新。

> 这两个函数等效于minimize() 方法，拆开用**梯度修正**

#### 正则化方法：TV全变分模型

用法和L1, L2正则化的方法类似，在目标函数最后加一个正则项。

![img](https://pic3.zhimg.com/80/v2-7d6f3ad6cee5bca7b2cc433ccb480fde_1440w.jpg)

```python
tf.image.total_variation
```

计算并返回一个或多个图像的总体变化，总变化量是输入图像中相邻像素绝对差值的综合，是衡量图像中有多少噪声的重要因素。

#### 残差网络学习

为什么要引入残差网络？

深度学习中网络层数的增多一般会伴随以下问题：

1. 计算资源消耗 -- 可以通过GPU集群来解决
2. 过拟合  -- 可以通过采集海量数据并且配合dropout解决
3. 梯度消失，梯度爆炸 -- 可以通过BN来解决

看似通过增加模型深度就可以从中获益，但是随着网络深度的增加模型出现了退化现象: 随着网络层数的增加，训练集loss逐渐下降然后区域饱和；若再增加深度，训练集的loss反而会增大。这并不是过拟合，因为过拟合会使训练误差不断减小。

当网络退化时，浅层网络能够达到的效果比深层网路更好，这时如果我们把低层特征传到高层，那么效果至少不会比浅层网络差，或者说一个vgg-100的网络再第98层使用的是和vgg-16在14层使用的特征相同，那么vgg100的效果会和vgg16相同。所以，可以在vgg100的98层和vgg16的14层之间加一条直接映射来达到此效果。

> 信息论：由于DPI（数据处理不等式）的存在，在前向传输的过程中，随着层数的增加，feature map包含的图像信息会逐层减少，而resnet的直接映射加入保证了l+1层的网路一定比l层包含更多的图像信息。

#### 残差网络

##### 残差块

![img](https://pic2.zhimg.com/80/v2-bd76d0f10f84d74f90505eababd3d4a1_720w.jpg)

残差网络是由一系列残差块构成，曲线右边是残差部分，一般由两到三个卷积操作构成。

在卷积网络中，``x_l``和``x_l+1``的feature map数量不同，需要用``1*1``的卷积网路升维或者降维。

##### 残差网络的原理

![[公式]](https://www.zhihu.com/equation?tex=y_l%3D+h%28x_l%29%2B%5Cmathcal%7BF%7D%28x_l%2C+%7BW_l%7D%29%5Ctag%7B3%7D)

![[公式]](https://www.zhihu.com/equation?tex=x_%7Bl%2B1%7D+%3D+f%28y_l%29%5Ctag%7B4%7D)

对于更深的层L, 其与l层的关系可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=x_L+%3D+x_l+%2B+%5Csum_%7Bi%3D1%7D%5E%7BL-1%7D%5Cmathcal%7BF%7D%28x_i%2C+%7BW_i%7D%29%5Ctag%7B6%7D)

反映了残差网络的两个属性：

1. L 层可以由任意一个比它浅的层表示
2. L 是各个残差块的单位累加和，而MLP是特征矩阵的累积。

根据BP求导法则，损失函数loss关于x_l的梯度可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+%5Cvarepsilon%7D%7B%5Cpartial+x_l%7D+%3D+%5Cfrac%7B%5Cpartial+%5Cvarepsilon%7D%7B%5Cpartial+x_L%7D%5Cfrac%7B%5Cpartial+x_L%7D%7B%5Cpartial+x_l%7D+%3D+%5Cfrac%7B%5Cpartial+%5Cvarepsilon%7D%7B%5Cpartial+x_L%7D%281%2B%5Cfrac%7B%5Cpartial+%7D%7B%5Cpartial+x_l%7D%5Csum_%7Bi%3D1%7D%5E%7BL-1%7D%5Cmathcal%7BF%7D%28x_i%2C+%7BW_i%7D%29%29+%3D+%5Cfrac%7B%5Cpartial+%5Cvarepsilon%7D%7B%5Cpartial+x_L%7D%2B%5Cfrac%7B%5Cpartial+%5Cvarepsilon%7D%7B%5Cpartial+x_L%7D+%5Cfrac%7B%5Cpartial+%7D%7B%5Cpartial+x_l%7D%5Csum_%7Bi%3D1%7D%5E%7BL-1%7D%5Cmathcal%7BF%7D%28x_i%2C+%7BW_i%7D%29+%5Ctag%7B7%7D)

1. 在整个训练中，第二项不可能一直为-1，残差网络不会出现梯度消失
2. 第一项表示L层的梯度可以传到任何一个比它浅的层。

> 直接映射是最好的选择

可以给(6)第一项加一个系数$$\lambda$$，求导后发现若$$\lambda>1$$则会梯度爆炸，若$$\lambda<1$$则会梯度消失

### 实验记录如下

#### 学习率对于训练效果的影响

![lr.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/lr.PNG?raw=true)

可以看到学习率从0.001变化到0.01，模型的收敛速度更快，并且最终达到的效果不同，学习率越大，图像最终越接近真实结果。

实验采样不同学习率下迭代1000次的图像效果：

![对比图.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/%E5%AF%B9%E6%AF%94%E5%9B%BE.PNG?raw=true)

#### Total-Variation 正则化的影响

- #### 加入TV 正则化

  > 源自[TF 2.0 官方手册](https://www.tensorflow.org/api_docs/python/tf/image/total_variation)

  ```python
  tv_regular = tf.reduce_sum(tf.image.total_variation(noise_layer))
  ```

  ![tv_02.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/tv_02.PNG?raw=true)

- #### 提高TV 正则化的比例

  ![tv_01.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/tv_01.PNG?raw=true)

- #### 提高``fea/rep``的比例

  

#### 深层网络和浅层网络的对比

> vgg16使用conv3_1得到的图像

![conv3_1.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/conv3_1.PNG?raw=true)

> vgg16使用conv1_1得到的图像

![conv1_1.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/conv1_1.PNG?raw=true)

> vgg16使用conv3_1+fc6得到的图像

![对比图](E:\Junior Year\大三下\深度学习\Assignment\MyWorkPlace\02-inversion\project\工程文档\img\fc6conv3.PNG)

#### 不同模型下的特征提取

resnet-18使用res2得到的图像

![res2.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/res2.PNG?raw=true)

resnet-18使用middle1得到的图像

![middle0.PNG](https://github.com/Gary11111/02-Inversion/blob/master/img/middle0.PNG?raw=true)

代码修改：模型加载和目标层的选择 ==> 详见``q4.diff``

#### Other：使用自己的图片得到的图像

使用resNet-18 的res2得到的结果如下。

![self.png](https://github.com/Gary11111/02-Inversion/blob/master/img/self.png?raw=true)

### 论文阅读情况

> 神经网络的可解释性方面

![paper.png](https://github.com/Gary11111/02-Inversion/blob/master/img/paper.png?raw=true)