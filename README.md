# AlexNet

**网络模型如下：**

![AlexNet](https://github.com/Y-nk-FE/AlexNet/assets/107013750/8e1fb1c9-fc8a-45b9-b580-2b08c4007b17)




这是一个图像二分类问题，主要解决对肺部X-Ray的医学影像数据进行分类预测，判断该患者的肺部状态为normal还是pneumonia，该项目选择了AlexNet网络结构，该网络结构简单，可以满足电脑对深度学习的要求。

---

##### 效果

经过多次训练，发现AlexModel_V11.pth、AlexModel_V21.pth的测试准确度最高，test_acc = 0.9375


![test_resualt](https://github.com/Y-nk-FE/AlexNet/assets/107013750/09b3f26d-e97e-45e4-98a6-d57bf2341e5c)


---

##### 项目文件说明

（文件解压后放在AlexNet目录下，并删除压缩文件）

Chest_XRay：原始数据文件，其中包含训练集、验证集、测试集(from kaggle)(donwload url:https://www.kaggle.com/datasets/alifrahman/chestxraydataset)

data_enhance：已经对Chest_XRay中的训练集进行过随机数据增强的文件

**数据说明**

由于数据过于庞大，不方便上传至GitHub，目前将数据上传到了百度网盘

log：用于存放模型训练的日志

plt：用于存放acc和loss的折线图

pth_save：用于存放训练好的模型文件

Data_Enhance.py  数据增强文件，对Chest_XRay文件中的训练集进行数据增强并且保存在data_enchance文件夹下

net.py  AlexNet网络模型文件

train.py  网络训练文件

test.py 模型测试文件

---

##### 环境

Pytoch环境下运行

```
In [4]:torch.__version__
Out[4]: '2.1.1'
```

python3.10

```
(pytorch) PS D:\User\AI_Experiment\Classification\AlexNet_V3> python
Python 3.10.0 | packaged by conda-forge | (default, Nov 10 2021, 13:20:59) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
```

---

##### 备注

作者已经完善代码中的路径，在环境配置好的情况下应该可以直接运行

若有遗漏的错误请指出，有问题请直接联系作者

Email：y_years@126.com
