# 实验进度

- 深度学习部分-采用resnet18+acrface loss

  - tongji训练集

    420个epoch，正确率99%

    ![image-20221223144539624](/Users/wuyihang/Library/CloudStorage/OneDrive-个人/my_idea/img/image-20221223144539624.png)

  - tongji测试集

    平均识别正确率98.4%

    ![image-20221223144710628](/Users/wuyihang/Library/CloudStorage/OneDrive-个人/my_idea/img/image-20221223144710628.png)

  - CASIA数据集

    - 首先提取ROI-谷点提取法，有的图像用谷点的算法算不出，用最大内切圆法提取（大概占10%～20%左右）

    在CASIA测试集上平均正确率78%

    ![image-20221223145116928](/Users/wuyihang/Library/CloudStorage/OneDrive-个人/my_idea/img/image-20221223145116928.png)

  - IITD数据集

    测试集平均正确率为93.9%

    ![image-20221224163140306](/Users/wuyihang/Library/CloudStorage/OneDrive-个人/my_idea/img/image-20221224163140306.png)
- 方向特征：竞争编码
  - IITD：90%
  - CASIA：69%
  - tongji：71%