# 区域形态分析
# area-shape-analysis
这是中科大研一课程数字图像分析的课程大作业。以下为题目要求。实现语言为matlab，具体内容见代码和report_dia.pdf。

## 题目
### 说明：
图像中含有大小不同的两类目标，请将各个目标分割出来，然后对分割的目标，利用合适的特征，将其进行分类。
segment the stones out and classify them with proper feature.


### 提示方法：
1、基于marker-controlled watershed进行分割,然后对分割的目标，选取特征进行Kmean聚类。

segment by marker-controlled watershed and clauster them by k-means

关于Marker-controlled watershed(基于标记的分水岭算法)可以参考：《数字图像处理（第二版）》(冈萨雷斯著) 见10.5.4节。
Gonzalez's book of chapter 10.5.4

