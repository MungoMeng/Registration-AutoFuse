# AutoFuse: Automatic Fusion Networks for Unsupervised and Semi-supervised Medical Image Registration
Recently, Deep Neural Networks (DNNs) have been widely recognized for their ability to perform fast end-to-end registration. For end-to-end registration, DNNs need to explore the spatial information of each image and fuse this information to characterize spatial correspondence. This fusion raises an essential question: what is the optimal fusion strategy to characterize spatial correspondence? Existing fusion strategies (e.g., early fusion, late fusion) were empirically designed to fuse information with manually defined prior knowledge, which inevitably adds human bias and can limit the registration performance with potentially-flawed human designs.  In this study, we depart from existing empirically-designed fusion strategies and develop a data-driven fusion strategy for deformable image registration. To achieve this, we propose an Automatic Fusion network (AutoFuse) that provides flexibility to fuse information at many potential network locations. A Fusion Gate (FG) module is also proposed to control how to fuse information at each potential location based on training data. Our AutoFuse can automatically optimize its fusion strategy during training and can be generalizable to both unsupervised registration (without any labels) and semi-supervised registration (with weak labels provided for partial training data).  
**For more details, please refer to our paper. [[arXiv]()]**

## Overview
![architecture](https://github.com/MungoMeng/Registration-AutoFuse/blob/master/Figure/Overview.png)

## Publication
If this repository helps your work, please kindly cite our paper:
* **Mingyuan Meng, Michael Fulham, Dagan Feng, Lei Bi, Jinman Kim, "AutoFuse: Automatic Fusion Networks for Unsupervised and Semi-supervised Deformable Medical Image Registration," Under review. [[arXiv]()]**
