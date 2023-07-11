# Robust Outlier Rejection for 3D Registration with Variational Bayes (CVPR2023)

PyTorch implementation of the paper ["Robust Outlier Rejection for 3D Registration with Variational Bayes"](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_Robust_Outlier_Rejection_for_3D_Registration_With_Variational_Bayes_CVPR_2023_paper.pdf).

Haobo Jiang, Zheng Dang, Zhen Wei, Jin Xie, Jian Yang, and Mathieu Salzmann.


## Introduction

Learning-based outlier (mismatched correspondence) rejection for robust 3D registration generally formulates the outlier removal as an inlier/outlier classification problem. The core for this to be successful is to learn the discriminative inlier/outlier feature representations. In this paper, we develop a novel variational non-local network-based outlier rejection framework for robust alignment. By reformulating the non-local feature learning with variational Bayesian inference, the Bayesian-driven long-range dependencies can be modeled to aggregate discriminative geometric context information for inlier/outlier distinction. Specifically, to achieve such Bayesian-driven contextual de- pendencies, each query/key/value component in our non-local network predicts a prior feature distribution and a posterior one. Embedded with the inlier/outlier label, the posterior feature distribution is label-dependent and discriminative. Thus, pushing the prior to be close to the dis- criminative posterior in the training step enables the features sampled from this prior at test time to model high-quality long-range dependencies. Notably, to achieve effective posterior feature guidance, a specific probabilistic graphical model is designed over our non-local model, which lets us derive a variational low bound as our optimization objective for model training. Finally, we propose a voting-based inlier searching strategy to cluster the high-quality hypothetical inliers for transformation estimation. Extensive experiments on 3DMatch, 3DLoMatch, and KITTI datasets verify the effectiveness of our method.


## Requirements

If you are using conda, you may configure VBReg as:

    conda env create -f environment.yml
    conda activate VBReg

If you also want to use FCGF as the 3d local descriptor, please install [MinkowskiEngine v0.5.0](https://github.com/NVIDIA/MinkowskiEngine) and download the FCGF model (pretrained on 3DMatch) from [here](http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-3dmatch-v0.05.pth).

## Dataset Preprocessing

### 3DMatch

The raw point clouds of 3DMatch can be downloaded from [FCGF repo](http://node2.chrischoy.org/data/datasets/registration/threedmatch.tgz). The test set point clouds and the ground truth poses can be downloaded from [3DMatch Geometric Registration website](http://3dmatch.cs.princeton.edu/#geometric-registration-benchmark). 
Please make sure the data folder contains the following:

```
.                          
├── fragments                 
│   ├── 7-scene-redkitechen/       
│   ├── sun3d-home_at-home_at_scan1_2013_jan_1/      
│   └── ...                
├── gt_result                   
│   ├── 7-scene-redkitechen-evaluation/   
│   ├── sun3d-home_at-home_at_scan1_2013_jan_1-evaluation/
│   └── ...         
├── threedmatch            
│   ├── *.npz
│   └── *.txt                            
```

To reduce the training time, we follow PointDSC to pre-compute the 3D local descriptors (FCGF or FPFH) so that we can directly build the input correspondence using NN search during training. Please use `misc/cal_fcgf.py` or `misc/cal_fpfh.py` to extract FCGF or FPFH descriptors. 

### KITTI

The raw point clouds can be download from [KITTI Odometry website.](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) Please follow the similar steps as 3DMatch dataset for pre-processing.

## Pretrained Model

We provide the pre-trained model of 3DMatch in `snapshot/VBReg_3DMatch_release` and KITTI in `snapshot/VBReg_KITTI_release`. 
Both of them take correspondence coordinates rather than correspondence feature+corrdinate as input.  


## Instructions to training and testing

### 3DMatch / KITTI 

The training and testing on 3DMatch / KITTI dataset can be done by running
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_3DMatch.py

CUDA_VISIBLE_DEVICES=0 python3 train_KITTI.py

CUDA_VISIBLE_DEVICES=0 python3 evaluation/test_3DMatch.py

CUDA_VISIBLE_DEVICES=0 python3 evaluation/test_KITTI.py
```


### 3DLoMatch
We also evaluate our method on 3DLoMatch benchmark following [OverlapPredator](https://github.com/ShengyuH/OverlapPredator),
```bash
CUDA_VISIBLE_DEVICES=0 python3 evaluation/test_3DLoMatch.py
```
Note: You first need to follow the offical instruction of [OverlapPredator](https://github.com/ShengyuH/OverlapPredator) to extract the features.

## Citation

If you find this project useful, please cite:

```bash
@InProceedings{Jiang_2023_CVPR,
    author    = {Jiang, Haobo and Dang, Zheng and Wei, Zhen and Xie, Jin and Yang, Jian and Salzmann, Mathieu},
    title     = {Robust Outlier Rejection for 3D Registration With Variational Bayes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {1148-1157}
}
```

## Acknowledgments
We thank the authors of 
- [PointDSC](https://github.com/XuyangBai/PointDSC)
- [FCGF](https://github.com/chrischoy/FCGF)
- [DGR](https://github.com/chrischoy/DeepGlobalRegistration)
- [OverlapPredator](https://github.com/ShengyuH/OverlapPredator)

for open sourcing their methods.
