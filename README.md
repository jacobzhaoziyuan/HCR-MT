<div align="center">
   
# Hierarchical Consistency Regularized Mean Teacher for Semi-supervised 3D Left Atrium Segmentation
   
[![EMBC2021](https://img.shields.io/badge/arXiv-2105.10369-blue)](https://arxiv.org/abs/2105.10369)
[![EMBC2021](https://img.shields.io/badge/Conference-EMBC2021-green)](https://ieeexplore.ieee.org/document/9629941)
   
</div>

Pytorch implementation of our method for EMBC2021 paper: "Hierarchical Consistency Regularized Mean Teacher for Semi-supervised 3D Left Atrium Segmentation". 

Contents
---
- [Abstract](#Abstract)
- [Dataset](#Dataset)
- [Installation](#Installation)
- [Training](#Training)
- [Testing](#Testing)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)


## Abstract
Deep learning has achieved promising segmentation performance on 3D left atrium MR images. However, annotations for segmentation tasks are expensive, costly and
difficult to obtain. In this paper, we introduce a novel hierarchical consistency regularized mean teacher framework for 3D left atrium segmentation. In each iteration, the student model is optimized by multi-scale deep supervision and hierarchical consistency regularization, concurrently. Extensive experiments have shown that our method achieves competitive performance as compared with full annotation, outperforming other stateof-the-art semi-supervised segmentation methods.



<p align="center">
<img src="/assets/workflow.PNG" width="700">
</p>

## Dataset

* Download the GE-MRI: dataset from [2018 Atrial Segmentation Chanllenge](http://atriaseg2018.cardiacatlas.org/)

    * We evaluate our proposed semi-supervised method on the dataset of 2018 Atrial Segmentation Challenge for left atrium segmentation in 3D gadolinium-enhanced MR image scans (GE-MRIs). The dataset consists of 100 scans with segmentation masks. We select 80 samples for training and 20 for testing and use the same data preprocessing methods for a fair comparison.
    * The training data can be downloaded [here](https://github.com/yulequan/UA-MT/tree/master/data).

## Installation

1. Clone the repository:
```
git clone https://github.com/ShumengLI/Hierarchical-Consistency-Regularized-MT.git 
cd Hierarchical-Consistency-Regularized-MT
```
2. Put the data in `data/2018LA_Seg_Training Set`.

## Training

3. Train the model
```
cd code
python train_mmt.py --gpu 0 --exp model_name
```
Params are the best setting in our experiment.

## Testing
4. Test the model
```
python test_hierarchy.py --gpu 0 --model model_name
```
Our best model is uploaded.

## Citation
If you find the codebase useful for your research, please cite the paper:
```
@inproceedings{li2021hierarchical,
  title={Hierarchical Consistency Regularized Mean Teacher for Semi-supervised 3D Left Atrium Segmentation},
  author={Li, Shumeng and Zhao, Ziyuan and Xu, Kaixin and Zeng, Zeng and Guan, Cuntai},
  booktitle={2021 43rd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={3395--3398},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgement

Part of the code is adapted from open-source codebase and original implementations of algorithms, 
we thank these authors for their fantastic and efficient codebase:

*  UA-MT: https://github.com/yulequan/UA-MT
