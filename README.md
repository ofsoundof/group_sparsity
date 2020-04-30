# Group Sparsity
This is the official code implementing "Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression".

## Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Train and Test](#train-and-test)
4. [Results](#results)
5. [Reference](#reference)
6. [Acknowledgements](#acknowledgements)

## Introduction
In this paper, we analyze two popular network compression techniques, i.e. filter pruning and low-rank decomposition, in a unified sense. By simply changing the way the sparsity regularization is enforced, filter pruning and lowrank decomposition can be derived accordingly. This provides another flexible choice for network compression because the techniques complement each other. For example, in popular network architectures with shortcut connections (e.g. ResNet), filter pruning cannot deal with the last convolutional layer in a ResBlock while the low-rank decomposition methods can. In addition, we propose to compress the whole network jointly instead of in a layer-wise manner. Our approach proves its potential as it compares favorably to the state-of-the-art on several benchmarks.

<img src="/figs/teaser.png" width="400">

A sparsity-inducing matrix A is attached to a normal convolution. The matrix acts as the hinge between filter pruning and decomposition. By enforcing group sparsity to the columns and rows of the matrix, equivalent pruning and decomposition operations can be obtained. 

<img src="/figs/flowchart.png" width="400">

The flowchart of the proposed algorithm.


## Dependencies
* Python 3.7.4
* PyTorch >= 1.2.0
* numpy
* matplotlib
* tqdm
* scikit-image
* easydict
* IPython


## Results

<img src="/figs/hinge_kse_flops.eps" width="900">

<img src="/figs/hinge_kse_params.eps" width="600">

<img src="/figs/resnet164_cifar100.eps" width="400">

<img src="/figs/resnext164_cifar100.eps" width="400">


## Reference
If you find our work useful in your research of publication, please cite our work:

```
@inproceedings{li2020group,
  title={Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression},
  author={Li, Yawei and Gu, Shuhang and Mayer, Christoph and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2020}
}
```
