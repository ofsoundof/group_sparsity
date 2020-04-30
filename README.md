# [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935)
This is the official implementation of "Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression".

## Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Contribution](#contribution)
4. [Dependencies](#dependencies)
5. [Test](#test)
5. [Train](#train)
4. [Results](#results)
5. [Reference](#reference)
6. [Acknowledgements](#acknowledgements)

## Introduction
In this paper, we analyze two popular network compression techniques, i.e. filter pruning and low-rank decomposition, in a unified sense. By simply changing the way the sparsity regularization is enforced, filter pruning and lowrank decomposition can be derived accordingly. This provides another flexible choice for network compression because the techniques complement each other. For example, in popular network architectures with shortcut connections (e.g. ResNet), filter pruning cannot deal with the last convolutional layer in a ResBlock while the low-rank decomposition methods can. In addition, we propose to compress the whole network jointly instead of in a layer-wise manner. Our approach proves its potential as it compares favorably to the state-of-the-art on several benchmarks.

## Motivation
Filter pruning and filter decomposition (also termed low-rank approximation) have been developing steadily. Filter pruning nullifies the weak filter connections that have the least influence on the accuracy of the network while low-rank decomposition converts a heavy convolution to a lightweight one and a linear combination. Despite their success, both the pruning-based and decomposition-based approaches have their respective limitations. Filter pruning can only take effect in pruning output channels of a tensor and equivalently cancelling out inactive filters. This is not feasible under some circumstances. The skip connection in a block is such a case where the output feature map of the block is added to the input. Thus, pruning the output could amount to cancelling a possible important input feature map. This is the reason why many pruning methods fail to deal with the second convolution of the ResNet basic block. As for filter decomposition, it always introduces another 1-by-1 convolutional layer, which means additional overhead of calling CUDA kernels. In this paper, we analyze the relationship between the two techniques from the perspective of compact tensor approximation.

<img src="/figs/teaser.png" width="900">

A sparsity-inducing matrix A is attached to a normal convolution. The matrix acts as the hinge between filter pruning and decomposition. By enforcing group sparsity to the columns and rows of the matrix, equivalent pruning and decomposition operations can be obtained. 

## Contribution
**1. The connection between filter pruning and decomposition is analyzed from the perspective of compact tensor approximation.**

**2. A sparsity-inducing matrix is introduced to hinge filter pruning and decomposition and bring them under the same formulation.**

**3. A bunch of techniques including binary search, gradient based learning rate adjustment, layer balancing, and annealing methods are developed to solve the problem.**

**4. The proposed method can be applied to various CNNs. We apply this method to VGG, DenseNet, ResNet, ResNeXt, and WRN.**

<img src="/figs/flowchart.png" width="500">

The flowchart of the proposed algorithm.

<img src="/figs/group_sparsity_column.png" width="500">
Group sparsity enforced on the column of the sparsity-inducing matrix.

<img src="/figs/group_sparsity_row.png" width="500">
Group sparsity enforced on the row of the sparsity-inducing matrix.

## Dependencies
* Python 3.7.4
* PyTorch >= 1.2.0
* numpy
* matplotlib
* tqdm
* scikit-image
* easydict
* IPython

## Test

1. Download the model zoo from [Google Drive](https://drive.google.com/file/d/1B057k6BHFXDUWFypuuIiGRqiXHnqUr1y/view?usp=sharing) or [Baidu Wangpan (extraction code: )](). This contains the pretrained original models and the compressed models. Place the models in `./model_zoo`.

2. Cd to [`./scripts`](./scripts). 

3. Use the following scripts in [`./scripts/demo_test.sh`](./scripts/demo_test.sh) to test the compressed models. 

    Be sure the change the directories `SAVE_PATH` and `DATA_PATH`.

    `SAVE_PATH`: where the dataset is stored.

    `SAVE_PATH`: where you want to save the results.

```bash
	MODEL_PATH=../model_zoo/compressed
    SAVE_PATH=~/projects/logs/hinge_test/new
    DATA_PATH=~/projects/data

    ######################################
    # 1. VGG, CIFAR10
    ######################################
    MODEL=Hinge_VGG
    TEMPLATE=CIFAR10
    CHECKPOINT=${MODEL}_${TEMPLATE}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template "linear3_${TEMPLATE}_VGG" --model ${MODEL} --vgg_type 16 --test_only \
    --pretrain ${MODEL_PATH}/vgg_cifar10.pt --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 2. DenseNet, CIFAR10
    ######################################
    MODEL=Hinge_DENSENET_SVD
    LAYER=40
    TEMPLATE=CIFAR10
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template DenseNet --model ${MODEL} --depth ${LAYER} --test_only \
    --pretrain ${MODEL_PATH}/densenet_cifar10.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 3. ResNet164, CIFAR10
    ######################################
    MODEL=Hinge_RESNET_BOTTLENECK
    LAYER=164
    TEMPLATE=CIFAR10
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --no_bias --test_only \
    --pretrain ${MODEL_PATH}/resnet164_cifar10.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 4. ResNet164, CIFAR100
    ######################################
    MODEL=Hinge_RESNET_BOTTLENECK
    LAYER=164
    TEMPLATE=CIFAR100
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --no_bias --test_only \
    --pretrain ${MODEL_PATH}/resnet164_cifar100.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 5. ResNet56, CIFAR10
    ######################################
    MODEL=Hinge_ResNet_Basic_SVD
    LAYER=56
    CHECKPOINT=${MODEL}_CIFAR10_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ResNet --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
    --pretrain ${MODEL_PATH}/resnet56_cifar10.pt --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 6. ResNet20, CIFAR10
    ######################################
    MODEL=Hinge_ResNet_Basic_SVD
    LAYER=20
    TEMPLATE=CIFAR10
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
    --pretrain ${MODEL_PATH}/resnet20_cifar10.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 7. ResNet20, CIFAR100
    ######################################
    MODEL=Hinge_ResNet_Basic_SVD
    LAYER=20
    TEMPLATE=CIFAR100
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
    --pretrain ${MODEL_PATH}/resnet20_cifar100.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 8. ResNeXt164, CIFAR10
    ######################################
    MODEL=Hinge_RESNEXT
    LAYER=164
    TEMPLATE=CIFAR10
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
    --pretrain ${MODEL_PATH}/resnext164_cifar10.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 9. ResNeXt164, CIFAR100
    ######################################
    MODEL=Hinge_RESNEXT
    LAYER=164
    TEMPLATE=CIFAR100
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
    --pretrain ${MODEL_PATH}/resnext164_cifar100.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 10. ResNeXt20, CIFAR10
    ######################################
    MODEL=Hinge_RESNEXT
    LAYER=20
    TEMPLATE=CIFAR10
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
    --pretrain ${MODEL_PATH}/resnext20_cifar10.pt --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 11. ResNeXt20, CIFAR100
    ######################################
    MODEL=Hinge_RESNEXT
    LAYER=20
    TEMPLATE=CIFAR100
    CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
    --pretrain ${MODEL_PATH}/resnext20_cifar100.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 12. WRN, CIFAR100, 0.5
    ######################################
    MODEL=Hinge_WIDE_RESNET
    TEMPLATE=CIFAR100
    CHECKPOINT=${MODEL}_${TEMPLATE}_0.5
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_Wide_ResNet" --model ${MODEL} --depth 16 --widen_factor 10 --test_only \
    --pretrain ${MODEL_PATH}/wrn_cifar100_5.pt  --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}

    ######################################
    # 13. WRN, CIFAR100, 0.7
    ######################################
    MODEL=Hinge_WIDE_RESNET
    TEMPLATE=CIFAR100
    CHECKPOINT=${MODEL}_${TEMPLATE}_0.7
    echo $CHECKPOINT
    CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_Wide_ResNet" --model ${MODEL} --depth 16 --widen_factor 10 --test_only \
    --pretrain ${MODEL_PATH}/wrn_cifar100_7.pt --dir_save ${SAVE_PATH} --dir_data ${DATA_PATH}
```

    To test the original uncompressed models, please refer to [`./scripts/baseline_test.sh`](./scripts/baseline_test.sh)

## Train
The scripts for compressing ResNet, DenseNet, VGG, ResNeXt, and WRN are released.

1. Cd to [`./scripts`](./scripts)

2. Make sure that the pretrained original models are already downloaded and placed in `./model_zoo/baseline`.

3. Run the scripts `hinge_XXX.sh` to reproduce the results in our paper, where `XXX` may be replace by `vgg`, `densenet`, `resnet`, `resnext`, and `wide_resnet` depending on which network you want to compress. 

4. Be sure the change the directories `SAVE_PATH` and `DATA_PATH` in `hinge_XXX.sh`.
 
## Results

<img src="/figs/hinge_kse_flops.png" width="400"> <img src="/figs/hinge_kse_params.png" width="400">

FLOP and parameter comparison between KSE and Hinge under different compression ratio. ResNet56 is compressed. Top-1 error rate is reported.

<img src="/figs/resnet164_cifar100.png" width="400"> <img src="/figs/resnext164_cifar100.png" width="400">

Comparison between SSS and the proposed Hinge method on ResNet and ResNeXt. Top-1 error rate is reported for CIFAR100.

<img src="/figs/table1.png" width="550">
<img src="/figs/table2.png" width="500">
<img src="/figs/table4.png" width="450">


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
