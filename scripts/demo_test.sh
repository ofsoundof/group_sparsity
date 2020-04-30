#!/bin/bash
#Submit to GPU


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
--pretrain ${MODEL_PATH}/vgg_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 2. DenseNet, CIFAR10
######################################
MODEL=Hinge_DENSENET_SVD
LAYER=40
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template DenseNet --model ${MODEL} --depth ${LAYER} --test_only \
--pretrain ${MODEL_PATH}/densenet_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 3. ResNet164, CIFAR10
######################################
MODEL=Hinge_RESNET_BOTTLENECK
LAYER=164
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --no_bias --test_only \
--pretrain ${MODEL_PATH}/resnet164_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 4. ResNet164, CIFAR100
######################################
MODEL=Hinge_RESNET_BOTTLENECK
LAYER=164
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --no_bias --test_only \
--pretrain ${MODEL_PATH}/resnet164_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 5. ResNet56, CIFAR10
######################################
MODEL=Hinge_ResNet_Basic_SVD
LAYER=56
CHECKPOINT=${MODEL}_CIFAR10_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ResNet --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
--pretrain ${MODEL_PATH}/resnet56_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 6. ResNet20, CIFAR10
######################################
MODEL=Hinge_ResNet_Basic_SVD
LAYER=20
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
--pretrain ${MODEL_PATH}/resnet20_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 7. ResNet20, CIFAR100
######################################

MODEL=Hinge_ResNet_Basic_SVD
LAYER=20
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
--pretrain ${MODEL_PATH}/resnet20_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 8. ResNeXt164, CIFAR10
######################################
MODEL=Hinge_RESNEXT
LAYER=164
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext164_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 9. ResNeXt164, CIFAR100
######################################
MODEL=Hinge_RESNEXT
LAYER=164
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext164_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}

#<<COMMENT
######################################
# 10. ResNeXt20, CIFAR10
######################################
MODEL=Hinge_RESNEXT
LAYER=20
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext20_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 11. ResNeXt20, CIFAR100
######################################
MODEL=Hinge_RESNEXT
LAYER=20
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext20_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 12. WRN, CIFAR100, 0.5
######################################
MODEL=Hinge_WIDE_RESNET
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_0.5
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_Wide_ResNet" --model ${MODEL} --depth 16 --widen_factor 10 --test_only \
--pretrain ${MODEL_PATH}/wrn_cifar100_5.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 13. WRN, CIFAR100, 0.7
######################################
MODEL=Hinge_WIDE_RESNET
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_0.7
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_Wide_ResNet" --model ${MODEL} --depth 16 --widen_factor 10 --test_only \
--pretrain ${MODEL_PATH}/wrn_cifar100_7.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}

