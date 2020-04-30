#!/bin/bash


MODEL_PATH=../model_zoo/baseline
SAVE_PATH=~/projects/logs/hinge_test/
DATA_PATH=~/projects/data


######################################
# 1. VGG, CIFAR10
######################################
MODEL=VGG
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template "linear3_${TEMPLATE}_VGG" --model ${MODEL} --vgg_type 16 --test_only \
--pretrain ${MODEL_PATH}/vgg.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 2. DenseNet, CIFAR10
######################################
MODEL=DENSENET
LAYER=40
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template DenseNet --model ${MODEL} --depth ${LAYER} --test_only \
--pretrain ${MODEL_PATH}/densenet_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 3. ResNet164, CIFAR10
######################################
MODEL=ResNet
LAYER=164
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER}  --no_bias --test_only \
--pretrain ${MODEL_PATH}/resnet164_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 4. ResNet164, CIFAR100
######################################
MODEL=ResNet
LAYER=164
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER}  --no_bias --test_only \
--pretrain ${MODEL_PATH}/resnet164_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 5. ResNet56, CIFAR10
######################################
MODEL=ResNet
LAYER=56
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
--pretrain ${MODEL_PATH}/resnet56_b128e164.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 6. ResNet20, CIFAR10
######################################
MODEL=ResNet
LAYER=20
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
--pretrain ${MODEL_PATH}/resnet20_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 7. ResNet20, CIFAR100
######################################
MODEL=ResNet
LAYER=20
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --downsample_type A --test_only \
--pretrain ${MODEL_PATH}/resnet20_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 8. ResNeXt164, CIFAR10
######################################
MODEL=ResNeXt
LAYER=164
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext164_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 9. ResNeXt164, CIFAR100
######################################
MODEL=ResNeXt
LAYER=164
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext164_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH} 


######################################
# 10. ResNeXt20, CIFAR10
######################################
MODEL=ResNeXt
LAYER=20
TEMPLATE=CIFAR10
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext20_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 11. ResNeXt20, CIFAR100
######################################
MODEL=ResNeXt
LAYER=20
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} --cardinality 32 --bottleneck_width 1 --test_only \
--pretrain ${MODEL_PATH}/resnext20_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# 12. WRN, CIFAR100
######################################
MODEL=Wide_ResNet
TEMPLATE=CIFAR100
CHECKPOINT=${MODEL}_${TEMPLATE}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main.py --save $CHECKPOINT --template "${TEMPLATE}_Wide_ResNet" --model ${MODEL} --depth 16 --widen_factor 10 --test_only \
--pretrain ${MODEL_PATH}/wide_resnet_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


