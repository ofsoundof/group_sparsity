#!/bin/bash
#Submit to GPU


MODEL_PATH=../model_zoo/baseline
SAVE_PATH=~/projects/logs/
DATA_PATH=~/projects/data


######################################
# DenseNet, CIFAR10, ratio = 0.5522, svd2
######################################
MODEL=Hinge_DENSENET_SVD
RATIO=0.5522
TEMPLATE=CIFAR10
LR=0.1
LR_RATIO=0.01
LR_FACTOR=0.1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=2
INIT_METHOD=svd2
THRESHOLD=1e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_${INIT_METHOD}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_DenseNet" --model ${MODEL} --depth 40 --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --init_method ${INIT_METHOD} --threshold ${THRESHOLD} --annealing_factor ${ANNEAL} \
--stop_limit ${STOP_LIMIT} --distillation \
--teacher ${MODEL_PATH}/densenet_cifar10.pt \
--pretrain ${MODEL_PATH}/densenet_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


