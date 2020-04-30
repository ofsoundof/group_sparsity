#!/bin/bash
#Submit to GPU


MODEL_PATH=../model_zoo/baseline
SAVE_PATH=~/projects/logs/
DATA_PATH=~/projects/data


######################################
# Wide_ResNet, CIFAR100, ratio = 0.5731
######################################
MODEL=Hinge_WIDE_RESNET
RATIO=0.5731
TEMPLATE=CIFAR100
LR=0.1
LR_RATIO=0.01
LR_FACTOR=0.1
REGULARIZER=l1
REG_FACTOR=4e-4
ANNEAL=2
INIT_METHOD=p-identity
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=200
STEP=hingestep-60-120-160
CHECKPOINT=${MODEL}_${TEMPLATE}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_${INIT_METHOD}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Same_Balance
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_Wide_ResNet" --model ${MODEL} --depth 16 --widen_factor 10 --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --init_method ${INIT_METHOD} --threshold ${THRESHOLD} \
--annealing_factor ${ANNEAL} --annealing_t1 400 --annealing_t2 350 --stop_limit ${STOP_LIMIT} --p1_p2_same_ratio --layer_balancing \
--teacher ${MODEL_PATH}/wide_resnet_cifar100.pt \
--pretrain ${MODEL_PATH}/wide_resnet_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# Wide_ResNet, CIFAR100, ratio = 0.7556
######################################
MODEL=Hinge_WIDE_RESNET
RATIO=0.7556
TEMPLATE=CIFAR100
LR=0.1
LR_RATIO=0.01
LR_FACTOR=0.1
REGULARIZER=l1
REG_FACTOR=4e-4
ANNEAL=2
INIT_METHOD=p-identity
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=200
STEP=hingestep-60-120-160
CHECKPOINT=${MODEL}_${TEMPLATE}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_${INIT_METHOD}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Same_Balance
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_Wide_ResNet" --model ${MODEL} --depth 16 --widen_factor 10 --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --init_method ${INIT_METHOD} --threshold ${THRESHOLD} \
--annealing_factor ${ANNEAL} --annealing_t1 400 --annealing_t2 350 --stop_limit ${STOP_LIMIT} --p1_p2_same_ratio --layer_balancing \
--teacher ${MODEL_PATH}/wide_resnet_cifar100.pt \
--pretrain ${MODEL_PATH}/wide_resnet_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}
