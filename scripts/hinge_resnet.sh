#!/bin/bash
#Submit to GPU


MODEL_PATH=../model_zoo/baseline
SAVE_PATH=~/projects/logs/
DATA_PATH=~/projects/data


######################################
# ResNet56, CIFAR100, ratio = 0.5, svd2
######################################
MODEL=Hinge_ResNet_Basic_SVD
RATIO=0.5
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1d2
REG_FACTOR=4e-4
ANNEAL=2
INIT_METHOD=svd2
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_CIFAR10_L56_LR${LR}r${LR_RATIO}f${LR_FACTOR}_${INIT_METHOD}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Same_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ResNet --model ${MODEL} --depth 56 --batch_size 64 --downsample_type A \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --init_method ${INIT_METHOD} --threshold ${THRESHOLD} --annealing_factor ${ANNEAL} \
--stop_limit ${STOP_LIMIT} --p1_p2_same_ratio --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnet56_b128e164.pt \
--pretrain ${MODEL_PATH}/resnet56_b128e164.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# ResNet20, CIFAR100, ratio = 0.3298
######################################
MODEL=Hinge_ResNet_Basic_SVD
RATIO=0.3298
LAYER=20
TEMPLATE=CIFAR100
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=2
INIT_METHOD=svd2
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_${INIT_METHOD}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Same_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_ResNet${LAYER}" --model ${MODEL} --depth ${LAYER} --batch_size 64 --downsample_type A \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --init_method ${INIT_METHOD} --threshold ${THRESHOLD} \
--annealing_factor ${ANNEAL} --annealing_t1 30 --annealing_t2 25 \
--stop_limit ${STOP_LIMIT} --p1_p2_same_ratio --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnet20_cifar100.pt \
--pretrain ${MODEL_PATH}/resnet20_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# ResNet20, CIFAR10, ratio = 0.4516
######################################
MODEL=Hinge_ResNet_Basic_SVD
RATIO=0.4516
LAYER=20
TEMPLATE=CIFAR10
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=2
INIT_METHOD=svd2
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_${INIT_METHOD}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Same_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_ResNet${LAYER}" --model ${MODEL} --depth ${LAYER} --batch_size 64 --downsample_type A \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --init_method ${INIT_METHOD} --threshold ${THRESHOLD} \
--annealing_factor ${ANNEAL} --annealing_t1 20 --annealing_t2 15 \
--stop_limit ${STOP_LIMIT} --p1_p2_same_ratio --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnet20_cifar10.pt \
--pretrain ${MODEL_PATH}/resnet20_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# ResNet164, CIFAR100, ratio = 0.5533
######################################
MODEL=Hinge_RESNET_BOTTLENECK
RATIO=0.5533
LAYER=164
TEMPLATE=CIFAR100
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=0
THRESHOLD=1e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Same_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_ResNet${LAYER}" --model ${MODEL} --depth ${LAYER} --no_bias --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --threshold ${THRESHOLD} --annealing_factor ${ANNEAL} \
--stop_limit ${STOP_LIMIT} --p1_p2_same_ratio --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnet164_cifar100.pt \
--pretrain ${MODEL_PATH}/resnet164_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}

######################################
# ResNet164, CIFAR10, ratio = 0.5353
######################################
MODEL=Hinge_RESNET_BOTTLENECK
RATIO=0.5353
LAYER=164
TEMPLATE=CIFAR10
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=0
THRESHOLD=1e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Same_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../main_hinge.py --save $CHECKPOINT --template "${TEMPLATE}_ResNet${LAYER}" --model ${MODEL} --depth ${LAYER}  --no_bias --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --threshold ${THRESHOLD} --annealing_factor ${ANNEAL} \
--stop_limit ${STOP_LIMIT} --p1_p2_same_ratio --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnet164_cifar10.pt \
--pretrain ${MODEL_PATH}/resnet164_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}

