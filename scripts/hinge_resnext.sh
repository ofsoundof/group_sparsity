#!/bin/bash
#Submit to GPU


MODEL_PATH=../model_zoo/baseline
SAVE_PATH=~/projects/logs/
DATA_PATH=~/projects/data


######################################
# ResNeXt164, CIFAR10, ratio 0.4438
######################################
MODEL=Hinge_RESNEXT
RATIO=0.4438
LAYER=164
TEMPLATE=CIFAR10
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=0
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} \
--cardinality 32 --bottleneck_width 1 --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --threshold ${THRESHOLD} --annealing_factor ${ANNEAL} \
--stop_limit ${STOP_LIMIT} --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnext164_cifar10.pt \
--pretrain ${MODEL_PATH}/resnext164_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}


######################################
# ResNeXt164, CIFAR100, ratio = 0.4769
######################################
MODEL=Hinge_RESNEXT
RATIO=0.4769
LAYER=164
TEMPLATE=CIFAR100
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=0
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} \
--cardinality 32 --bottleneck_width 1 --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --threshold ${THRESHOLD} --annealing_factor ${ANNEAL} \
--stop_limit ${STOP_LIMIT} --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnext164_cifar100.pt \
--pretrain ${MODEL_PATH}/resnext164_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}

######################################
# ResNeXt20, CIFAR10, ratio 0.5921
######################################
MODEL=Hinge_RESNEXT
RATIO=0.5921
LAYER=20
TEMPLATE=CIFAR10
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=2
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} \
--cardinality 32 --bottleneck_width 1 --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --threshold ${THRESHOLD} \
--annealing_factor ${ANNEAL} --annealing_t1 75 --annealing_t2 60 \
--stop_limit ${STOP_LIMIT} --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnext20_cifar10.pt \
--pretrain ${MODEL_PATH}/resnext20_cifar10.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}

######################################
# ResNeXt20, CIFAR100, ratio = 0.5351
######################################
MODEL=Hinge_RESNEXT
RATIO=0.5351
LAYER=20
TEMPLATE=CIFAR100
LR=0.1
LR_RATIO=0.01
LR_FACTOR=1
REGULARIZER=l1
REG_FACTOR=2e-4
ANNEAL=2
THRESHOLD=5e-3
STOP_LIMIT=0.1
EPOCH=300
STEP=hingestep-150-225
CHECKPOINT=${MODEL}_${TEMPLATE}_L${LAYER}_LR${LR}r${LR_RATIO}f${LR_FACTOR}_R${REG_FACTOR}_T${THRESHOLD}_S${STOP_LIMIT}_A${ANNEAL}_E${EPOCH}_${REGULARIZER}_Ratio${RATIO}_Balance_Dis
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=1 python ../main_hinge.py --save $CHECKPOINT --template ${TEMPLATE} --model ${MODEL} --depth ${LAYER} \
--cardinality 32 --bottleneck_width 1 --batch_size 64 \
--epochs ${EPOCH} --decay ${STEP} --lr ${LR} --lr_ratio ${LR_RATIO} --lr_factor ${LR_FACTOR} --optimizer PG --ratio ${RATIO} \
--sparsity_regularizer ${REGULARIZER} --regularization_factor ${REG_FACTOR} --threshold ${THRESHOLD} \
--annealing_factor ${ANNEAL} --annealing_t1 100 --annealing_t2 75 \
--stop_limit ${STOP_LIMIT} --layer_balancing --distillation \
--teacher ${MODEL_PATH}/resnext20_cifar100.pt \
--pretrain ${MODEL_PATH}/resnext20_cifar100.pt \
--dir_save ${SAVE_PATH} \
--dir_data ${DATA_PATH}




