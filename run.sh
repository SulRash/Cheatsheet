BATCH_SIZE=128

#MODEL="vit_small_patch8_224"
#MODEL="resnet18"
#MODEL="deit3_base_patch16_384_in21ft1k"
MODEL="vit_base_patch8_224"
DATASET="cifar100"

CHEATSHEET=1
CS_SIZE=8

if [ $CHEATSHEET == 1 ]; then
    EXP_NAME="Cheatsheet-${DATASET}-${MODEL}-CS_SIZE_${CS_SIZE}-Batch${BATCH_SIZE}pergpu"   
    deepspeed --num_gpus 1 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --cheatsheet --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json"
else

    EXP_NAME="NoCheatsheet-${DATASET}-${MODEL}-CS_SIZE_${CS_SIZE}-Batch${BATCH_SIZE}pergpu"
    deepspeed --num_gpus 1 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json"
fi