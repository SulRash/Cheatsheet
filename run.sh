BATCH_SIZE=128

MODEL="vit_base_patch8_224"
DATASET="cifar"

CHEATSHEET=0
CS_SIZE=16

EXP_NAME="NoCheatsheet-NoBlackColumn-vit_base_patch8_224-160x160-Batch128pergpu"
# EXP_NAME="DEBUGGING"

if [ $CHEATSHEET == 1 ]; then
    deepspeed --num_gpus 8 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --cheatsheet --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json"
else
    deepspeed --num_gpus 8 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json"
fi