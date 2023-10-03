BATCH_SIZE=128

MODEL="vit_small_patch8_224"
DATASET="cifar"

CHEATSHEET=0
CS_SIZE=16
let CS_SIZE_TIMES_TEN=${CS_SIZE}*10
let CS_SIZE_TIMES_ELEVEN=${CS_SIZE}*11

if [ $CHEATSHEET == 1 ]; then
    EXP_NAME="Cheatsheet-${MODEL}-${CS_SIZE_TIMES_ELEVEN}x${CS_SIZE_TIMES_TEN}-Batch${BATCH_SIZE}pergpu"   
    deepspeed --num_gpus 8 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --cheatsheet --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json"
else

    EXP_NAME="NoCheatsheet-${MODEL}-${CS_SIZE_TIMES_TEN}x${CS_SIZE_TIMES_TEN}-Batch${BATCH_SIZE}pergpu"
    deepspeed --num_gpus 8 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json"
fi