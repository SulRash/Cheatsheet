NUM_GPUS=8
BATCH_SIZE=16

let GLOBAL_BATCH=${NUM_GPUS}*${BATCH_SIZE}

#MODEL="vit_small_patch8_224"
#MODEL="resnet18"
#MODEL="deit3_base_patch16_384_in21ft1k"
MODEL="vit_base_patch8_224"
DATASET="cifar10"

CHEATSHEET=1
CS_SIZE=8

if [ $CHEATSHEET == 1 ]; then
    EXP_NAME="ONEIMAGEPERCLASS-Cheatsheet-${DATASET}-${MODEL}-CS_SIZE_${CS_SIZE}-GlobalBatch${GLOBAL_BATCH}-RandomizeSheet"   
    deepspeed --num_gpus 8 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --one_image_per_class --randomize_sheet --batch_size ${GLOBAL_BATCH} --cheatsheet --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json" --test_interval 1
else
    EXP_NAME="NoCheatsheet-${DATASET}-${MODEL}-CS_SIZE_${CS_SIZE}-GlobalBatch${BATCH_SIZE}"
    deepspeed --num_gpus 8 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${GLOBAL_BATCH} --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json" --test_interval 1
fi