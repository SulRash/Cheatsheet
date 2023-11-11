NUM_GPUS=2
BATCH_SIZE=96

let GLOBAL_BATCH=${NUM_GPUS}*${BATCH_SIZE}

#MODEL="vit_small_patch8_224"
#MODEL="resnet18"
#MODEL="deit3_base_patch16_384_in21ft1k"
MODEL="vit_base_patch8_224"
#MODEL="vit_large_patch14_224"
DATASET="cifar10"
IMG_PER_CLASS=0
CHEATSHEET=1
CS_SIZE=8

if [ $CHEATSHEET == 1 ]; then
    EXP_NAME="lr2e_5-Cheatsheet-${DATASET}-${MODEL}-CS_SIZE${CS_SIZE}-GlobalBatch${GLOBAL_BATCH}-RandomizeSheet-ImgPerClass${IMG_PER_CLASS}"
    deepspeed --num_gpus 2 src/train.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --exp_name ${EXP_NAME} \
    --img_per_class ${IMG_PER_CLASS} \
    --randomize_sheet \
    --batch_size ${GLOBAL_BATCH} \
    --cheatsheet --cs_size ${CS_SIZE} \
    --deepspeed_config "src/conf/ds_config.json" \
    --save_interval 250 \
    --test_interval 5 \
    --train_epochs 1500
else
    EXP_NAME="NoCheatsheet-${DATASET}-${MODEL}-CS_SIZE_${CS_SIZE}-GlobalBatch${BATCH_SIZE}"
    deepspeed --num_gpus 8 src/main.py --model ${MODEL} --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${GLOBAL_BATCH} --cs_size ${CS_SIZE} --deepspeed_config "src/conf/ds_config.json" --test_interval 1
fi