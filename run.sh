NUM_GPUS=8
BATCH_SIZE=512

let GLOBAL_BATCH=${NUM_GPUS}*${BATCH_SIZE}

#MODEL="vit_small_patch8_224"
#MODEL="resnet18"
#MODEL="deit3_base_patch16_384_in21ft1k"
MODEL="vit_base_patch8_224"
#MODEL="vit_large_patch14_224"
DATASET="cifar10"
IMG_PER_CLASS=0
CHEATSHEET=1
CS_SIZE=14

EXP_NAME="1CYC-Cheatsheet-${DATASET}-${MODEL}-CS_SIZE${CS_SIZE}-GlobalBatch${GLOBAL_BATCH}-RandomizeSheet-ImgPerClass${IMG_PER_CLASS}"

mkdir experiments/${EXP_NAME} -p
cp src/conf/ds_config.json experiments/${EXP_NAME}/ds_config.json

deepspeed --num_gpus ${NUM_GPUS} src/train.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --exp_name ${EXP_NAME} \
    --img_per_class ${IMG_PER_CLASS} \
    --randomize_sheet \
    --batch_size ${GLOBAL_BATCH} \
    --cheatsheet --cs_size ${CS_SIZE} \
    --deepspeed_config "src/conf/ds_config.json" \
    --save_interval 1000 \
    --test_interval 10 \
    --train_epochs 10000
