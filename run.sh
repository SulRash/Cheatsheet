BATCH_SIZE=32
DATASET="cifar"
CHEATSHEET=0
EXP_NAME="CheatsheetResnet18-88x80-Batch32"

deepspeed --num_gpus 0 src/main.py --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --deepspeed_config "src/conf/ds_config.json" --cheatsheet --cs_size 8