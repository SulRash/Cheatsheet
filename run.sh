BATCH_SIZE=32
DATASET="cifar"
CHEATSHEET=0
EXP_NAME="NoCheatsheetResnet18-Batch32"

deepspeed --num_gpus 0 src/main.py --dataset ${DATASET} --cheatsheet ${CHEATSHEET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --deepspeed_config "src/conf/ds_config.json"