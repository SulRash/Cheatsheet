BATCH_SIZE=128
DATASET="cifar"
CHEATSHEET=0
EXP_NAME="NoCheatsheetResnet18-88x80-Batch128pergpu"

deepspeed --num_gpus 8 src/main.py --dataset ${DATASET} --exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --deepspeed_config "src/conf/ds_config.json" --cs_size 8