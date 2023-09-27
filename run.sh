BATCH_SIZE=32
DATASET="cifar"

deepspeed src/main.py --dataset ${DATASET} --batch_size ${BATCH_SIZE} --deepspeed_config "src/conf/ds_config.json"