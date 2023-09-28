BATCH_SIZE=128
DATASET="cifar"

deepspeed --num_gpus 0 src/main.py --dataset ${DATASET} --batch_size ${BATCH_SIZE} --deepspeed_config "src/conf/ds_config.json"