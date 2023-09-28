BATCH_SIZE=32
DATASET="cifar"

deepspeed --num_gpus 0 src/main.py --dataset ${DATASET} --cheatsheet false --batch_size ${BATCH_SIZE} --deepspeed_config "src/conf/ds_config.json"