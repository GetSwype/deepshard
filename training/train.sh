python3 -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=5102 /root/documents/deepshard/training/train.py \
    --model_name_or_path swype/deepshard-13B-raw \
    --data_path /root/documents/deepshard/datasets/data/train.jsonl \
    --eval_path /root/documents/deepshard/datasets/data/test.jsonl \
    --search False \
    --fp16 True \
    --output_dir /root/documents/deepshard/training/finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

# python3 /home/paperspace/Documents/deepshard/training/push.py