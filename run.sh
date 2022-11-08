
# BART 돌리는 것
CUDA_VISIBLE_DEVICES=3,4,6,7 nohup python train.py \
--save_filename /home/mnt/hyeon/2.checkpoints/LFQA/gen_model/psg_top10/bart \
--lr 1e-05 \
--gpus 4 \
--accelerator ddp \
--batch_size 8 \
--precision 16 \
--num_node 4 \
--plugins deepspeed_stage_2 \
--model_type bart > logs/psg_top10-bart.log &


# longt5돌리는 것
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py \
--save_filename /home/mnt/hyeon/2.checkpoints/LFQA/gen_model/psg_top10/longt5 \
--lr 1e-05 \
--gpus 4 \
--accelerator ddp \
--batch_size 4 \
--precision 32 \
--max_len 1024 \
--num_node 8 \
--plugins deepspeed_stage_2 \
--model_type longt5 > logs/psg_top10-longt5.log &
