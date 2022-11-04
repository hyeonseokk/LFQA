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
