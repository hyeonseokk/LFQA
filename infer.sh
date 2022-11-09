CUDA_VISIBLE_DEVICES=0 nohup python inference.py \
--ckpt /home/mnt/hyeon/2.checkpoints/LFQA/gen_model/phr_top200/bart \
--testfile /home/mnt/hyeon/1.Dataset/LFQA/split/processed/dev_preprocessed_1507_top200_phrase-top200.json > logs/gen_phr_phr_bart &
