local_rank=0
CUDA_SET_VISIBLE_DEVICES=0 python -m TrivialAugment.train_pd_batch -c confs/imnet/adam_5.yaml --dataroot ~/imnet-100-data/ --tag PD_5 --wandb_log --save results/imnet_pd_${seed} --local_rank ${local_rank} --project imnet100