local_rank=1
method="baselines"
python create_variants_of_set_config.py confs/imnet/${method}/adam_5.yaml 1
for seed in 1 2 3 4
do
	CUDA_VISIBLE_DEVICES=${local_rank} python -m TrivialAugment.train -c confs/imnet/${method}/adam_5__seed=${seed}__1try.yaml --dataroot ~/imnet-100-data/ --tag PD_Adam_5 --wandb_log --save results/imnet_pd_${seed} --project imnet100
done