python create_variants_of_set_config.py confs/ablations/margin/wrn40-2/pd_cifar100_ablation.yaml 1
for seed in 0 1 2 3
do
	for margin in 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7
	do
		python -m TrivialAugment.train_pd_batch -c confs/grid-search/RA/pd_cifar100_ablation__seed=${seed}__PD.margin=${margin}__1try.yaml --dataroot data --tag PD_ablation_CIFAR10 --wandb_log --save results/PD_single_CIFAR100_${margin}_${seed} --local_rank ${LOCAL_RANK} --cv-ratio 0.1
	done
done