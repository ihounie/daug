python create_variants_of_set_config.py confs/ablations/margin/wrn40-2/pd_cifar10_ablation.yaml 1
for seed in 0 1 2 3
do
	for margin in 0.2 0.5 0.8 1.1 1.4 1.7 2.0 2.3 2.6
	do
		python -m TrivialAugment.train_pd_batch -c confs/ablations/margin/pd_cifar10_ablation__seed=${seed}__PD.margin=${margin}__1try.yaml --dataroot data --tag PD_ablation_CIFAR10 --wandb_log --save results/PD_single_CIFAR100_${margin}_${seed}
	done
done
