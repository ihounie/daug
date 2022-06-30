for seed in 0 1 2 3 4
do
	for margin in 0.2 0.5 0.8 1.1 1.4 1.7 2.0 2.3 2.6
	do
		python -m TrivialAugment.train_pd_batch -c confs/pd_cifar10_ablation__seed=${seed}__PD.margin=${margin}__1try.yaml --dataroot data --tag PD_ablation_CIFAR10 --wandb_log --save results/PD_single_CIFAR100_${margin}_${seed}
	done
done
