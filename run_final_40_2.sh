for seed in 0 1 2 3 4
do
	python -m TrivialAugment.train_pd_batch -c confs/pd_cifar10_wrn40-2_final__seed=${seed}__1try.yaml --dataroot data --tag PD_CIFAR10 --wandb_log --save results/PD_single_CIFAR10_${seed}
	python -m TrivialAugment.train_pd_batch -c confs/pd_cifar100_wrn40-2_final__seed=${seed}__1try.yaml --dataroot data --tag PD_CIFAR100 --wandb_log --save results/PD_single_CIFAR100_${seed}
done
