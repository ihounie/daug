for seed in 0 1 2 3
do
	for steps in 4 8 16
	do
		python -m TrivialAugment.train_pd_batch -c confs/pd_cifar10_wrn40-2__seed=${seed}__MH.steps=${steps}__1try.yaml --dataroot data --tag PD_ablation_CIFAR10 --wandb_log --save results/PD_single_CIFAR10_steps_${steps}_seed_${seed} --cv-ratio 0.1
	done
done
