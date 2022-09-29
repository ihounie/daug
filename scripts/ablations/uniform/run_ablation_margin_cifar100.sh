python create_variants_of_set_config.py confs/ablations/uniform/pd_cifar100_ablation.yaml 1
for seed in 0 1 2 3 4
do
	for margin in 1.2 0.9 0.6
	do
		python -m TrivialAugment.train_pd_batch -c confs/ablations/uniform/pd_cifar100_ablation__seed=${seed}__PD.margin=${margin}__1try.yaml --dataroot data --tag PD_ablation_CIFAR10 --wandb_log --save results/PD_single_CIFAR100_${margin}_${seed} --local_rank ${LOCAL_RANK} --cv-ratio 0.1
	done
done