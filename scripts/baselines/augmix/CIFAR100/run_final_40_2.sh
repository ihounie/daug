python create_variants_of_set_config.py confs/baselines/augmix/pd_cifar100_wrn40-2_final.yaml 1
for seed in 0 1 2 3 4
do
	python -m TrivialAugment.train -c confs/baselines/augmix/pd_cifar100_wrn40-2_final__seed=${seed}__1try.yaml --dataroot data --tag PD_CIFAR100 --wandb_log --save results/PD_single_CIFAR100_${seed} --local_rank ${local_rank}
done
