python create_variants_of_set_config.py confs/baselines/TA/wide/pd_svhn_wrn28-10_final.yaml 1
for seed in 0 1 2 3 4
do
	python -m TrivialAugment.train -c confs/baselines/TA/wide/pd_svhn_wrn28-10_final__seed=${seed}__1try.yaml --dataroot data --tag PD_CIFAR10 --wandb_log --save results/PD_single_CIFAR10_${seed}
done