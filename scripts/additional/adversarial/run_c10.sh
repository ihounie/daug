python create_variants_of_set_config.py confs/additional/adversarial/cifar10_wrn40-2.yaml 1
for seed in 0 1 2 3 4
do
	python -m TrivialAugment.train -c confs/additional/adversarial/cifar10_wrn40-2__seed=${seed}__1try.yaml --dataroot data --tag ADV_CIFAR10 --wandb_log --save results/ALL_CIFAR10_${seed} --local_rank 0  --project DAug-Gen-adv
done