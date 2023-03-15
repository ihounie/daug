LOCAL_RANK=0
python create_variants_of_set_config.py confs/ablations/batch/pd_cifar100_ablation.yaml 1
for seed in 0 1 2 3 4
do
	for m in 2
	do
		python -m TrivialAugment.train_pd_batch -c confs/ablations/batch/pd_cifar100_ablation__n_aug=${m}__seed=${seed}__1try.yaml --dataroot data --tag DEBUG_PD_batch_CIFAR100 --save results/PD_batch_CIFAR100_${m}_${seed} --local_rank ${LOCAL_RANK} --cv-ratio 0.1
	done
done