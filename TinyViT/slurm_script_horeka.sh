#!/bin/bash
#SBATCH -J gelu # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work) /%A.%a.out # %x.%N.%A.%a
#SBATCH -e log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work) log/%x.%N.%j.out
#SBATCH -p accelerated-h100
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:4
# SBATCH -c 1 # number of cores
#SBATCH -a 1-3


source ~/.bashrc
conda activate tinyvit

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started on $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "Running job array $SLURM_ARRAY_TASK_ID";


start=`date +%s`

results_folder='vit-golu-exps'

torchrun --nproc_per_node64 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 64 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
# torchrun --nproc_per_node64 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 64 --output ./output_{$results_folder}_golu_stable_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLU_stable
# torchrun --nproc_per_node64 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 64 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --nproc_per_node64 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 64 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu --act=Swish
# torchrun --nproc_per_node64 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 64 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu --act=ELU

# torchrun --nproc_per_node64 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 64 --output ./output_{$results_folder}_fvit5_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=fvit5 --act=fvit5








# cd AutoFormer
# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-T.yaml --batch-size 128 --num_workers 2 --data-path ./data/c10

# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-T.yaml --epochs 10 --warmup-epochs 2 --batch-size 16 --num_workers 2 --data-path ./data/c10
# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-T.yaml --batch-size 128 --num_workers 2 --data-path ./data/c10

# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 10 --warmup-epochs 2 --batch-size 16 --num_workers 2 --data-path ./data/c10

# cd TinyViT
# python main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path ./data/c10 --batch-size 128 --output ./output




end=`date +%s`

echo "DONE";
echo "Finished on $(date)";

runtime=$((end-start))
echo Runtime: $runtime

conda deactivate

