#!/bin/bash
#SBATCH -J vitt-go # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work) /%A.%a.out # %x.%N.%A.%a
#SBATCH -e log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work) log/%x.%N.%j.out
#SBATCH -p single # gpu-multi # partition (queue)
#SBATCH --nodes=1
#SBATCH -t 5-00:00:00 # time (D-HH:MM:SS)
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
#SBATCH --gres=gpu:8
# SBATCH -c 1 # number of cores
#SBATCH -a 1-3

module load devel/cuda
# export OMP_NUM_THREADS=${SLURM_NTASKS}


unset OMP_NUM_THREADS
unset OMP_PROC_BIND
unset OMP_PLACES

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=master
export OMP_PLACES=sockets


source ~/.bashrc
conda activate tinyvit

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started on $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "Running job array $SLURM_ARRAY_TASK_ID";


start=`date +%s`

results_folder='vit-cream-helix'

# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_{$results_folder}_golu_stable_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu_stable --act=GoLU_stable








# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_tinyvit_gelu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=vit-exps-meta --run-name=gelu
# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_tinyvit_golu_clamp_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=vit-exps-meta --run-name=golu_clamp --act=GoLU_stable


# cd TinyViT
# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_tinyvit_meta_gelu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=tinyvit-meta --run-name=gelu
# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_tinyvit_meta_golu_clamp_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=tinyvit-meta --run-name=golu_clamp --act=GoLU_stable


# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_meta_gelu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=TinyViT-META --run-name=gelu
# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_meta_vit1_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=TinyViT-META --run-name=vit1
# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_meta_vit2_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=TinyViT-META --run-name=vit2
# torchrun --nproc_per_node 8 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 32 --output ./output_meta_vit3_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=TinyViT-META --run-name=vit3

# cd TinyViT
# python -m torch.distributed.launch --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 128 --output ./output_temp # --use-wandb
# python -m torch.distributed.launch --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /data/datasets/ImageNet/imagenet-pytorch --batch-size 16 --output ./output_temp # --accumulation-steps=4 --use-wandb

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

