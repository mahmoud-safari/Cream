#!/bin/bash
#SBATCH -J mish_final # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work) /%A.%a.out # %x.%N.%A.%a
#SBATCH -e log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work) log/%x.%N.%j.out
#SBATCH -p accelerated-h100 # accelerated
#SBATCH -t 1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# SBATCH --cpus-per-task=12 # ----
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:4
# SBATCH -c 1 # number of cores
#SBATCH -a 1-3


source ~/.bashrc
conda activate tinyvit

# export CC=$(which gcc)
# export CXX=$(which g++)


export MODEL_NAME="TinyViT"
export ACTIVATION="GoLUCUDA"
export DATA_PATH="$TMPDIR/$MODEL_NAME/$ACTIVATION"
mkdir -p $DATA_PATH
rsync -ahr --progress "/hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet_tar/" $DATA_PATH


# Print some information about the job to STDOUT
echo "Data directory: $DATA_PATH";
echo "Workingdir: $PWD";
echo "Started on $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "Running job array $SLURM_ARRAY_TASK_ID";

echo $TMPDIR



start=`date +%s`

results_folder='vit-golucuda-cream-horeka'

srun --ntasks=1 --cpus-per-task=12 python -m setup_dataset --dataset_path $DATA_PATH --dataset_name "imagenet_1k"


# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_golu_cuda_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu-final --act=GoLUCUDA
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu-final
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu-final --act=ReLU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu-final --act=Swish
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu-final --act=ELU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_lrelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=leakyrelu-final --act=LeakyReLU
torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_mish_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=mish-final --act=Mish




# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_golu_cuda_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLUCUDA
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu --act=Swish
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu --act=ELU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_lrelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=leakyrelu --act=LeakyReLU







# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data --batch-size 256 --output ./output_{$results_folder}_golu_cuda_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLUCUDA
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data --batch-size 256 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data --batch-size 256 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data --batch-size 256 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu --act=Swish
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data --batch-size 256 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu --act=ELU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data --batch-size 256 --output ./output_{$results_folder}_lrelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=leakyrelu --act=LeakyReLU


# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet --batch-size 256 --output ./output_{$results_folder}_golu_cuda_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLUCUDA
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet --batch-size 256 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet --batch-size 256 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet --batch-size 256 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu --act=Swish
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet --batch-size 256 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu --act=ELU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet --batch-size 256 --output ./output_{$results_folder}_lrelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=leakyrelu --act=LeakyReLU




# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_golu_cuda_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLUCUDA
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu --act=Swish
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu --act=ELU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_lrelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=leakyrelu --act=LeakyReLU









# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 256 --output ./output_{$results_folder}_golu_cuda_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLUCUDA
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 256 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 256 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 256 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu --act=Swish
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 256 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu --act=ELU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 256 --output ./output_{$results_folder}_lrelu_{$SLURM_ARRAY_TASK_ID} --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=leakyrelu --act=LeakyReLU





# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_golu_cuda_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLUCUDA


# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_gelu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=gelu
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_golu_stable_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=golu --act=GoLU_stable
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_relu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_silu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=silu --act=Swish
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_elu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=elu --act=ELU
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_lrelu_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=leakyrelu --act=LeakyReLU


# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_fvit1_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=fvit1 --act=fvit1
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_fvit2_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=fvit2 --act=fvit2
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_fvit3_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=fvit3 --act=fvit3
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_fvit4_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=fvit4 --act=fvit4
# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_fvit5_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=fvit5 --act=fvit5

# torchrun --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path /home/hk-project-p0021863/fr_ms2108/imagenet --batch-size 64 --output ./output_{$results_folder}_fvit52_{$SLURM_ARRAY_TASK_ID} --accumulation-steps=4 --seed=$SLURM_ARRAY_TASK_ID --use-wandb --project=$results_folder --run-name=fvit52 --act=fvit52







# cd AutoFormer
# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-T.yaml --batch-size 128 --num_workers 2 --data-path ./data/c10

# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-T.yaml --epochs 10 --warmup-epochs 2 --batch-size 16 --num_workers 2 --data-path ./data/c10
# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-T.yaml --batch-size 128 --num_workers 2 --data-path ./data/c10

# python supernet_train.py --data-set=CIFAR10 --gp --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 10 --warmup-epochs 2 --batch-size 16 --num_workers 2 --data-path ./data/c10

# cd TinyViT
# python main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path ./data/c10 --batch-size 128 --output ./output


# torchrun --standalone --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_relu_{1} --seed=1 --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --standalone --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_relu_{2} --seed=2 --use-wandb --project=$results_folder --run-name=relu --act=ReLU
# torchrun --standalone --nproc_per_node 4 main.py --cfg configs/1k/tiny_vit_21m.yaml --data-path $DATA_PATH --batch-size 256 --output ./output_{$results_folder}_relu_{3} --seed=3 --use-wandb --project=$results_folder --run-name=relu --act=ReLU



end=`date +%s`

echo "DONE";
echo "Finished on $(date)";

runtime=$((end-start))
echo Runtime: $runtime

conda deactivate

