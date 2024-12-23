#!/bin/bash
#SBATCH -J grid_cnn # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work) /%A.%a.out # %x.%N.%A.%a
#SBATCH -e log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work) log/%x.%N.%j.out
#SBATCH -p dev_accelerated # accelerated-h100
#SBATCH -t 0-01:00:00
#SBATCH --nodes=1
# SBATCH --ntasks-per-node=4
# SBATCH --cpus-per-task=12
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
# SBATCH -c 1 # number of cores
# SBATCH -a 1-3


source ~/.bashrc
conda activate tinyvit

# export CC=$(which gcc)
# export CXX=$(which g++)



# Print some information about the job to STDOUT
echo "Data directory: $DATA_PATH";
echo "Workingdir: $PWD";
echo "Started on $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "Running job array $SLURM_ARRAY_TASK_ID";



start=`date +%s`

results_folder='cnn_grid'

python cnn_grid_search.py --log-wandb #--wandb_project=$results_folder


end=`date +%s`

echo "DONE";
echo "Finished on $(date)";

runtime=$((end-start))
echo Runtime: $runtime

conda deactivate

