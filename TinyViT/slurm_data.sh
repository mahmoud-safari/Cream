#!/bin/bash
#SBATCH -J imgnet # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work) /%A.%a.out # %x.%N.%A.%a
#SBATCH -e log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work) log/%x.%N.%j.out
#SBATCH -p mldlc_gpu-rtx2080 # alldlc_gpu-rtx2080 # ml_gpu-rtx2080 # mldlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 1-00:00:00 # time (D-HH:MM:SS)
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
# SBATCH --gres=gpu:8
# SBATCH -c 1 # number of cores
# SBATCH -a 0-4


# source ~/.bashrc
# conda activate tinyvit

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started on $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "Running job array $SLURM_ARRAY_TASK_ID";


start=`date +%s`

scp -r -o ProxyJump=safarim@aadlogin.informatik.uni-freiburg.de /data/datasets/ImageNet/imagenet-pytorch di35quh@pvc.supermuc.lrz.de:/dss/dsshome1/07/di35quh/data

end=`date +%s`

echo "DONE";
echo "Finished on $(date)";

runtime=$((end-start))
echo Runtime: $runtime

# conda deactivate

