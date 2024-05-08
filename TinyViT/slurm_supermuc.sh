#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=pn68xi
# SBATCH --array=1-3 # You can change this
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 # You can change this
#SBATCH --export=NONE
#SBATCH -J job_name
#SBATCH -o ./logs/%x.%N.%j.out # STDOUT
#SBATCH -e ./logs/%x.%N.%j.err # STDERR
#SBATCH -D ./
#SBATCH --mail-type=END
# SBATCH --mail-user=<enter_your_email>
#SBATCH --time=0-03:00:00
#SBATCH --ear=off

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

module load intel-toolkit/2024.0.0
module load intel-vtune
module load slurm_setup
export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_IPC=1
export I_MPI_OFFLOAD_RDMA=1
export I_MPI_OFFLOAD_L0_D2D_ENGINE_TYPE=1
export I_MPI_DEBUG=6
export I_MPI_OFFLOAD_CELL=device

source ~/.conda_init
conda activate tinyvit

export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_AFFINITY_MASK=0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1  ## for all 4 GPUs / 8 tiles
export CCL_ZE_IPC_EXCHANGE=sockets

export NP=${SLURM_NTASKS}
export NNODES=${SLURM_NNODES}
export PPN=${SLURM_NTASKS_PER_NODE:-$(( NP / NNODES ))}
export J=${SLURM_JOB_ID}
echo "NP =" $NP " PPN =" $PPN

mpirun -n $NP -ppn $PPN -l python main_supermuc.py

end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime