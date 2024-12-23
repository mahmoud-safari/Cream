#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH -p cpuonly
# SBATCH still got io issue. do I need to set num_workers?

# start=`date +%s`

export DATA_PATH="/hkfs/work/workspace/scratch/fr_ms2108-data/data"

wget -c --secure-protocol=TLSv1 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz -P $DATA_PATH
wget -c --secure-protocol=TLSv1 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -P $DATA_PATH
wget -c --secure-protocol=TLSv1 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -P $DATA_PATH
 

# mkdir imagenet

# mkdir imagenet/train && cd imagenet


# end=`date +%s`

# echo "DONE";
# echo "Finished on $(date)";

# runtime=$((end-start))
# echo Runtime: $runtime