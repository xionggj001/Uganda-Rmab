#!/bin/bash -x

#SBATCH -c 1                # Number of cores
#SBATCH -p shared,tambe,serial_requeue
#SBATCH -t 04:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=8000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o joblogs/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joblogs/%A_%a.err  # File to which STDERR will be written, %j inserts jobid

set -x

module load python/3.10.13-fasrc01
module load intel/24.0.1-fasrc01
module load openmpi/5.0.2-fasrc01
source activate llm

data="uganda"
save_string="uganda"
N=20
B=5.0
robust_keyword="sample_random" # other option is "mid"
n_train_epochs=30 
seed=0
cdir="."
no_hawkins=1
tp_transform=None
opt_in_rate=0.9
data_type="discrete"

bash run/run_uganda.sh ${cdir} ${SLURM_ARRAY_TASK_ID} 0 ${data} ${save_string} ${N} ${B}  \
    ${robust_keyword} ${n_train_epochs} ${no_hawkins} ${tp_transform} ${opt_in_rate}