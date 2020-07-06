#!/usr/bin/env bash

SOURCE_PATH="${HOME}/Workspace/disentanglement_sample_efficiency/sample_efficiency"
AT="@"

# Test the job before actually submitting
# SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch

RUNS_PATH="${SOURCE_PATH}/backbone/pca"
echo $RUNS_PATH

"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm_recall.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm_recall.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="poklukar${AT}kth.se"
#SBATCH --constrain="belegost|rivendell|shire"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150GB

echo "Sourcing conda.sh"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate base
nvidia-smi

python sample_efficiency_train_pca.py --dataset=3dshapes_model_all
HERE