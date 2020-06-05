#!/usr/bin/env bash

SOURCE_PATH="${HOME}/Workspace/disentanglement_sample_efficiency/slurm_logs"
AT="@"

# Test the job before actually submitting
#SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch

declare -a modelArr=(
		"vae"
		)

for model in "${modelArr[@]}"
do


"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${SOURCE_PATH}/%J_slurm.out"
#SBATCH --error="${SOURCE_PATH}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="poklukar${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB

echo "Sourcing conda.sh"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate base
nvidia-smi

python sample_efficiency_train.py \
        --model=$model

HERE
done
 