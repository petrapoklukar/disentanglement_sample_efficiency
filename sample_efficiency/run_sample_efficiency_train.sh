#!/usr/bin/env bash

SOURCE_PATH="${HOME}/Workspace/disentanglement_sample_efficiency/sample_efficiency"
AT="@"

# Test the job before actually submitting
SBATCH_OR_CAT=cat
#SBATCH_OR_CAT=sbatch

declare -a modelArr=(
		"bvae"
		"fvae"
		"btcvae"
		"annvae"
		)

declare -a datasetArr=(
		"3dshapes_model_s1000"
		)
		
declare -a seedArr=(
		1602
		)

for seed in "${seedArr[@]}"
do

for dataset in "${datasetArr[@]}"
do

for model in "${modelArr[@]}"
do


RUNS_PATH="${SOURCE_PATH}/3dshapes_models/${model}${dataset}_${seed}"
echo $RUNS_PATH
mkdir -p $RUNS_PATH

"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm.err"
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
        --model=$model \
        --dataset=$dataset \
        --rng=$seed \

HERE
done
done
done
 