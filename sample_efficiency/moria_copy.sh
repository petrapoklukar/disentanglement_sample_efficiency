#!/usr/bin/env bash

declare -a modelArr=(
	"b8vae"
	# "vae"
	# "bvae"
	# "fvae"
	# "btcvae"
	# "annvae"
)

declare -a datasetArr=(
	"3dshapes_model_s100"
	# "3dshapes_model_s1000"
	"3dshapes_model_s10000"
	# "3dshapes_model_s50000"
	# "3dshapes_model_s100000"
	# "3dshapes_model_s150000"
	# "3dshapes_model_s250000"
	)
		
declare -a seedArr=(1602 1201 1012 2805)

for seed in "${seedArr[@]}"
do

for dataset in "${datasetArr[@]}"
do

for model in "${modelArr[@]}"
do
	echo ${model}
	echo ${dataset}
	echo ${seed}
	rsync -a --ignore-existing -avz -e "ssh -p 2222" ppoklukar@moria.csc.kth.se:Workspace/disentanglement_sample_efficiency/sample_efficiency/3dshapes_models/${model}${dataset}_${seed}/metrics 3dshapes_models/${model}${dataset}_${seed}/
done
done
done