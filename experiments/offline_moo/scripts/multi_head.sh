#!/bin/bash

# Synthetic Functions 
# "dtlz1 dtlz2 dtlz3 dtlz4 dtlz5 dtlz6 dtlz7 zdt1 zdt2 zdt3 zdt4 zdt6 vlmop1 vlmop2 vlmop3 omnitest"

# RE
# "re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61"

# MO-NAS
# "nb201_test c10mop1 c10mop2 c10mop3 c10mop4 c10mop5 c10mop6 c10mop7 c10mop8 c10mop9 in1kmop1 in1kmop2 in1kmop3 in1kmop4 in1kmop5 in1kmop6 in1kmop7 in1kmop8 in1kmop9"

# MORL
# "mo_hopper_v2 mo_swimmer_v2"

# MOCO 
# "bi_tsp_20 bi_tsp_50 bi_tsp_100 bi_tsp_500 tri_tsp_20 tri_tsp_50 tri_tsp_100 bi_cvrp_20 bi_cvrp_50 bi_cvrp_100 bi_kp_50 bi_kp_100 bi_kp_200"

# Scientific Design
# "zinc regex rfp molecule"

seeds="1000"
# tasks="zinc regex rfp molecule re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61 dtlz1 dtlz2 dtlz3 dtlz4 dtlz5 dtlz6 dtlz7 zdt1 zdt2 zdt3 zdt4 zdt6 vlmop1 vlmop2 vlmop3 omnitest nb201_test c10mop1 c10mop2 c10mop3 c10mop4 c10mop5 c10mop6 c10mop7 c10mop8 c10mop9 in1kmop1 in1kmop2 in1kmop3 in1kmop4 in1kmop5 in1kmop6 in1kmop7 in1kmop8 in1kmop9 bi_tsp_20 bi_tsp_50 bi_tsp_100 bi_tsp_500 tri_tsp_20 tri_tsp_50 tri_tsp_100 bi_cvrp_20 bi_cvrp_50 bi_cvrp_100 bi_kp_50 bi_kp_100 bi_kp_200 mo_hopper_v2 mo_swimmer_v2"
tasks="zdt1"
model="MultiHead"
train_modes="Vallina GradNorm PcGrad"
# "Vallina GradNorm PcGrad"

MAX_JOBS=16
AVAILABLE_GPUS="0 1"
MAX_RETRIES=1

get_gpu_allocation() {
    local job_number=$1
    local gpus=($AVAILABLE_GPUS)
    local num_gpus=${#gpus[@]}
    local gpu_id=$((job_number % num_gpus))
    echo ${gpus[gpu_id]}
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    echo $gpu_allocation
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=$gpu_allocation python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

for seed in $seeds; do 
    for task in $tasks; do 
        for train_mode in $train_modes; do

        check_jobs
        gpu_allocation=$(get_gpu_allocation $job_number)
        ((job_number++))
        run_with_retry "off_moo_baselines/multi_head/experiment.py \
            --model=${model} \
            --train_mode=${train_mode} \
            --task=${task} \
            --use_wandb=False \
            --retrain_model=False \
            --seed=${seed}" \
            "$gpu_allocation" & 

        done
    done 
done 

wait