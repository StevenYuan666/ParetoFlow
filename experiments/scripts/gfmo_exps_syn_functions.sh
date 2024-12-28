tasks=('dtlz1' 'dtlz7' 'zdt1' 'zdt2' 'zdt3' 'zdt4' 'zdt6' 'vlmop1' 'vlmop2' 'vlmop3' 'omnitest')
path="YOUR_FOLDER_PATH"

for task in "${tasks[@]}";
do
    echo "Task: $task"
    nohup YOUR_ENVIRONMENT_PATH/python3 ${path}/gfmo.py --task_name=$task --mode="train_proxies" --seed=0 > ${path}/log/task${task}_proxies_training.log 2>&1
    nohup YOUR_ENVIRONMENT_PATH/python3 ${path}/gfmo.py --task_name=$task --mode="train_flow_matching" --seed=0 > ${path}/log/task${task}_fm_training.log 2>&1
    for seed in {0..4};
    do
        nohup YOUR_ENVIRONMENT_PATH/python3 ${path}/gfmo.py --task_name=$task --mode="sampling" --seed=$seed > ${path}/log/task${task}_seed${seed}.log 2>&1
    done
done
