tasks=('nb201_test' 'c10mop1' 'c10mop2' 'c10mop3' 'c10mop4' 'c10mop5' 'c10mop6' 'c10mop7' 'c10mop8' 'c10mop9' 'in1kmop1' 'in1kmop2' 'in1kmop3' 'in1kmop4' 'in1kmop5' 'in1kmop6' 'in1kmop7' 'in1kmop8' 'in1kmop9')
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
