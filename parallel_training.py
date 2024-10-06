import os
import subprocess
from shutil import copyfile
import time
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type_exp', type=str, required=True)
args = parser.parse_args()

type_exp = args.type_exp

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Available GPUs: {len(gpus)}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU available, exiting...")
    exit(1)

epochs = 500
patience = 60
batch_size = 983

original_file = f'f0s0d0_original_{type_exp}.py'

if type_exp == 'PBM':
    tasks = [(fold, seed) for fold in range(3) for seed in range(3)]
else:
    tasks = [(0, 0), (2, 2)]

def create_and_run_script(fold, seed, gpu_id):
    new_file = f'f{fold}s{seed}d{gpu_id}_{type_exp}.py'

    copyfile(original_file, new_file)

    with open(new_file, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('f = 0', f'f = {fold}')
    filedata = filedata.replace('s = 0', f's = {seed}')
    filedata = filedata.replace('device = 0', f'device = {gpu_id}')

    with open(new_file, 'w') as file:
        file.write(filedata)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Launching {new_file} on GPU {gpu_id}")
    process = subprocess.Popen(['python3', new_file], env=env)
    return process


def manage_tasks():
    jobs = []

    task_index = 0
    num_tasks = len(tasks)
    num_gpus = len(gpus)

    for gpu_id in range(min(num_gpus, num_tasks)):
        fold, seed = tasks[task_index]
        process = create_and_run_script(fold, seed, gpu_id)
        jobs.append((process, gpu_id))
        task_index += 1

    while task_index < num_tasks:
        for i, (job, gpu_id) in enumerate(jobs):
            if job.poll() is not None:  
                fold, seed = tasks[task_index]  
                print(f"GPU {gpu_id} is free, launching new task fold={fold}, seed={seed}")
                process = create_and_run_script(fold, seed, gpu_id)
                jobs[i] = (process, gpu_id)  
                task_index += 1
                if task_index >= num_tasks:
                    break 

        time.sleep(5)

    for job, gpu_id in jobs:
        job.wait()

    print("All jobs completed.")

manage_tasks()

#delete scripts
for fold in range(3):
    for seed in range(3):
        for gpu_id in range(len(gpus)):
            temp_file = f'f{fold}s{seed}d{gpu_id}_{type_exp}.py'
            if os.path.exists(temp_file):
                os.remove(temp_file)
