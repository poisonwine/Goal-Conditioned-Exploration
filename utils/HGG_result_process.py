import os
import numpy as np
import argparse
import csv

env_name="FetchPickAndPlace-v2"
def process_epoch_data(path, cycle,id):
    data_path = os.path.join(path, "vanilla.npz")
    data = np.load(data_path, allow_pickle=True)
    results = np.array(data['info'])
    inds = np.arange(0, results.shape[0]+1, cycle)
    epoch_data = []
    for i in range(inds.shape[0]-1):
        success_rate = np.mean(results[inds[i]:inds[i+1], 1])
        epoch_data.append([i, success_rate])
    # print(epoch_data)
    save_path = os.path.join("results",env_name, "HGG")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "progress_" +str(id) +'.csv'),'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "test/success_rate"])
        writer.writerows(epoch_data)

path=os.path.join('./HGG_result', env_name)
file =["ddpg-FetchPickAndPlace-v2-hgg-7"]
#file = os.listdir(path)
for name in file:
    id = name.split('-')[-1]
    data_path = os.path.join(path, name)
    process_epoch_data(data_path, 2, id)