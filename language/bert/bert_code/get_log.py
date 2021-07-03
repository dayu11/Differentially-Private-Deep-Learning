import sys
import os
import json
import numpy as np
import pandas as pd
import argparse
import csv


# python get_log.py --prefix ~/la_blob/dihe/mcbert/FT_LOG/ --folders run-fixattn-roberta,run-fixattn-k0-b0,run-fixattn-k10-b1,run-fixattn-k10-b2,run-fixattn-k10-b5,run-fixattn2-k10-b1,run-fixattn2-k10-b2,run-fixattn2-k10-b5 --output test.csv
parser = argparse.ArgumentParser()

# common args
parser.add_argument('--prefix', type=str)
parser.add_argument('--folders', type=str)
parser.add_argument('--median', action='store_true', default=False)
parser.add_argument('--output', type=str)
parser.add_argument('--miss-output', type=str)

args = parser.parse_args()

def mean(res):
    return sum(res) / len(res)
def median(res):
    return np.median(res)

def get_from_line(line, metric):
    ss = line.split('|')
    for s in ss:
        ts = s.strip().split(' ')
        if ts[0] == metric:
            return float(ts[1])
    return -1

def get_from_file(filename, keyword, metric):
    try:
        if not os.path.isfile(filename):
            return -1
        with open(filename, "r") as input:
            ret = -1
            check_cnt = 0
            for line in input.readlines():
                if keyword in line and 'valid on' in line:
                    cur_res = get_from_line(line, metric)
                    ret = max(cur_res, ret)
                if 'loaded checkpoint' in line:
                    check_cnt += 1
                if 'done training' in line:
                    check_cnt += 1
                # if 'Find missing keys when loading' in line:
                #     check_cnt = 3
            if check_cnt == 2:
                if ret <= 0:
                    print("{}: accuracy is {}".format(filename, ret))
                return ret
            else:
                print("{} is broken".format(filename))
                # os.remove(filename)
                return -1
    except Exception as e:
        return -1

def one_point(prefix, folder, name, ckp, keyword, metric, agg_fun, miss_dict):
    lrs = ['0.00001', '0.00002', '0.00003', '0.00004']
    seeds = ['1', '2', '3', '4', '5']
    best_res = 0
    best_setting = None
    valid=True
    for lr in lrs:
        total_ret = []
        for seed in seeds:
            filename = prefix + '/' + folder + '/' + ckp + '/' + name + '/' + f'{lr}-{seed}/train_log.txt'
            res = get_from_file(filename, keyword, metric)
            if res < 0:
                valid = False
                miss_dict.add(f'{name},{folder},{ckp},{lr},{seed}')
            total_ret.append(res)
        if valid:
            cur_res = agg_fun(total_ret)
            if cur_res > best_res:
                best_res = cur_res
                best_setting = lr
            
    if valid:
        print(f'{folder} {name} {ckp}, best lr is {best_setting}')
        return best_res
    else:
        print(f'{folder} {name} {ckp} is not ready')
        return -1


names =  ['MNLI-m', 'MNLI-mm', 'QNLI', 'QQP', 'SST-2', 'CoLA', 'MRPC', 'RTE', 'STS-B']
tasks = ['MNLI', 'MNLI', 'QNLI', 'QQP', 'SST-2', 'CoLA', 'MRPC', 'RTE', 'STS-B']
keywords = ["'valid'" for _ in range(len(tasks))]
keywords[1] = "'valid1'"
metrics = ['accuracy' for _ in range(len(tasks))]
metrics[5] = 'mcc'
metrics[-1] = 'pearson_spearman'
# metrics[3] = 'acc_f1'
# metrics[-3] = 'acc_f1'

ckps = ['checkpoint_1_20000.pt', 'checkpoint_2_50000.pt', 'checkpoint_4_100000.pt', 'checkpoint_7_200000.pt', 'checkpoint_10_300000.pt',
        'checkpoint_13_400000.pt', 'checkpoint_16_500000.pt', 'checkpoint_19_600000.pt', 'checkpoint_22_700000.pt', 'checkpoint_25_800000.pt', 
        'checkpoint_28_900000.pt', 'checkpoint_31_1000000.pt']
# ckps = ['checkpoint_1_20000.pt', 'checkpoint_2_50000.pt', 'checkpoint_4_100000.pt', 'checkpoint_7_200000.pt', 
#         'checkpoint_13_400000.pt', 'checkpoint_19_600000.pt',  'checkpoint_25_800000.pt', 
#         'checkpoint_31_1000000.pt']


agg_fun = mean if not args.median else median
folders = args.folders.split(',')
result_dict = {}
miss_dict = set()
for i in range(len(names)):
    result_dict[names[i]] = {}
    for folder in folders:
        result_dict[names[i]][folder] = {}
        if folder == 'roberta-old-checkpoints':
            for ckp in ckps2:
                step = int(ckp.split('.')[0].split('_')[-1]) // 1000
                res = one_point(args.prefix, folder, tasks[i], ckp, keywords[i], metrics[i], agg_fun, miss_dict)
                result_dict[names[i]][folder][step] = res
        else:
            for ckp in ckps:
                step = int(ckp.split('.')[0].split('_')[-1]) // 1000
                res = one_point(args.prefix, folder, tasks[i], ckp, keywords[i], metrics[i], agg_fun, miss_dict)
                result_dict[names[i]][folder][step] = res


with open(args.output,'w') as f:
    for name in names:
        f.write(name + '\n')
        header = ','.join(folders)
        f.write('step,' + header + '\n')
        for ckp in ckps:
            step = int(ckp.split('.')[0].split('_')[-1]) // 1000
            to_write = [str(step)]
            for exp in folders:
                if result_dict[name][exp][step] > 0:
                    to_write.append(str(result_dict[name][exp][step]))
                else:
                    to_write.append('')
            f.write(','.join(to_write) + '\n')
        f.write('\n')

    f.write('average' + '\n')
    header = ','.join(folders)
    f.write('step,' + header + '\n')
    for ckp in ckps:
        step = int(ckp.split('.')[0].split('_')[-1]) // 1000
        to_write = [str(step)]
        for exp in folders:
            sum_res = 0
            for name in names:
                if result_dict[name][exp][step] >= 0:
                    sum_res += result_dict[name][exp][step]
                else:
                    sum_res = -1
                    break
            sum_res /= len(names)
            if sum_res >= 0:
                to_write.append(str(sum_res))
            else:
                to_write.append('')
        f.write(','.join(to_write) + '\n')

    f.write('average (no RTE)' + '\n')
    header = ','.join(folders)
    f.write('step,' + header + '\n')
    for ckp in ckps:
        step = int(ckp.split('.')[0].split('_')[-1]) // 1000
        to_write = [str(step)]
        for exp in folders:
            sum_res = 0
            for name in names:
                if name == "RTE":
                    continue
                if result_dict[name][exp][step] > 0:
                    sum_res += result_dict[name][exp][step]
                else:
                    sum_res = 0
                    break
            sum_res /=  (len(names) - 1)
            if sum_res > 0:
                to_write.append(str(sum_res))
            else:
                to_write.append('')
        f.write(','.join(to_write) + '\n')

with open(args.miss_output,'w') as f:
    for x in miss_dict:
        f.write(x + '\n')

