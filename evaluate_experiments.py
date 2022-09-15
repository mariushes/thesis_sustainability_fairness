import result_extraction as re
import json
import pandas as pd
import os
import distill
from init_layers import init_layers_dict



model_name = "bert-base-uncased"
task_name = "mnli"
seeds = [43154364]
dfs = []
for seed in seeds:
    df = re.eval_series_ef(task_name, model_name, seed=seed, bss=[32,16], lrs=[5e-5, 3e-5,2e-5])
    dfs.append(df)
    #display(df)
df = pd.concat(dfs)
display(df)
df.to_excel("results/bert_mnli_teacher_results.xlsx")


model_name = "bert-base-uncased"
task_name = "stsb"
seeds = [43154364]
dfs = []
for seed in seeds:
    df = re.eval_series_ef(task_name, model_name, seed=seed, bss=[32,16], lrs=[5e-5, 3e-5,2e-5])
    dfs.append(df)
    #display(df)
df = pd.concat(dfs)
display(df)
df.to_excel("results/bert_stsb_teacher_results.xlsx", index=False)


task_name = "mnli"
model_name = "bert-base-uncased"
nhls = [4]
hss = [96,192,384,576,768]
dfs = []
seeds = [43154364]
for seed in seeds:
    for num_hidden_layers in nhls:
        for hidden_size in hss:
            df = re.eval_series_kd(model_name, seed, num_hidden_layers, hidden_size, task_name=task_name)
            #display(df)
            dfs.append(df)
df = pd.concat(dfs)
display(df)
df.to_excel("results/mnli_kd_hl4_hs_results.xlsx", index=False)


task_name = "mnli"
model_name = "bert-base-uncased"
nhls = nhls = list(range(1,9)) + [10,12]
hss = [768]
seeds = [43154364]
dfs = []
for seed in seeds:
    for num_hidden_layers in nhls:
        for hidden_size in hss:
            df = re.eval_series_kd(model_name, seed, num_hidden_layers, hidden_size, task_name=task_name)
            #display(df)
            dfs.append(df)
df = pd.concat(dfs)
display(df)
df.to_excel("results/mnli_kd_hl_results.xlsx", index=False)


init_layers_list =  [
    [{"layer_S": 0, "layer_T": 0, "type": "all"}],
    [{"layer_S": 1, "layer_T": 3, "type": "all"}],
    [{"layer_S": 2, "layer_T": 6, "type": "all"}],
    [{"layer_S": 3, "layer_T": 9, "type": "all"}],
    [{"layer_S": 4, "layer_T": 12, "type": "all"}],
    [{"layer_S": 0, "layer_T": 0, "type": "all"}, {"layer_S": 1, "layer_T": 3, "type": "all"}, {"layer_S": 2, "layer_T": 6, "type": "all"}, {"layer_S": 3, "layer_T": 9, "type": "all"}, {"layer_S": 4, "layer_T": 12, "type": "all"}]]

task_name = "mnli"
model_name = "bert-base-uncased"
num_hidden_layers = 4
hidden_size = 768
seeds = [43154364]
dfs = []
for seed in seeds:
    for init_layers in init_layers_list:

        df = re.eval_series_kd(model_name, seed, num_hidden_layers, hidden_size, task_name=task_name, use_matches=False, init_layers=init_layers)
            #display(df)
        dfs.append(df)
df = pd.concat(dfs)
display(df)
df.to_excel("results/mnli_kd_init_results.xlsx", index=False)


task_name = "mnli"
model_name = "bert-base-uncased"
nhls = range(1,7)
hss = [768]
seeds = [43154364]
dfs = []
for seed in seeds:
    for num_hidden_layers in nhls:
        for hidden_size in hss:
            df = re.eval_series_kd(model_name, seed, num_hidden_layers, hidden_size, task_name=task_name, use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)])
            #display(df)
            dfs.append(df)
df = pd.concat(dfs)
display(df)
df.to_excel("results/mnli_kd_hl_all_init_results.xlsx", index=False)

task_name = "stsb"
model_name = "bert-base-uncased"
nhls = list(range(1,11))
hss = [768]
seeds = [43154364]
dfs = []
for seed in seeds:
    for num_hidden_layers in nhls:
        for hidden_size in hss:
            df = re.eval_series_kd(model_name, seed, num_hidden_layers, hidden_size, task_name=task_name,use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)])
            #display(df)
            dfs.append(df)
df = pd.concat(dfs)
display(df)
df.to_excel("results/stsb_kd_hl_all_init_results.xlsx", index=False)
