import json
import os
import distill
import re
import itertools
import pandas as pd
from evaluation_framework import *
from transformers import AutoTokenizer, AutoModelForMaskedLM
import crows_pair.metric as cp
from init_layers import init_layers_dict

def extract_stsb(model_name_or_path,task_name,learning_rate,batch_size,epochs):
    output_dir = "eval_out/" + model_name_or_path +"/" + task_name + "/" + "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_epochs"+str(epochs)
    bias_text_file = "/bias_results_stsb.txt"
    results_json_file = "/all_results.json"
    
    with open(output_dir+bias_text_file, 'r') as f:
        lines = f.readlines()
    abs_avg = float(lines[2].split()[3])
    avg = float(lines[3].split()[4].split(")")[1])
    

    with open(output_dir+results_json_file) as json_file:
        data = json.load(json_file)
    
    epoch = data["epoch"] - 2
    combined_score = data['eval_combined_score']
    
    
    return abs_avg, avg, epoch, combined_score


def get_best_model_kd_series(tps, bss, output_dir_prefix, base_model_name, task_name, num_hidden_layers, hidden_size, use_matches, init_layers, seed=""):
    best_output_dir = ""
    best_best_model_name = ""
    if task_name != "mlm":
        best_best_score = 0
    else:
        best_best_score = 9999999999
    
    init_layers_string = ""
    if init_layers:
        for layers_pair in init_layers:
            init_layers_string += "_" + "il" + str(layers_pair["layer_S"]) + "-" + str(layers_pair["layer_T"]) + layers_pair["type"]
    
    for temperature in tps:
        for batch_size in bss:
            output_dir = os.path.join(output_dir_prefix, base_model_name,str(seed),task_name, "hl"+ str(num_hidden_layers) + "_hs" +  str(hidden_size) + "_um" + str(use_matches)+ "_tp" + str(temperature) + "_bs" + str(batch_size) + init_layers_string)
            best_model_name = distill.get_best_model_name(output_dir)
            best_model_score = distill.get_best_model_score(output_dir)
            if (best_model_score > best_best_score and task_name != "mlm") or (best_model_score < best_best_score and task_name == "mlm"):
                best_best_score = best_model_score
                best_best_model_name = best_model_name
                best_output_dir = output_dir
                best_model_params = (temperature, batch_size, distill.get_best_model_epoch(output_dir))
    return best_best_score, best_best_model_name,best_output_dir, best_model_params

def get_best_model_ef_series(epochs, bss, lrs, model_name, task_name, output_dir_prefix="eval_out/", seed = ""):
    best_score = 0
    non_exist_dirs = []
    flag = True
    for learning_rate in lrs:
        for batch_size in bss:
            output_dir = os.path.join(output_dir_prefix, model_name, str(seed),task_name, "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_epochs"+str(epochs))
            if os.path.exists(output_dir):
                
                if task_name == "mnli":
                    combined_score, _ , _, epoch = get_score_and_epoch_mnli(output_dir)
                elif task_name == "stsb":
                    combined_score, epoch = get_score_and_epoch_stsb(output_dir)

                if combined_score > best_score:
                    flag = False
                    best_score = combined_score
                    best_model_dir = output_dir
                    best_model_params = (learning_rate, batch_size, epoch)
            else:
                non_exist_dirs.append(output_dir)
    if flag:
        raise ValueError(f"No model results found in the specfied dirs: {non_exist_dirs}")
    
    return best_model_dir, best_score, best_model_params
    
def get_score_and_epoch_stsb(path):
    results_json_file = os.path.join(path, "all_results.json")
    with open(results_json_file) as json_file:
        data = json.load(json_file)
    
    combined_score = float(data['eval_combined_score'])
    epoch = int(data["epoch"])
    
    return combined_score, epoch
    

def get_score_and_epoch_mnli(path):
    mnli_path = os.path.join(path, "eval-mnli_results.json")
    mnli_mm_path = os.path.join(path, "eval-mnli-mm_results.json")
    
    with open(mnli_path) as mnli_file:
        data = json.load(mnli_file)
        mnli_score = float(data["eval_accuracy"])
        if "epoch" in data:
            epoch = int(data["epoch"])
        else:
            epoch = None
        
    with open(mnli_mm_path) as mnli_mm_file:
        mnli_mm_score = float(json.load(mnli_mm_file)["eval_accuracy"])
    
    avg_score = (mnli_score + mnli_mm_score)/2
    
    return avg_score, mnli_score, mnli_mm_score, epoch

def get_bias_score_mnli_old(path, run_if_not_exists=False, base_model_name = None, num_hidden_layers = None, hidden_size = None, seed=1909):
    bias_json_file = os.path.join(path, "bias_results_mnli.json")
    if not os.path.exists(bias_json_file):
        if run_if_not_exists:
            ds_args = DatasetArguments(task_name = "mnli", max_seq_length=128)
            
            if "distilled" in path and not ("mlm" in path and ("stsb" in path or "mnli" in path)):
                model_path = os.path.join(path, "models", distill.get_best_model_name(path))
                m_args = ModelArguments(model_name_or_path=model_path, config_name = base_model_name, tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
            else:
                m_args = ModelArguments(model_name_or_path=path, tokenizer_name=base_model_name)
            t_args = TrainingArguments(output_dir=path, overwrite_output_dir = True, do_train=False,
                                       do_eval = False, do_predict = False, per_device_eval_batch_size = 32, logging_steps=1000, seed=seed) 

            e = Evaluation(ds_args, m_args, t_args, bias_eval = True, weat_eval=None)
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            e.load_model_tokenizer()
            e.train()
        else:
            raise ValueError("No Bias File in path. Run Bias Eval or set run_if_not_exists = True")
    
    with open(bias_json_file, "r") as f:
        data = json.load(f)
    return float(data["net_neutral"]), float(data["fraction_neutral"])
        

def get_bias_score_mnli(path, run_if_not_exists=False, base_model_name = None, num_hidden_layers = None, hidden_size = None, seed=1909):
    nli_bias_output_dir = os.path.join(path, "nli_bias")
    if not os.path.exists(nli_bias_output_dir):
        if run_if_not_exists:
            ds_args = DatasetArguments(task_name = "mnli", max_seq_length=128)
            
            if "distilled" in path and not ("mlm" in path and ("stsb" in path or "mnli" in path)):
                model_path = os.path.join(path, "models", distill.get_best_model_name(path))
                m_args = ModelArguments(model_name_or_path=model_path, config_name = base_model_name, tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
            else:
                m_args = ModelArguments(model_name_or_path=path, tokenizer_name=base_model_name)
            t_args = TrainingArguments(output_dir=path, overwrite_output_dir = True, do_train=False,
                                       do_eval = False, do_predict = False, per_device_eval_batch_size = 32, logging_steps=1000, seed=seed) 

            e = Evaluation(ds_args, m_args, t_args, bias_eval = True, weat_eval=None)
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            e.load_model_tokenizer()
            e.train()
        else:
            raise ValueError("No Bias File in path. Run Bias Eval or set run_if_not_exists = True")
    racial_bias_file = os.path.join(nli_bias_output_dir, "racial_bias_results.json")
    gender_bias_file = os.path.join(nli_bias_output_dir, "gender_bias_results.json")
    with open(gender_bias_file, "r") as f:
        gender_data = json.load(f)
    with open(racial_bias_file, "r") as f:
        racial_data = json.load(f)
        
    return float(gender_data["net_neutral"]), float(gender_data["fraction_neutral"]), float(racial_data["net_neutral"]), float(racial_data["fraction_neutral"])

def get_bias_score_stsb(path, run_if_not_exists=False, base_model_name = None, num_hidden_layers = None, hidden_size = None,seed=1909):
    bias_text_file = os.path.join(path,"bias_results_stsb.txt")
    if not os.path.exists(bias_text_file):
        if run_if_not_exists:
            ds_args = DatasetArguments(task_name = "stsb", max_seq_length=128)
            
            if "distilled" in path and not ("mlm" in path and ("stsb" in path or "mnli" in path)):
                model_path = os.path.join(path, "models", distill.get_best_model_name(path))
                m_args = ModelArguments(model_name_or_path=model_path, config_name = base_model_name, tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
            else:
                m_args = ModelArguments(model_name_or_path=path, tokenizer_name=base_model_name)
            t_args = TrainingArguments(output_dir=path, overwrite_output_dir = True, do_train=False,
                                       do_eval = False, do_predict = False, per_device_eval_batch_size = 32, logging_steps=1000, seed=seed) 

            e = Evaluation(ds_args, m_args, t_args, bias_eval = True, weat_eval=None)
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            e.load_model_tokenizer()
            e.train()
        else:
            raise ValueError("No Bias File in path. Run Bias Eval or set run_if_not_exists = True")
    with open(bias_text_file, 'r') as f:
        lines = f.readlines()
    abs_avg = float(lines[2].split()[3])
    avg = float(lines[3].split()[4].split(")")[1])
    
    return abs_avg, avg

def get_weat_score(path, run_if_not_exists=False, base_model_name = None, num_hidden_layers = 12, hidden_size = None,seed=1909):
    
    weat_dir = os.path.join(path, "weat")
    if not os.path.exists(weat_dir):
        if run_if_not_exists:
            if "mnli" in path:
                task_name = "mnli"
            else:
                task_name ="stsb"
            ds_args = DatasetArguments(task_name = task_name, max_seq_length=128)
            
            if "distilled" in path and not ("mlm" in path and ("stsb" in path or "mnli" in path)):
                model_path = os.path.join(path, "models", distill.get_best_model_name(path))
                m_args = ModelArguments(model_name_or_path=model_path, config_name = base_model_name, tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
            else:
                m_args = ModelArguments(model_name_or_path=path, tokenizer_name=base_model_name)
            t_args = TrainingArguments(output_dir=path, overwrite_output_dir = True, do_train=False,
                                       do_eval = False, do_predict = False, per_device_eval_batch_size = 32, logging_steps=1000, seed=seed) 

            e = Evaluation(ds_args, m_args, t_args, bias_eval = False, weat_eval=list(range(3,11)))
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            e.load_model_tokenizer()
            e.train()
        else:
            raise ValueError("No Bias File in path. Run Bias Eval or set run_if_not_exists = True")
    all_tests = []
    for test_number in range(3, 11):
        file_name = os.path.join(weat_dir, f"weat{test_number}_n0_m{num_hidden_layers}_result.json")
        with open(file_name, "r") as f:
            data = json.load(f)
        all_tests.append((float(data["effect_size"]), float(data["p"])))
        
    return all_tests


def get_seat_score(path, run_if_not_exists=False, base_model_name = None, num_hidden_layers = 12, hidden_size = None,seed=1909):
    
    seat_dir = os.path.join(path, "seat")
    if not os.path.exists(seat_dir):
        if run_if_not_exists:
            if "mnli" in path:
                task_name = "mnli"
            else:
                task_name ="stsb"
            ds_args = DatasetArguments(task_name = task_name, max_seq_length=128)
            
            if "distilled" in path and not ("mlm" in path and ("stsb" in path or "mnli" in path)):
                model_path = os.path.join(path, "models", distill.get_best_model_name(path))
                m_args = ModelArguments(model_name_or_path=model_path, config_name = base_model_name, tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
            else:
                m_args = ModelArguments(model_name_or_path=path, tokenizer_name=base_model_name)
            t_args = TrainingArguments(output_dir=path, overwrite_output_dir = True, do_train=False,
                                       do_eval = False, do_predict = False, per_device_eval_batch_size = 32, logging_steps=1000, seed=seed) 

            e = Evaluation(ds_args, m_args, t_args, bias_eval = False, weat_eval=None, seat_eval="all_sent")
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            e.load_model_tokenizer()
            e.train()
        else:
            raise ValueError("No Bias File in path. Run Bias Eval or set run_if_not_exists = True")
    seat_file = os.path.join(seat_dir, "seat_results.tsv")
    df = pd.read_csv(seat_file, sep='\t')
    return df

def get_crows_score(path, run_if_not_exists=True, base_model_name=None,output_dir_prefix="eval_out/distilled/", num_hidden_layers = None, hidden_size=768):
    

    crows_file = os.path.join(path, "crows", "crows_score.json")
        
    if True:#not os.path.exists(crows_file):
        if run_if_not_exists:

            init_layers=init_layers_dict[str(num_hidden_layers)]

            best_model_name = distill.get_best_model_name(path)
            student_model_path = os.path.join(path,"models",best_model_name)
            d = distill.MLMDistillation(num_samples=500, base_model_name = base_model_name, output_dir_prefix=output_dir_prefix)
            d.load_model_tokenizer(base_model_name, num_hidden_layers, hidden_size,init_layers, student_model_path)
            d.crows_evaluation(path)
        else:
            raise ValueError("No crows File in path. Run crows eval or set run_if_not_exists = True")
    with open(crows_file, "r") as f:
        data = json.load(f)
        
    return float(data["metric_score"]), float(data["gender_metric_score"]),float(data["race_metric_score"]),float(data["stereo_score"]), float(data["anti_stereo_score"])
        
def get_crows_score_base_model(base_model_name, output_dir_prefix="eval_out/"):
    
    crows_dir = os.path.join(output_dir_prefix, base_model_name,"crows")
    crows_file = os.path.join(crows_dir, "crows_score.json")
    if True:# not os.path.exists(crows_file):
        model = AutoModelForMaskedLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        uncased = ("uncased" in base_model_name) 
        cp.evaluate(model, uncased, tokenizer, crows_dir)
    
    with open(crows_file, "r") as f:
        data = json.load(f)
        
    return float(data["metric_score"]), float(data["gender_metric_score"]),float(data["race_metric_score"]),float(data["stereo_score"]), float(data["anti_stereo_score"])
        
    
    
    
def eval_series_kd_mlm(base_model_name, num_hidden_layers, hidden_size=768, tps=[4,8], bss=[64,128],use_matches=False,output_dir_prefix="eval_out/distilled/", init_layers=None):
    task_name = "mlm"
    best_score, best_model_name, best_model_dir, best_model_params = get_best_model_kd_series(tps, bss, output_dir_prefix, base_model_name, task_name, num_hidden_layers, hidden_size, use_matches, init_layers=init_layers)
    
    
    weat_results = get_weat_score(best_model_dir, run_if_not_exists=True, base_model_name=base_model_name, num_hidden_layers = num_hidden_layers)
    seat_results_df = get_seat_score(best_model_dir, run_if_not_exists=True, base_model_name=base_model_name, num_hidden_layers = num_hidden_layers)
    crows_results = get_crows_score(best_model_dir, run_if_not_exists=True, base_model_name=base_model_name,output_dir_prefix=output_dir_prefix, num_hidden_layers = num_hidden_layers)
    
    
    
    weat_columns = []
    for i in range (3,11):
        weat_columns.append(f"weat_{i}_effect_size")
        weat_columns.append(f"weat_{i}_p")
    
    seat_columns = []
    seat_results = []
    seat_es = seat_results_df["effect_size"].tolist()
    seat_p = seat_results_df["p_value"].tolist()
    for i, test in enumerate(seat_results_df["test"].tolist()):
        seat_columns.append(test+"_effect_size")
        seat_columns.append(test+"_p")
        seat_results.append((seat_es[i],seat_p[i]))
    
    columns = ["crows_score","crows_score_gender","crows_score_race"] + weat_columns + seat_columns

    df = pd.DataFrame(columns=columns)
    data = [crows_results[0],crows_results[1],crows_results[2]] + list(itertools.chain(*weat_results)) + list(itertools.chain(*seat_results))
    df.loc[len(df)]=data
    
    init_layers_student = []
    if init_layers:
        for pair in init_layers:
            init_layers_student.append(pair["layer_S"])
        
    keys = ["base_model_name","temperature", "batch_size", "epoch", "best_score", "num_hidden_layers","hidden_size","init_layers"]
    values = [base_model_name] +  list(best_model_params) + [best_score, num_hidden_layers, hidden_size, str(init_layers_student)]
    for key, value in zip(keys,values):
        df[key] = value
        
    return df 

def eval_mlm(base_model_name, output_dir):
    crows_results = get_crows_score_base_model(base_model_name)
    seat_results_df = get_seat_score(output_dir)
    weat_results = get_weat_score(output_dir)
    
    weat_columns = []
    for i in range (3,11):
        weat_columns.append(f"weat_{i}_effect_size")
        weat_columns.append(f"weat_{i}_p")
    
    seat_columns = []
    seat_results = []
    seat_es = seat_results_df["effect_size"].tolist()
    seat_p = seat_results_df["p_value"].tolist()
    for i, test in enumerate(seat_results_df["test"].tolist()):
        seat_columns.append(test+"_effect_size")
        seat_columns.append(test+"_p")
        seat_results.append((seat_es[i],seat_p[i]))
    
    columns = ["crows_score","crows_score_gender","crows_score_race"] + weat_columns + seat_columns

    df = pd.DataFrame(columns=columns)

    data = [crows_results[0],crows_results[1],crows_results[2]] + list(itertools.chain(*weat_results)) + list(itertools.chain(*seat_results))
    df.loc[len(df)]=data
    
    return df

    
def eval_series_ef_mlm_kd(task_name, model_name, num_hidden_layers, output_dir_prefix, mlm_perplexity, kd_parameters, bss=[32,16], lrs =[5e-5, 3e-5,2e-5], epochs=10,seed=""):
    output_dir = os.path.join(output_dir_prefix, str(seed))
    df = eval_series_ef(task_name, model_name, "", bss, lrs, epochs, output_dir, True, num_hidden_layers)

    crows_score = get_crows_score(path=output_dir_prefix, run_if_not_exists=True, base_model_name=model_name, output_dir_prefix="eval_out/distilled/", num_hidden_layers = num_hidden_layers, hidden_size=768)
    
    df.at[0,"mlm_perplexity"] = mlm_perplexity
    df.at[0,"kd_temperature"] = kd_parameters[0]
    df.at[0,"kd_batch_size"] = kd_parameters[1]
    df.at[0,"kd_epoch"] = kd_parameters[2]
    df.at[0,"num_hidden_layers"] = num_hidden_layers
    df.at[0,"crows_score"] = crows_score[0]
    if seed:
        df.at[0,"seed"] = seed
    
    return df
    
def eval_series_ef(task_name, model_name, seed="", bss=[32,16], lrs =[5e-5, 3e-5,2e-5], epochs=10, output_dir_prefix="eval_out/", mlm_finetuned=False, num_hidden_layers = None):
    
    if not mlm_finetuned:
        model_name_get_best = model_name
    else:
        model_name_get_best = ""
    best_model_dir, best_score, best_model_params = get_best_model_ef_series(epochs, bss, lrs, model_name_get_best, task_name, output_dir_prefix, seed=seed)

    if task_name == "mnli":
        bias_result = get_bias_score_mnli(best_model_dir, run_if_not_exists=True, base_model_name = model_name)
    elif task_name == "stsb":
        abs_avg, avg = get_bias_score_stsb(best_model_dir, run_if_not_exists=True, base_model_name = model_name)
        bias_result = abs_avg, avg
    if not num_hidden_layers:
        if "small" in model_name:
            num_hidden_layers = 6
        else:
            num_hidden_layers = 12
        
    weat_results = get_weat_score(best_model_dir, run_if_not_exists=True, base_model_name = model_name, num_hidden_layers=num_hidden_layers)
    
    seat_results_df = get_seat_score(best_model_dir, run_if_not_exists=True, base_model_name = model_name, num_hidden_layers=num_hidden_layers)
    
    crows_results = get_crows_score_base_model(model_name)
    
    if task_name == "mnli":
        bias_columns = ["gender_net_neutral", "gender_fraction_neutral","racial_net_neutral", "racial_fraction_neutral"]
    elif task_name == "stsb":
        bias_columns = ["abs_avg", "avg"]
    weat_columns = []
    for i in range (3,11):
        weat_columns.append(f"weat_{i}_effect_size")
        weat_columns.append(f"weat_{i}_p")
    
    
    seat_columns = []
    seat_results = []
    seat_es = seat_results_df["effect_size"].tolist()
    seat_p = seat_results_df["p_value"].tolist()
    for i, test in enumerate(seat_results_df["test"].tolist()):
        seat_columns.append(test+"_effect_size")
        seat_columns.append(test+"_p")
        seat_results.append((seat_es[i],seat_p[i]))
    
    columns = ["model_name", "seed","learning_rate", "batch_size", "epoch", "best_score", "crows_score"] + bias_columns + weat_columns + seat_columns

    df = pd.DataFrame(columns=columns)

    data = [model_name, seed] +  list(best_model_params) + [best_score, crows_results[0]] + list(bias_result) + list(itertools.chain(*weat_results)) + list(itertools.chain(*seat_results))
    df.loc[len(df)]=data
    
    return df

def eval_series_kd(base_model_name, seed, num_hidden_layers, hidden_size,task_name="mnli", tps=[4,8], bss=[64,128],use_matches=True,output_dir_prefix="eval_out/distilled/", init_layers=None):
    
    
    best_score, best_model_name, best_model_dir, best_model_params = get_best_model_kd_series(tps, bss, output_dir_prefix, base_model_name, task_name, num_hidden_layers, hidden_size, use_matches, init_layers=init_layers, seed=seed)
    
    if task_name == "mnli":
        bias_result = get_bias_score_mnli(best_model_dir, run_if_not_exists=True, base_model_name = base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, seed=seed)
    elif task_name == "stsb":
        abs_avg, avg = get_bias_score_stsb(best_model_dir, run_if_not_exists=True, base_model_name = base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, seed=seed)
        bias_result = abs_avg, avg

    weat_results = get_weat_score(best_model_dir, run_if_not_exists=True, base_model_name = base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size,seed=seed)
    
    seat_results_df = get_seat_score(best_model_dir, run_if_not_exists=True, base_model_name = base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size,seed=seed)
        
    if task_name == "mnli":
        bias_columns = ["gender_net_neutral", "gender_fraction_neutral","racial_net_neutral", "racial_fraction_neutral"]
    elif task_name == "stsb":
        bias_columns = ["abs_avg", "avg"]
    weat_columns = []
    for i in range (3,11):
        weat_columns.append(f"weat_{i}_effect_size")
        weat_columns.append(f"weat_{i}_p")
    
    seat_columns = []
    seat_results = []
    seat_es = seat_results_df["effect_size"].tolist()
    seat_p = seat_results_df["p_value"].tolist()
    for i, test in enumerate(seat_results_df["test"].tolist()):
        seat_columns.append(test+"_effect_size")
        seat_columns.append(test+"_p")
        seat_results.append((seat_es[i],seat_p[i]))
    
    columns = ["base_model_name", "seed", "num_hidden_layers","hidden_size","init_layers","temperature", "batch_size", "epoch", "best_score"] + bias_columns + weat_columns + seat_columns

    df = pd.DataFrame(columns=columns)
    
    init_layers_student = []
    if init_layers:
        for pair in init_layers:
            init_layers_student.append(pair["layer_S"])
        
    
    data = [base_model_name, seed, num_hidden_layers, hidden_size, str(init_layers_student)] +  list(best_model_params) + [best_score] + list(bias_result) + list(itertools.chain(*weat_results)) + list(itertools.chain(*seat_results))
    df.loc[len(df)]=data
    
    return df
    
    