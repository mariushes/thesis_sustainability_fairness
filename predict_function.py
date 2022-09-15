import numpy as np
import os
import torch
from torch.utils.data import SequentialSampler,DistributedSampler,DataLoader
from utils_glue import compute_metrics
from tqdm import tqdm
import logging
from collections import defaultdict
#logger = logging.getLogger(__name__)
import datasets
from transformers import PretrainedConfig
from textbrewer.distiller_utils import move_to_device
import json
from statistics import mean

def predict(model,eval_datasets,step,epoch,logger,output_dir,task_name,local_rank,predict_batch_size,device, do_train_eval=False, train_dataset=None, data_collator=None):
    eval_task_names = [task_name]
    if task_name == "mnli": eval_task_names.append("mnli-mm")
    if do_train_eval: eval_task_names.append("train")
    if do_train_eval: eval_datasets.append(train_dataset)
    eval_output_dir = output_dir
    task_results = {}
    for eval_task,eval_dataset in zip(eval_task_names, eval_datasets):
        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("***** Running predictions *****")
        logger.info(f"task name =  {str(eval_task)}, Num  examples = {str(len(eval_dataset))}, Step = {str(step)}")
        eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=predict_batch_size, collate_fn=data_collator)
        model.eval()
        if task_name in ["stsb", "mnli"]:
            metric = datasets.load_metric("glue",task_name)
        elif task_name == "mlm":
            metric = []
        else:
            raise ValueError("No metric found for task name.")
            
        for batch in eval_dataloader:
            labels = batch["labels"]
            batch = move_to_device(batch, device)
            with torch.no_grad():
                model_outputs = model(**batch)
                if task_name != "mlm":
                    model_predictions = model_outputs["logits"]
                    if task_name !="stsb":
                        model_predictions = np.array(model_predictions.detach().cpu(),dtype=np.float32)
                        model_predictions = np.argmax(model_predictions, axis=1)
                        
                    metric.add_batch(predictions=model_predictions, references=labels)
                    
                else:
                    loss = model_outputs["loss"]
                    perplexity = float(torch.exp(loss).cpu().detach().numpy())
                    metric.append(perplexity)
        
        if task_name != "mlm":
            results = metric.compute()
        else:
            results = {"perplexity": mean(metric)}
        
                    
                    
        for key in sorted(results.keys()):
            logger.info(f"Eval results: {eval_task} {key} = {results[key]:.5f}")

        task_results[eval_task] = results
    
    
    write_results_json(eval_output_dir,step,task_results,eval_task_names)
    write_results(eval_output_dir,step,task_results,eval_task_names)
    model.train()
    
    stop_training = False
    
    return task_results, stop_training


def predict_and_early_stopping(model,eval_datasets,step,epoch,logger,output_dir,task_name,local_rank,
                               predict_batch_size,device, early_stopping_patience=2, do_train_eval=False, train_dataset=None, data_collator=None):
    
    task_results, _ = predict(model,eval_datasets,step,epoch,logger,output_dir,task_name,local_rank,
                              predict_batch_size,device, do_train_eval=do_train_eval, train_dataset=train_dataset, data_collator=data_collator)
    
    current_epoch = epoch + 1
    
    stop_training = False
    
    output_eval_file = os.path.join(output_dir, "eval_results.json")
    with open(output_eval_file) as json_file:
        result_json = json.load(json_file)

    keys = []
    for key in result_json.keys():
        if key == "best_result":
            continue
        keys.append(int(key))
    keys.sort()
    
    steps_per_epoch = keys[0]
    
    eval_task_names = [task_name]
    if task_name == "mnli": eval_task_names.append("mnli-mm")
    
    if task_name == "stsb":
        metric_name = "pearson"
    elif task_name == "mnli":
        metric_name = "accuracy"
    elif task_name == "mlm":
        metric_name = "perplexity"
    
    
    if task_name == "mlm":
        best_score = 9999999999
    else:
        best_score = -1
    best_score_step = 0
    
    for step in keys:
        avg_score = 0
        for task in eval_task_names:
            avg_score += result_json[str(step)][task][metric_name]
        avg_score = avg_score / len(eval_task_names)
        
        
        
        if (avg_score > best_score and task_name != "mlm") or (avg_score < best_score and task_name == "mlm") :
            best_score = avg_score
            best_score_step = step
            best_score_epoch = best_score_step / steps_per_epoch
            
            
    result_json["best_result"] = {"epoch": best_score_epoch, "step":best_score_step, "score": best_score}
    with open(output_eval_file, 'w') as json_file:
        json.dump(result_json, json_file)
            
    epoch_diff = current_epoch - best_score_epoch
    logger.info(f"Best score {str(best_score)} with step {str(best_score_step)} at epoch {str(best_score_epoch)} with Diff. of {str(epoch_diff)} epochs.")
    
    if epoch_diff >= early_stopping_patience:
        stop_training = True
        logger.info(f"Stop training at epoch {str(current_epoch)} with {str(epoch_diff)} epochs difference to the best epoch.")
        
    return task_results, stop_training

        

def write_results(eval_output_dir,step,task_results,eval_task_names):
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(f"step: {step:<8d} ")
        line = "Res.: "
        for eval_task in eval_task_names:
            for key in task_results[eval_task].keys():
                res = task_results[eval_task][key]
                line += f"{eval_task}_{key}={res:.5f} "
        line += "\n"
        writer.write(line)
        
def write_results_json(eval_output_dir,step,task_results,eval_task_names):
    output_eval_file = os.path.join(eval_output_dir, "eval_results.json")
    
    if os.path.exists(output_eval_file):
        with open(output_eval_file) as json_file:
            result_json = json.load(json_file)
    else:
        result_json = {}

    result_json[step] = task_results

    with open(output_eval_file, 'w') as json_file:
        json.dump(result_json, json_file)

            

def predict_ens(models,eval_datasets,step,args):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_output_dir = args.output_dir
    task_results = {}
    for eval_task,eval_dataset in zip(eval_task_names, eval_datasets):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info(" task name = %s", eval_task)
        logger.info("  Num  examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)
        for model in models:
            model.eval()

        pred_logits = []
        label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            input_ids, input_mask, segment_ids, labels = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)

            with torch.no_grad():
                logits_list = [model(input_ids, input_mask, segment_ids) for model in models]
            logits = sum(logits_list)/len(logits_list)
            pred_logits.append(logits.detach().cpu())
            label_ids.append(labels)
        pred_logits = np.array(torch.cat(pred_logits),dtype=np.float32)
        label_ids = np.array(torch.cat(label_ids),dtype=np.int64)

        preds = np.argmax(pred_logits, axis=1)
        results = compute_metrics(eval_task, preds, label_ids)

        logger.info("***** Eval results {} task {} *****".format(step, eval_task))
        for key in sorted(results.keys()):
            logger.info(f"{eval_task} {key} = {results[key]:.5f}")
        task_results[eval_task] = results

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    write_results(output_eval_file,step,task_results,eval_task_names)
    for model in models:
        model.train()
    return task_results



