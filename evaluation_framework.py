import numpy as np
import random
import torch


from torch.utils.data import DataLoader
from textbrewer.distiller_utils import move_to_device
from datasets import load_dataset

import logging
import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_device(no_cuda=False):
    # If there's a GPU available...
    if torch.cuda.is_available() and not no_cuda:    
        DEVICE_NUM = 0
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda:" + str(DEVICE_NUM))
        torch.cuda.set_device(DEVICE_NUM)
        print("Current device: ", torch.cuda.current_device())

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(DEVICE_NUM))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device




# adapted from https://aclanthology.org/2021.findings-emnlp.411/
# Returns a list with lists inside. Each list represents a sentence pair: [[s1,s2], [s1,s2], ...]
def get_dataset_bias_sts(data_path):
    file1 = open(data_path, 'r', encoding="utf-8")
    lines = file1.readlines()
    sentence_pairs = []
    for line in lines: 
        entries = line.split("\t")
        if len(entries) > 1:  # ignore empty lines
            pair = [entries[0].replace('\n', ''), entries[1].replace('\n', ''), entries[2].replace('\n', ''), entries[3].replace('\n', '')]
            sentence_pairs.append(pair)
    return sentence_pairs


# model runs on STS-B task and returns similarity value of the sentence pair 
def predict_bias_sts(sentence, sentence2, model, tokenizer, device):
    max_length = 128
    """
    # create input ids
    input_ids1 = tokenizer(sentence, add_special_tokens=True, padding="max_length", max_length=max_length)
    input_ids2 = tokenizer(sentence2, add_special_tokens=True, padding="max_length", max_length=max_length)
    input_concat = list(input_ids1) + input_ids2[1:]
    input_ids = input_concat + ([0] * (max_length - len(input_concat)))

    # create attention mask
    attention_mask = ([1] * len(input_concat)) + ([0] * (max_length - len(input_concat)))

    # create token type ids
    token_type_ids = ([0] * len(input_ids1)) + ([1] * (len(input_ids2)-1)) + ([0] * (max_length - len(input_concat)))

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)
    """
    inputs = tokenizer(sentence, sentence2, add_special_tokens=True, padding="max_length", max_length=max_length, return_tensors='pt')
    inputs.to(device)
    # predict output tensor
    #outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda()) 
    outputs = model(**inputs)
    #print(type(outputs))
    #print(outputs)
    return outputs[0].tolist()[0][0]


# bias evaluation on CDA STS-B dataset
def evaluate_bias_sts(model, tokenizer, data_path, output_directory, device, logger):
    
    
    # get bias evaluation dataset
    pairs = get_dataset_bias_sts(data_path)
    number_pairs = len(pairs)

    # evaluation metrics
    highest_male = -1000.0 # highest similarity score for 'male' sentence pair
    lowest_male = 1000.0 # lowest similarity score for 'male' sentence pair
    highest_female = -1000.0 # highest similarity score for 'female' sentence pair
    lowest_female = 1000.0 # lowest similarity score for 'female' sentence pair
    pair_highest_male = [] # 'male' sentence pair with highest similarity score
    pair_lowest_male = [] # 'male' sentence pair with lowest similarity score
    pair_highest_female = [] # 'female' sentence pair with highest similarity score
    pair_lowest_female = [] # 'female' sentence pair with lowest similarity score
    highest_diff = 0.0 # highest similarity difference between a 'male' and 'female' sentence pair
    lowest_diff = 1000.0 # lowest similarity difference between a 'male' and 'female' sentence pair
    pair_highest_diff = [] # the two sentence pairs with the highest similarity difference
    pair_lowest_diff = [] # the two sentence pairs with the lowest similarity difference
    difference_abs_avg = 0.0 # absolute average of all differences between 'male' and 'female' sentence pairs: abs(male - female) 
    difference_avg = 0.0 # average of all differences between 'male' and 'female' sentence pairs: male - female
    male_avg = 0.0 # average similarity score for 'male' sentence pairs
    female_avg = 0.0 # average similarity score for 'female' sentence pairs
    threshold_01 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.1
    threshold_03 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.3
    threshold_05 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.5
    threshold_07 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.7

    # count the occurences to calculate the results
    counter = 0
    for p in pairs:
        if (counter % 1000) == 0:
            print(counter, " / ", number_pairs)
        # measure similarity of 'male' sentence pair 
        sim_male = predict_bias_sts(p[0], p[1], model, tokenizer, device)
        # measure similarity of 'female' sentence pair 
        sim_female = predict_bias_sts(p[2], p[3], model, tokenizer, device)
        """
        print()
        print()
        print()
        print(sim_male)
        print(sim_female)
        print()
        print()
        print()
        """
        # adjust measurements
        difference_abs = abs(sim_male - sim_female)
        difference = sim_male - sim_female
        if sim_male < lowest_male:
            lowest_male = sim_male
            pair_lowest_male = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if sim_female < lowest_female:
            lowest_female = sim_female
            pair_lowest_female = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if sim_male > highest_male:
            highest_male = sim_male
            pair_highest_male = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if sim_female > highest_female:
            highest_female = sim_female
            pair_highest_female = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if difference_abs < lowest_diff:
            lowest_diff = difference_abs
            pair_lowest_diff = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if difference_abs > highest_diff:
            highest_diff = difference_abs
            pair_highest_diff = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        male_avg += sim_male
        female_avg += sim_female
        difference_abs_avg += difference_abs
        difference_avg += difference
        if difference_abs > 0.1:
            threshold_01 += 1
        if difference_abs > 0.3:
            threshold_03 += 1
        if difference_abs > 0.5:
            threshold_05 += 1
        if difference_abs > 0.7:
            threshold_07 += 1
        counter += 1

    # get final results
    difference_abs_avg = difference_abs_avg / number_pairs
    difference_avg = difference_avg / number_pairs
    male_avg = male_avg / number_pairs
    female_avg = female_avg / number_pairs
    threshold_01 = threshold_01 / number_pairs
    threshold_03 = threshold_03 / number_pairs
    threshold_05 = threshold_05 / number_pairs
    threshold_07 = threshold_07 / number_pairs

    # print results
    logger.info("Difference absolut avg: "+ str(difference_abs_avg))
    logger.info("Difference avg (male - female): "+ str(difference_avg))
    logger.info("Male avg: "+ str(male_avg))
    logger.info("Female avg: "+ str(female_avg))
    logger.info("Threshold 01: "+ str(threshold_01))
    logger.info("Threshold 03: "+ str(threshold_03))
    logger.info("Threshold 05: "+ str(threshold_05))
    logger.info("Threshold 07: "+ str(threshold_07))
    logger.info("Highest prob male: "+ str(highest_male)+"   "+ str(pair_highest_male))
    logger.info("Highest prob female: "+ str(highest_female)+ "   "+ str(pair_highest_female))
    logger.info("Lowest prob male: "+ str(lowest_male)+ "   "+ str(pair_lowest_male))
    logger.info("Lowest prob female: "+ str(lowest_female)+"   "+ str(pair_lowest_female))
    logger.info("Highest diff: "+ str(highest_diff) + "   "+ str(pair_highest_diff))
    logger.info("Lowest diff: "+ str(lowest_diff)+ "   "+ str(pair_lowest_diff))

    #result_file = open("{}/evaluations/bias_results_bias_sts.txt".format(output_directory), "w")
    result_string = "Evaluation using Bias-sts-b dataset\n\nDifference absolut avg {0}\nDifference avg (male - female){1}\nMale avg {2}\nFemale avg {3}\nThreshold 01: {4}\nThreshold 03: {5}\nThreshold 05: {6}\nThreshold 07: {7}\nHighest similarity male: {8}   {9}\nHighest similarity female: {10}   {11}\nLowest similarity male: {12}   {13}\nLowest similarity female: {14}   {15}\nHighest difference: {16}   {17}\nLowest difference: {18}   {19}\n\n".format(round(difference_abs_avg,3), round(difference_avg,3), round(male_avg,3), round(female_avg,3), round(threshold_01,3), round(threshold_03,3), round(threshold_05,3), round(threshold_07,3), highest_male, pair_highest_male, highest_female, pair_highest_female, lowest_male, pair_lowest_male, lowest_female, pair_lowest_female, highest_diff, pair_highest_diff, lowest_diff, pair_lowest_diff)

    result_json = {
    "diff_abs_avg": difference_abs_avg,
        "diff_avg": difference_avg
    }
    
    return result_string, result_json


# In[3]:


# adapted from https://aclanthology.org/2021.findings-emnlp.411/
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
# Returns a list with lists inside. Each list represents a sentence pair: [[s1,s2], [s1,s2], ...]
def get_dataset(data_path, limit_pairs = None):
    sentence_pairs = []
    counter = 0
    with open(data_path, 'r') as file:
        for line in file:
            # skip head line and first empty line
            counter += 1
            if counter < 3:
                continue
            if counter == limit_pairs:
                break
            entries = line.split(",")
            if len(entries) > 1:  # ignore empty lines
                pair = [entries[5], entries[6].replace('\n', '')]
                sentence_pairs.append(pair)
            
    return sentence_pairs

# model runs on MNLI task and returns scores for neutral, entailment and contradiction 
def predict(sentence, sentence2, model, tokenizer, device):
    max_length = 128
    
    inputs = tokenizer(sentence, sentence2, add_special_tokens=True, padding="max_length", max_length=max_length, return_tensors='pt')
    inputs.to(device)
    
    outputs = model(**inputs)
    
    return outputs[0]


# bias evaluation on Bias-NLI occupation-gender dataset
def evaluate_nli(model, tokenizer, data_path, output_directory, device, logger, limit_pairs= None, batched=False, tokenizer_name=None):
    logger.info("Start NLI Bias function")
    softmax_function = nn.Softmax(dim=1)
        # evaluation metrics
    net_values = [0.0, 0.0, 0.0]
    fractions = [0.0, 0.0, 0.0]
    threshold_01 = [0.0, 0.0, 0.0]
    threshold_03 = [0.0, 0.0, 0.0]
    threshold_05 = [0.0, 0.0, 0.0]
    threshold_07 = [0.0, 0.0, 0.0]
    
    if batched:
        raw_datasets = load_dataset("csv", data_files=data_path)
        raw_datasets = raw_datasets.remove_columns(['id', 'pair type', 'premise_filler_word',
                                                    'hypothesis_filler_word', 'template_type'])
        
        sentence_keys = task_to_keys["mnli"]
        MAX_LENGTH = 128
        if tokenizer_name:
            directory = f'/work/mhessent/cache/{tokenizer_name}/{data_path.split(".")[0]}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            cache_files_names = {"train": f'/work/mhessent/cache/{tokenizer_name}/{data_path.split(".")[0]}/tokenized.arrow'}
        else:
            cache_files_names = None
            
        dataset = raw_datasets.map(lambda e: tokenizer(e[sentence_keys[0]],e[sentence_keys[1]],
                                                       truncation=True, padding='max_length',
                                                       max_length=MAX_LENGTH), batched=True, 
                                   cache_file_names = cache_files_names)
        if "roberta" == model.config.model_type:
            input_columns = ['input_ids', 'attention_mask']
        else:   
            input_columns = ['input_ids', 'token_type_ids', 'attention_mask']

        dataset.set_format(type='torch', columns=input_columns)
        
        dataloader = DataLoader(dataset["train"], batch_size=64)

        model.to(device)
        model.eval()
        all_probs = []
        all_preds = []
        
        logger.addHandler(TqdmLoggingHandler())
        for i,batch in enumerate(tqdm.tqdm(dataloader)):
            batch = {k: v for k, v in batch.items()}
            batch = move_to_device(batch,device)
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            probs = softmax_function(logits).tolist()[0]
            predictions = torch.argmax(logits, dim=-1)
            all_probs.append(probs)
            all_preds.append(predictions)
        
        
        for probs, pred_tensor in zip(all_probs, all_preds):
            net_values[0] += probs[0]
            net_values[1] += probs[1]
            net_values[2] += probs[2]
            for i in range(len(probs)):
                if probs[i] > 0.1:
                    threshold_01[i] += 1
                if probs[i] > 0.3:
                    threshold_03[i] += 1
                if probs[i] > 0.5:
                    threshold_05[i] += 1
                if probs[i] > 0.7:
                    threshold_07[i] += 1
            for pred in pred_tensor:
                fractions[pred] += 1

        number_pairs = len(dataset["train"])
        number_batches = len(dataloader)
        
        for i in range(3):
            net_values[i] = net_values[i] / number_batches
            fractions[i] = fractions[i] / number_pairs
            threshold_01[i] = threshold_01[i] / number_batches
            threshold_03[i] = threshold_03[i] / number_batches
            threshold_05[i] = threshold_05[i] / number_batches
            threshold_07[i] = threshold_07[i] / number_batches
        
    else:
        # get bias evaluation dataset
        pairs = get_dataset(data_path, limit_pairs=limit_pairs)
        number_pairs = len(pairs)

        # count the occurencies to calculate the results
        counter = 0
        for p in pairs:
            if (counter % 10000) == 0:
                logger.info(f"{str(counter)}  /  {str(number_pairs)}")
            # get scores for neutral, entailment and contradiction and apply softmax function to get probabilities
            prediction = predict(p[0], p[1], model, tokenizer, device)
            probs = softmax_function(prediction).tolist()[0]
            # print(probs)
            net_values[0] += probs[0]
            net_values[1] += probs[1]
            net_values[2] += probs[2]
            max_prob_label = torch.argmax(prediction).item()
            fractions[max_prob_label] += 1
            for i in range(len(probs)):
                if probs[i] > 0.1:
                    threshold_01[i] += 1
                if probs[i] > 0.3:
                    threshold_03[i] += 1
                if probs[i] > 0.5:
                    threshold_05[i] += 1
                if probs[i] > 0.7:
                    threshold_07[i] += 1
            counter += 1

        # get final results
        for i in range(3):
            net_values[i] = net_values[i] / number_pairs
            fractions[i] = fractions[i] / number_pairs
            threshold_01[i] = threshold_01[i] / number_pairs
            threshold_03[i] = threshold_03[i] / number_pairs
            threshold_05[i] = threshold_05[i] / number_pairs
            threshold_07[i] = threshold_07[i] / number_pairs

    # print results
    logger.info("net values: " + str(net_values))
    logger.info("fractions: "+ str(fractions))
    logger.info("threshold 0.1" + str(threshold_01))
    logger.info("threshold 0.3"+ str(threshold_03))
    logger.info("threshold 0.5"+ str(threshold_05))
    logger.info("threshold 0.7"+ str(threshold_07))
    logger.info("Net Neutral: "+ str(net_values[1]))
    logger.info("Fraction Neutral: "+ str(fractions[1]))
    logger.info("Threshold 0.1: "+  str(threshold_01[1]))
    logger.info("Threshold 0.3: "+ str(threshold_03[1]))
    logger.info("Threshold 0.5: "+ str(threshold_05[1]))
    logger.info("Threshold 0.7: "+ str(threshold_07[1]))
    result_string = "Evaluation using Bias-NLI dataset\n\n---All values:---\nnet values: {0}\nfractions: {1}\nthreshold 0.1: {2}\nthreshold 0.3: {3}\nthreshold 0.5: {4}\nthreshold 0.7: {5}\n\n---Values for neutral---\nNet Neutral: {6}\nFraction Neutral: {7}\nThreshold 0.1: {8}\nThreshold 0.3: {9}\nThreshold 0.5: {10}\nThreshold 0.7: {11}\n".format(net_values, fractions, threshold_01, threshold_03, threshold_05, threshold_07, net_values[1], fractions[1], threshold_01[1], threshold_03[1], threshold_05[1], threshold_07[1])
    #result_file = open("{}/evaluations/bias_results_bias_nli.txt".format(output_directory), "w")
    #result_file.write(result_sring)
    #result_file.close()
    result_json = {
        "net_values": net_values,
        "fractions": fractions,
        "threshold0.1": threshold_01,
        "threshold0.3": threshold_03,
        "threshold0.5": threshold_05,
        "threshold0.7": threshold_07,
        "net_neutral": net_values[1],
        "fraction_neutral": fractions[1]
    }
    return result_string, result_json


# In[6]:


# based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
import logging
import os
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import weat
import seat.sentbias.main as seat


class DatasetArguments:
    task_name = None
    dataset_name = None
    dataset_congif_name = None
    max_seq_length = None
    overwrite_cache = False
    pad_to_max_length = True
    train_file = None
    validation_file = None
    test_file = None
    max_train_samples = None
    max_eval_samples = None
    max_predict_samples = None
    
    
    def __init__(self, max_seq_length, task_name = None, dataset_name = None, dataset_config_name = None, overwrite_cache = False, pad_to_max_length = True, train_file = None, validation_file = None, test_file = None, max_train_samples = None, max_eval_samples= None, max_predict_samples = None):
        
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.max_seq_length = max_seq_length
        self.overwrite_cache = overwrite_cache
        self.pad_to_max_length = pad_to_max_length
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples
        
        
        
        if task_name is not None:
            self.task_name = task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
        



class ModelArguments:
    model_name_or_path = None
    config_name = None
    tokenizer_name = None
    cache_dir = None
    use_fast_tokenizer = False
    model_revision = None
    use_auth_token = False
    num_hidden_layers = None
    hidden_size = None
    
    def __init__(self, model_name_or_path, config_name = None, tokenizer_name = None, cache_dir = None, use_fast_tokenizer = False, model_revision = None, use_auth_token = False, num_hidden_layers=None, hidden_size=None, random_init=False):
        self.model_name_or_path = model_name_or_path
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        if config_name == None:
            self.config_name = model_name_or_path
        else:
            self.config_name = config_name
        
        if tokenizer_name == None:
            self.tokenizer_name = model_name_or_path
        else:
            self.tokenizer_name = tokenizer_name
        self.random_init = random_init

            
class Evaluation:
    data_args = None
    model_args = None
    training_args = None
    bias_eval = None
    weat_eval = None
    seat_eval = None
    
    logger = logging.getLogger(__name__)
    num_labels = None
    callsbacks = None
    
    save_state_dict = None
    
    model = None
    tokenizer = None
    config = None
    
    raw_datasets = None
    is_regression = None
    label_list = None
    last_checkpoint = None
    
    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    
    
    
    
    def __init__(self, ds_args, m_args, t_args, bias_eval=None, weat_eval=None, seat_eval=None, callbacks=None, save_state_dict=True):
        self.data_args = ds_args
        self.model_args = m_args
        self.training_args = t_args
        self.bias_eval = bias_eval
        self.weat_eval = weat_eval
        self.seat_eval = seat_eval
        self.callbacks = callbacks
        self.save_state_dict = save_state_dict
        
        random.seed(t_args.seed)
        np.random.seed(t_args.seed)
        torch.manual_seed(t_args.seed)
        torch.cuda.manual_seed_all(t_args.seed)
        
    def set_up_logging(self):
        log = logging.getLogger()
        for hdlr in log.handlers[:]:
            log.removeHandler(hdlr)
            
        if not os.path.exists(self.training_args.output_dir):
            os.makedirs(self.training_args.output_dir)
        logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(filename=self.training_args.output_dir+"/logs.log", mode='a')],
        )
        # To prevent overflowing tokens warning
        transformers.logging.set_verbosity_error()
        
        log_level = self.training_args.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        self.logger.warning(
        f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
        + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
    )
        self.logger.info(f"Training/evaluation parameters {self.training_args}")
        
        
        
        
    def detect_checkpoint(self):
        last_checkpoint = None
        if os.path.isdir(self.training_args.output_dir) and self.training_args.do_train and not self.training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
                self.logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        self.last_checkpoint = last_checkpoint
    
    def load_data(self):
        if self.data_args.task_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset("glue", self.data_args.task_name, cache_dir= self.model_args.cache_dir)
        elif self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name, self.data_args.dataset_config_name, cache_dir=self.model_args.cache_dir
            )
        else:
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {"train": self.data_args.train_file, "validation": self.data_args.validation_file}

            # Get the test dataset: you can provide your own CSV/JSON test file (see below)
            # when you use `do_predict` without specifying a GLUE benchmark task.
            if self.training_args.do_predict:
                if self.data_args.test_file is not None:
                    train_extension = self.data_args.train_file.split(".")[-1]
                    test_extension = self.data_args.test_file.split(".")[-1]
                    assert (
                        test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."
                    data_files["test"] = self.data_args.test_file
                else:
                    raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

            for key in data_files.keys():
                self.logger.info(f"load a local file for {key}: {data_files[key]}")

            if self.data_args.train_file.endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=self.model_args.cache_dir)
            else:
                # Loading a dataset from local json files
                raw_datasets = load_dataset("json", data_files=data_files, cache_dir=self.model_args.cache_dir)
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
        
        # Labels
        if self.data_args.task_name is not None:
            is_regression = self.data_args.task_name == "stsb"
            if not is_regression:
                self.label_list = raw_datasets["train"].features["label"].names
                self.num_labels = len(self.label_list)
            else:
                self.num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
            if is_regression:
                self.num_labels = 1
            else:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                self.label_list = raw_datasets["train"].unique("label")
                self.label_list.sort()  # Let's sort it for determinism
                self.num_labels = len(self.label_list)
        print(self.label_list)
        self.raw_datasets = raw_datasets
        self.is_regression = is_regression
        
        return raw_datasets
    
    def load_model_tokenizer(self):
        set_seed(self.training_args.seed)
        
        
        config = AutoConfig.from_pretrained(
        self.model_args.config_name,
        num_labels=self.num_labels,
        finetuning_task=self.data_args.task_name,
        cache_dir=self.model_args.cache_dir,
        revision=self.model_args.model_revision,
        use_auth_token=True if self.model_args.use_auth_token else None,
    )
        if self.model_args.num_hidden_layers:
            config.num_hidden_layers = self.model_args.num_hidden_layers
        if self.model_args.hidden_size:
            config.hidden_size = self.model_args.hidden_size
            config.num_attention_heads = int(self.model_args.hidden_size / 64)
            config.intermediate_size = self.model_args.hidden_size * 4
            
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None
        )
        
        if self.model_args.random_init:
            model = AutoModelForSequenceClassification.from_config(
                config=config
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        return model, tokenizer
    
    def preprocess_data(self):
        
        raw_datasets = self.raw_datasets
        is_regression = self.is_regression
        label_list = self.label_list
        
        if self.data_args.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[self.data_args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        # Padding strategy
        if self.data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
            self.model.config.label2id != PretrainedConfig(num_labels=self.num_labels).label2id
            and self.data_args.task_name is not None
            and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in self.model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(self.num_labels)}
            else:
                self.logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif self.data_args.task_name is None and not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        if label_to_id is not None:
            self.model.config.label2id = label_to_id
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}
        elif self.data_args.task_name is not None and not is_regression:
            self.model.config.label2id = {l: i for i, l in enumerate(label_list)}
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}

        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            self.logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        max_seq_length = min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result
        
        
        directory = f'/work/mhessent/cache/{self.model_args.tokenizer_name}/datasets/{self.data_args.task_name}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
          
        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                cache_file_names = {k: f'/work/mhessent/cache/{self.model_args.tokenizer_name}/datasets/{self.data_args.task_name}/tokenized_{str(k)}.arrow' for k in raw_datasets}
            )
            
        train_dataset = None
        eval_dataset = None
        predict_dataset = None
        
        if self.training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.data_args.max_train_samples))

        if self.training_args.do_eval:
            if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            # TODO: raw_datasets["validation_mismatched"] included in eval_datasets
            eval_dataset = raw_datasets["validation_matched" if self.data_args.task_name == "mnli" else "validation"]
            if self.data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(self.data_args.max_eval_samples))

        if self.training_args.do_predict or self.data_args.task_name is not None or self.data_args.test_file is not None:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test_matched" if self.data_args.task_name == "mnli" else "test"]
            if self.data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(self.data_args.max_predict_samples))

        # Log a few random samples from the training set:
        if self.training_args.do_train:
            for index in random.sample(range(len(train_dataset)), 3):
                self.logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.predict_dataset = predict_dataset
        self.raw_datasets = raw_datasets
        
        return train_dataset, eval_dataset, predict_dataset

            
    def train(self):
          # Get the metric function
        if self.data_args.task_name is not None:
            metric = load_metric("glue", self.data_args.task_name)
        else:
            metric = load_metric("accuracy")

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
            if self.data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif self.is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif self.training_args.fp16:
            data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=self.callbacks
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(self.train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            
            train_metrics = metrics
        
            if self.save_state_dict:
                torch.save(self.model.state_dict(), self.training_args.output_dir + '/torch_state_dict.pt')
            
            
        eval_metrics = None
        # Evaluation
        if self.training_args.do_eval:
            self.logger.info("*** Evaluate ***" + str(self.training_args.do_eval))
            eval_metrics = []
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [self.data_args.task_name]
            eval_datasets = [self.eval_dataset]
            if self.data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                eval_datasets.append(self.raw_datasets["validation_mismatched"])

            for eval_dataset, task in zip(eval_datasets, tasks):
                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = (
                    self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

                trainer.log_metrics("eval-"+task, metrics)
                trainer.save_metrics("eval-"+task, metrics)
                eval_metrics.append(metrics)
                
        predictions_list = None
        if self.training_args.do_predict:
            self.logger.info("*** Predict ***")
            predictions_list = []
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [self.data_args.task_name]
            predict_datasets = [self.predict_dataset]
            if self.data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                predict_datasets.append(self.raw_datasets["test_mismatched"])

            for predict_dataset, task in zip(predict_datasets, tasks):
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                predict_dataset = predict_dataset.remove_columns("label")
                predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
                predictions = np.squeeze(predictions) if self.is_regression else np.argmax(predictions, axis=1)
                predictions_list.append(predictions)
                output_predict_file = os.path.join(self.training_args.output_dir, f"predict_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_predict_file, "w") as writer:
                        self.logger.info(f"***** Predict results {task} *****")
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if self.is_regression:
                                writer.write(f"{index}\t{item:3.3f}\n")
                            else:
                                item = self.label_list[item]
                                writer.write(f"{index}\t{item}\n")
        
        if self.bias_eval:
            self.logger.info("*** Bias Evaluation ***")
            device = get_device(no_cuda=self.training_args.no_cuda)
            if self.data_args.task_name == "stsb":
                output_bias_file = os.path.join(self.training_args.output_dir, f"bias_results_{self.data_args.task_name}.txt")
                output_bias_json = os.path.join(self.training_args.output_dir, f"bias_results_{self.data_args.task_name}.json")
                

                bias_result, bias_result_json = evaluate_bias_sts(self.model, self.tokenizer, "datasets/bias_evaluation_STS-B.tsv", "", device, self.logger)

                with open(output_bias_file, "w") as writer:
                    writer.write(bias_result)

                with open(output_bias_json, "w") as json_file:
                    json.dump(bias_result_json, json_file)
        
            elif self.data_args.task_name == "mnli":
                nli_bias_output_dir = os.path.join(self.training_args.output_dir, "nli_bias")
                racial_bias_file = os.path.join(nli_bias_output_dir, "racial_bias_results.json")
                gender_bias_file = os.path.join(nli_bias_output_dir, "gender_bias_results.json")

                gender_bias_result, gender_bias_result_json = evaluate_nli(self.model, self.tokenizer, "datasets/gender_bias_nli_new.csv", "", device, logger=self.logger, limit_pairs = self.data_args.max_predict_samples, batched=True, tokenizer_name = self.model_args.tokenizer_name)

                racial_bias_result, racial_bias_result_json = evaluate_nli(self.model, self.tokenizer, "datasets/racial_bias_nli.csv", "", device, logger=self.logger, limit_pairs = self.data_args.max_predict_samples, batched=True, tokenizer_name = self.model_args.tokenizer_name)

                if not os.path.exists(nli_bias_output_dir):
                    os.makedirs(nli_bias_output_dir)

                with open(racial_bias_file, "w") as json_file:
                    json.dump(racial_bias_result_json, json_file)

                with open(gender_bias_file, "w") as json_file:
                    json.dump(gender_bias_result_json, json_file)
                                            
            
                
                
        if self.weat_eval:
            self.logger.info("*** WEAT Evaluation ***")
            device = get_device(no_cuda=self.training_args.no_cuda)
            for test_number in self.weat_eval:
                weat_result = weat.run_weat(test_number=test_number, model=self.model, tokenizer=self.tokenizer, lower=True, similarity_type="cosine", permutation_number=None, output_dir= self.training_args.output_dir,device = device, logger=self.logger, all_layer_combinations=True)
  
        if self.seat_eval:
            self.logger.info("*** SEAT Evaluation ***")
            seat_output_dir = os.path.join(self.training_args.output_dir, "seat")
            device = get_device(no_cuda=self.training_args.no_cuda)
            if type(self.seat_eval) is str:
                seat_tests = self.seat_eval
            else:
                seat_tests = "all_sent"
            seat_result = seat.run_seat(model=self.model, tokenizer=self.tokenizer, device=device, seed=self.training_args.seed, output_dir = seat_output_dir, tests=seat_tests,logger=self.logger)
            
            
        
        kwargs = {"finetuned_from": self.model_args.model_name_or_path, "tasks": "text-classification"}
        if self.data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = self.data_args.task_name
            kwargs["dataset"] = f"GLUE {self.data_args.task_name.upper()}"
    
            
        
        if self.training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)
        
    



from transformers import EarlyStoppingCallback
def evaluate(model_name_or_path, task_name):


    lrs = [5e-5, 3e-5,2e-5]
    bts = [32,16]

    metric_for_best_model = "combined_score" if task_name == "stsb" else "accuracy"

    learning_rate = 3e-5
    epochs = 10
    batch_size = 32
    max_seq_length = 128

    for learning_rate in lrs:
        for batch_size in bts:

            if model_name_or_path.startswith("eval_out/"):
                output_dir = model_name_or_path
            else:
                output_dir = "eval_out/" + model_name_or_path +"/" + task_name + "/" + "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_epochs"+str(epochs)

            ds_args = DatasetArguments(task_name = task_name, max_seq_length=max_seq_length)
                                       #,max_train_samples = 2000, max_eval_samples=2000, max_predict_samples = 500)
            m_args = ModelArguments(model_name_or_path=model_name_or_path)
            t_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir = True, do_train=True,
                                       do_eval = True, do_predict = True, learning_rate = learning_rate,
                                       per_device_train_batch_size = batch_size, per_device_eval_batch_size = 32,
                                       num_train_epochs = epochs, load_best_model_at_end = True,
                                       metric_for_best_model = metric_for_best_model, evaluation_strategy = "epoch",
                                       save_strategy = "epoch", save_steps = 5000, logging_steps=1000,
                                       save_total_limit = 1, no_cuda =False, seed=RANDOM_SEED) 


            e = Evaluation(ds_args, m_args, t_args, bias_eval = True, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            model, tokenizer = e.load_model_tokenizer()
            e.preprocess_data()
            e.train()




