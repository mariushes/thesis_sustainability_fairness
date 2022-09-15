import json
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM,DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric, DatasetDict
from functools import partial
from predict_function import predict, predict_and_early_stopping
import textbrewer
from src.textbrewer import GeneralDistiller
from src.textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from src.textbrewer.distiller_utils import move_to_device
import logging
from matches import matches
import crows_pair.metric as cp
import numpy as np

class Distillation:

    
    def __init__(self, task_name, base_model_name, output_dir_prefix):
        self.task_name = task_name
        self.base_model_name = base_model_name
        self.output_dir_prefix = output_dir_prefix
        
        

    def load_tokenizer_dataset_preprocess(self):
        self.train_dataset = load_dataset('glue', self.task_name, split='train')
        if self.task_name == "mnli":
            self.val_dataset = load_dataset('glue', self.task_name, split='validation_matched')
            self.val_mm_dataset = load_dataset('glue', self.task_name, split='validation_mismatched')
            self.test_dataset = load_dataset('glue', self.task_name, split='test_matched')
        elif self.task_name == "stsb":
            self.val_dataset = load_dataset('glue', self.task_name, split='validation')
            self.test_dataset = load_dataset('glue', self.task_name, split='test')
            



        self.train_dataset = self.train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        self.val_dataset = self.val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        if self.task_name == "mnli":
            self.val_mm_dataset = self.val_mm_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        self.test_dataset = self.test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

        self.val_dataset = self.val_dataset.remove_columns(['label'])
        if self.task_name == "mnli":
            self.val_mm_dataset = self.val_mm_dataset.remove_columns(['label'])
        self.test_dataset = self.test_dataset.remove_columns(['label'])
        self.train_dataset = self.train_dataset.remove_columns(['label'])


        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)


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
        sentence_keys = task_to_keys[self.task_name]
        MAX_LENGTH = 128
        self.train_dataset = self.train_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        self.val_dataset = self.val_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        if self.task_name == "mnli":
            self.val_mm_dataset = self.val_mm_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        self.test_dataset = self.test_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

        if "roberta" in self.base_model_name.lower():
            input_columns = ['input_ids', 'attention_mask', 'labels']
        else:   
            input_columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
            
        self.train_dataset.set_format(type='torch', columns=input_columns)
        self.val_dataset.set_format(type='torch', columns=input_columns)
        if self.task_name == "mnli":
            self.val_mm_dataset.set_format(type='torch', columns=input_columns)
        self.test_dataset.set_format(type='torch', columns=input_columns)

    
            
        
    def distill(self, teacher_model_path, num_epochs, num_hidden_layers, hidden_size = 768,temperature = 4, batch_size= 128, use_matches=False, init_layers = None, evaluate_teacher=True, seed=None):

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        teacher_config = AutoConfig.from_pretrained(self.base_model_name)
        teacher_config.output_hidden_states = True
        #device = 'cpu'

        # config
        teacher_config = AutoConfig.from_pretrained(self.base_model_name)
        teacher_config.output_hidden_states = True
        if self.task_name == "mnli":
            teacher_config.num_labels = 3
        elif self.task_name == "stsb":
            teacher_config.num_labels = 1
        
        if teacher_model_path.endswith(".pt"):
            teacher_model = AutoModelForSequenceClassification.from_config(teacher_config)
            teacher_model.load_state_dict(torch.load(teacher_model_path))
        else:
            teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_path, output_hidden_states = True)


        teacher_model = teacher_model.to(device=device)
        


        student_config = AutoConfig.from_pretrained(self.base_model_name)
        student_config.output_hidden_states = True
        if self.task_name == "mnli":
            student_config.num_labels = 3
        elif self.task_name == "stsb":
            student_config.num_labels = 1

        student_config.num_hidden_layers = num_hidden_layers
        student_config.hidden_size = hidden_size
        student_config.num_attention_heads = int(hidden_size / 64)
        student_config.intermediate_size = hidden_size * 4
        
        self.student_config = student_config

        continue_training = False
        student_model = AutoModelForSequenceClassification.from_config(student_config)
        if continue_training:
            student_model.load_state_dict(torch.load(''))
        
        
        if init_layers:
            for init_pair in init_layers:
                if init_pair["type"] == "all":
                    if init_pair["layer_T"] == 0:
                        teacher_layer = teacher_model.base_model.embeddings
                    else:
                        teacher_layer = teacher_model.base_model.encoder.layer[init_pair["layer_T"]-1]
                    
                    if init_pair["layer_S"] == 0:
                        student_model.base_model.embeddings = teacher_layer
                    else:
                        student_model.base_model.encoder.layer[init_pair["layer_S"]-1] = teacher_layer
                    #print(f"Initialized layer_S {str(init_pair["layer_S"])} with layer_T {str(init_pair["layer_S"])} with all weights.")
            
        
        student_model = student_model.to(device=device)
              
        
        
        assert teacher_model.config.vocab_size == student_model.config.vocab_size
        assert teacher_model.config.vocab_size == len(self.tokenizer)

        if evaluate_teacher:
            eval_dataloader = DataLoader(self.val_dataset, batch_size=32)

            metric= load_metric("glue",self.task_name)
            teacher_model.to(device)
            teacher_model.eval()
            for batch in eval_dataloader:
                batch = {k: v for k, v in batch.items()}
                batch = move_to_device(batch,device)
                with torch.no_grad():
                    outputs = teacher_model(**batch)

                predictions = outputs.logits
                if self.task_name == "mnli":
                    predictions = torch.argmax(predictions, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            print("Teacher model validation dataset score:")
            print(metric.compute())



        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128) #prepare dataloader
        num_training_steps = len(train_dataloader) * num_epochs
        # Optimizer and learning rate scheduler
        optimizer = AdamW(student_model.parameters(), lr=1e-4)

        scheduler_class = get_linear_schedule_with_warmup
        # arguments dict except 'optimizer'
        scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


        def simple_adaptor(batch, model_outputs):
            return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}
        
        intermediate_matches = []
        if use_matches:
            if hidden_size == 768:
                match_list = [f"L{str(num_hidden_layers)}_hidden_mse", f"L{str(num_hidden_layers)}_hidden_smmd"]
                for match in match_list:
                    intermediate_matches += matches[match]
            elif num_hidden_layers == 3:
                match_list = [f"L{str(num_hidden_layers)}n_hidden_mse", f"L{str(num_hidden_layers)}_hidden_smmd"]
                for match in match_list:
                    intermediate_matches += matches[match]
                if hidden_size != 384:
                    for match in intermediate_matches:
                        if "proj" in match:
                            match["proj"][1] = hidden_size
            elif num_hidden_layers == 4:
                match_list = [f"L{str(num_hidden_layers)}t_hidden_mse", f"L{str(num_hidden_layers)}_hidden_smmd"]
                for match in match_list:
                    intermediate_matches += matches[match]
                for match in intermediate_matches:
                    if "proj" in match:
                        match["proj"][1] = hidden_size
            
        
        output_dir = get_output_dir(init_layers, seed, self.output_dir_prefix, self.base_model_name, self.task_name, student_model.config.num_hidden_layers, student_model.config.hidden_size, use_matches, temperature, batch_size)
        
        if self.task_name == "stsb":
            kd_loss_type='mse'
        else:
            kd_loss_type="ce"
        
        distill_config = DistillationConfig(
            kd_loss_type=kd_loss_type,
            intermediate_matches=intermediate_matches
        )
        train_config = TrainingConfig(device=device, output_dir = output_dir + "models/", log_dir="./tensorboard_logs")



        # prepare callback function
        local_rank = -1
        predict_batch_size = 32
        device = device
        do_train_eval = True
        
        if self.task_name == "mnli":
            eval_datasets = [self.val_dataset,self.val_mm_dataset]
        else:
            eval_datasets = [self.val_dataset]

        callback_func = partial(predict_and_early_stopping, eval_datasets=eval_datasets, output_dir=output_dir+"results/",
                                task_name=self.task_name, local_rank=local_rank,
                                predict_batch_size=predict_batch_size,
                                device=device, early_stopping_patience=4, do_train_eval=do_train_eval, train_dataset=self.train_dataset.select(range(min(len(self.train_dataset),10000))))

        distiller = GeneralDistiller(
            train_config=train_config, distill_config=distill_config,
            model_T=teacher_model, model_S=student_model, 
            adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


        with distiller:
            distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=callback_func)
        
        
        clean_saved_models(output_dir)
        best_model_path = os.path.join(output_dir, "models", get_best_model_name(output_dir))
        student_model.load_state_dict(torch.load(best_model_path))
        self.student_model = student_model
        
    def test_student(self, test_model_path=None):

        if test_model_path:
            test_model = AutoModelForSequenceClassification.from_config(self.student_config)
            test_model.load_state_dict( torch.load(test_model_path) )
        else:
            test_model = self.student_model

        eval_dataloader = DataLoader(self.val_dataset, batch_size=32)



        metric= load_metric("glue",self.task_name)
        test_model.eval()
        for batch in eval_dataloader:
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs = test_model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        return metric.compute()
    
    
    
    
class MLMDistillation(Distillation):
    tokenizer=None
    output_dir=None
    def __init__(self, num_samples, base_model_name, output_dir_prefix):
        
        super().__init__("mlm", base_model_name,output_dir_prefix)
        self.num_samples = num_samples
    
    def load_tokenizer_dataset_preprocess(self):
        wiki = DatasetDict()
        num_test_samples = 10000
        wiki["test"] = load_dataset("wikipedia", "20200501.en", split=f"train[0:{num_test_samples}]", cache_dir="/work/mhessent/cache/wikipedia")
        wiki["train"] = load_dataset("wikipedia", "20200501.en", split=f"train[{num_test_samples}:{num_test_samples+self.num_samples}]", cache_dir="/work/mhessent/cache/wikipedia")
        wiki = wiki.flatten()

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=False)
        
        def preprocess_function(examples):
            return self.tokenizer(["".join(x) for x in examples["text"]], return_special_tokens_mask=True, truncation=True, padding=True)
        
        if self.num_samples > 10000:
            num_proc = 20
        else:
            num_proc = 8
        
        tokenized_wiki = wiki.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=wiki["train"].column_names)
            
        block_size = 128
        

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        self.lm_dataset = tokenized_wiki.map(group_texts, batched=True, num_proc=num_proc)
        
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.lm_dataloader = torch.utils.data.DataLoader(
            self.lm_dataset["train"],
            batch_size=128,
            collate_fn=self.data_collator,
    drop_last=True
        )
        
    def load_model_tokenizer(self, teacher_model_path, num_hidden_layers, hidden_size = 768, init_layers = None, student_model_path=None):
        
        
        device ='cuda' if torch.cuda.is_available() else 'cpu'
        
        if teacher_model_path.endswith(".pt"):
            teacher_config = AutoConfig.from_pretrained(self.base_model_name)
            teacher_config.output_hidden_states = True
            teacher_model = AutoModelForMaskedLM.from_config(teacher_config)
            teacher_model.load_state_dict(torch.load(teacher_model_path))
        else:
            teacher_model = AutoModelForMaskedLM.from_pretrained(teacher_model_path, output_hidden_states = True)
        
        teacher_model = teacher_model.to(device=device)
        
        student_config = AutoConfig.from_pretrained(self.base_model_name)
        student_config.output_hidden_states = True
        
        student_config.num_hidden_layers = num_hidden_layers
        student_config.hidden_size = hidden_size
        student_config.num_attention_heads = int(hidden_size / 64)
        student_config.intermediate_size = hidden_size * 4
        
        self.student_config = student_config
        
        student_model = AutoModelForMaskedLM.from_config(student_config)
        if student_model_path:
            student_model.load_state_dict(torch.load(student_model_path))
        elif init_layers:
            for init_pair in init_layers:
                if init_pair["type"] == "all":
                    if init_pair["layer_T"] == 0:
                        teacher_layer = teacher_model.base_model.embeddings
                    else:
                        teacher_layer = teacher_model.base_model.encoder.layer[init_pair["layer_T"]-1]
                    
                    if init_pair["layer_S"] == 0:
                        student_model.base_model.embeddings = teacher_layer
                    else:
                        student_model.base_model.encoder.layer[init_pair["layer_S"]-1] = teacher_layer
        
        student_model = student_model.to(device=device)
        self.student_model = student_model
        self.teacher_model = teacher_model
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=False)

        return self.teacher_model, self.student_model, self.tokenizer
    
    
    def distill(self, teacher_model_path, num_epochs, num_hidden_layers, hidden_size = 768,temperature = 4, batch_size= 128, use_matches=False, init_layers = None, evaluate_teacher=True, seed=None):
        
        device ='cuda' if torch.cuda.is_available() else 'cpu'
        
        if teacher_model_path.endswith(".pt"):
            teacher_config = AutoConfig.from_pretrained(self.base_model_name)
            teacher_config.output_hidden_states = True
            teacher_model = AutoModelForMaskedLM.from_config(teacher_config)
            teacher_model.load_state_dict(torch.load(teacher_model_path))
        else:
            teacher_model = AutoModelForMaskedLM.from_pretrained(teacher_model_path, output_hidden_states = True)
        
        teacher_model = teacher_model.to(device=device)
        
        student_config = AutoConfig.from_pretrained(self.base_model_name)
        student_config.output_hidden_states = True
        
        student_config.num_hidden_layers = num_hidden_layers
        student_config.hidden_size = hidden_size
        student_config.num_attention_heads = int(hidden_size / 64)
        student_config.intermediate_size = hidden_size * 4
        
        self.student_config = student_config
        
        student_model = AutoModelForMaskedLM.from_config(student_config)
        
        if init_layers:
            for init_pair in init_layers:
                if init_pair["type"] == "all":
                    if init_pair["layer_T"] == 0:
                        teacher_layer = teacher_model.base_model.embeddings
                    else:
                        teacher_layer = teacher_model.base_model.encoder.layer[init_pair["layer_T"]-1]
                    
                    if init_pair["layer_S"] == 0:
                        student_model.base_model.embeddings = teacher_layer
                    else:
                        student_model.base_model.encoder.layer[init_pair["layer_S"]-1] = teacher_layer
        
        student_model = student_model.to(device=device)
              
        assert teacher_model.config.vocab_size == student_model.config.vocab_size
        assert teacher_model.config.vocab_size == len(self.tokenizer)
        
        
        num_training_steps = len(self.lm_dataloader) * num_epochs
        # Optimizer and learning rate scheduler
        optimizer = AdamW(student_model.parameters(), lr=1e-4)

        scheduler_class = get_linear_schedule_with_warmup
        # arguments dict except 'optimizer'
        scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


        def simple_adaptor(batch, model_outputs):
            return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}

        output_dir = get_output_dir(init_layers, seed, self.output_dir_prefix, self.base_model_name, self.task_name, student_model.config.num_hidden_layers, student_model.config.hidden_size, use_matches, temperature, batch_size)
        self.output_dir = output_dir
        eval_datasets = [self.lm_dataset["test"]]
        train_dataset = self.lm_dataset["train"].select(range(min(5000, len(self.lm_dataset["train"]))))
        
        if evaluate_teacher:
            logger = logging.getLogger(__name__)
            test_teacher_dir = os.path.join(output_dir,"test_teacher")
            test_student_dir = os.path.join(output_dir,"test_student")
            if not os.path.exists(test_teacher_dir):
                os.makedirs(test_teacher_dir)
            if not os.path.exists(test_student_dir):
                os.makedirs(test_student_dir)
            
            teacher_result = predict(teacher_model,eval_datasets=eval_datasets, step=0,epoch=0,logger=logger, output_dir=test_teacher_dir, task_name=self.task_name, local_rank=-1,predict_batch_size=128, device=device, data_collator=self.data_collator, do_train_eval=True,
                                train_dataset=train_dataset)
            print(teacher_result)
            student_result = predict(student_model,eval_datasets=eval_datasets,step=0,epoch=0,logger=logger,output_dir=test_student_dir,task_name=self.task_name,local_rank=-1,predict_batch_size=128,device=device, data_collator=self.data_collator, do_train_eval=True,
                                train_dataset=train_dataset)
            print(student_result)
        
        intermediate_matches = []
        if use_matches:
            for match in match_list_L6_cos:
                intermediate_matches += matches[match]
        
        distill_config = DistillationConfig(intermediate_matches=intermediate_matches)    

        train_config = TrainingConfig(device=device, output_dir = output_dir + "models/", log_dir="./tensorboard_logs")


        local_rank = -1
        predict_batch_size = 128
        device = device
        do_train_eval = True

        
        callback_func = partial(predict_and_early_stopping, eval_datasets=eval_datasets,
                                output_dir=output_dir+"results/",
                                task_name=self.task_name, local_rank=local_rank,
                                predict_batch_size=predict_batch_size,
                                device=device, early_stopping_patience=4, do_train_eval=do_train_eval,
                                train_dataset=train_dataset, data_collator=self.data_collator)

        distiller = GeneralDistiller(
            train_config=train_config, distill_config=distill_config,
            model_T=teacher_model, model_S=student_model, 
            adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


        with distiller:
            distiller.train(optimizer, self.lm_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=callback_func)
        
        clean_saved_models(output_dir)
        best_model_path = os.path.join(output_dir, "models", get_best_model_name(output_dir))
        student_model.load_state_dict(torch.load(best_model_path))
        self.student_model = student_model
        
        
    def crows_evaluation(self, output_dir=None):
        uncased = True
        if not output_dir:
            if self.output_dir:
                output_dir = self.output_dir
            else:
                raise ValueError("Provide output dir or run distill to initilize self.output_dir.")
        student_output_dir = os.path.join(output_dir, "crows")
        result_student = cp.evaluate(self.student_model, uncased,self.tokenizer, student_output_dir)
        return result_student
    
class MLMDistillationDebias(MLMDistillation):
    
    def __init__(self, num_samples, base_model_name, output_dir_prefix):
        super().__init__(num_samples, base_model_name, output_dir_prefix)
    
    def distill(self, teacher_model_path, num_epochs, num_hidden_layers, hidden_size = 768,temperature = 4, batch_size= 128, use_matches=False, init_layers = None, evaluate_teacher=True, seed=None
               ):
        
        device ='cuda' if torch.cuda.is_available() else 'cpu'
        
        if teacher_model_path.endswith(".pt"):
            teacher_config = AutoConfig.from_pretrained(self.base_model_name)
            teacher_config.output_hidden_states = True
            teacher_model = AutoModelForMaskedLM.from_config(teacher_config)
            teacher_model.load_state_dict(torch.load(teacher_model_path))
        else:
            teacher_model = AutoModelForMaskedLM.from_pretrained(teacher_model_path, output_hidden_states = True)
        
        teacher_model = teacher_model.to(device=device)
        
        student_config = AutoConfig.from_pretrained(self.base_model_name)
        student_config.output_hidden_states = True
        
        student_config.num_hidden_layers = num_hidden_layers
        student_config.hidden_size = hidden_size
        student_config.num_attention_heads = int(hidden_size / 64)
        student_config.intermediate_size = hidden_size * 4
        
        self.student_config = student_config
        
        student_model = AutoModelForMaskedLM.from_config(student_config)
        
        if init_layers:
            for init_pair in init_layers:
                if init_pair["type"] == "all":
                    if init_pair["layer_T"] == 0:
                        teacher_layer = teacher_model.base_model.embeddings
                    else:
                        teacher_layer = teacher_model.base_model.encoder.layer[init_pair["layer_T"]-1]
                    
                    if init_pair["layer_S"] == 0:
                        student_model.base_model.embeddings = teacher_layer
                    else:
                        student_model.base_model.encoder.layer[init_pair["layer_S"]-1] = teacher_layer
        
        student_model = student_model.to(device=device)
              
        assert teacher_model.config.vocab_size == student_model.config.vocab_size
        assert teacher_model.config.vocab_size == len(self.tokenizer)
        
        
        num_training_steps = len(self.lm_dataloader) * num_epochs
        # Optimizer and learning rate scheduler
        optimizer = AdamW(student_model.parameters(), lr=1e-4)

        scheduler_class = get_linear_schedule_with_warmup
        # arguments dict except 'optimizer'
        scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


        def debias_adaptor(batch, model_outputs):
            return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states, "batch":batch}

        output_dir = get_output_dir(init_layers, seed, self.output_dir_prefix, self.base_model_name, self.task_name, student_model.config.num_hidden_layers, student_model.config.hidden_size, use_matches, temperature, batch_size)
        self.output_dir = output_dir
        eval_datasets = [self.lm_dataset["test"]]
        train_dataset = self.lm_dataset["train"].select(range(min(5000, len(self.lm_dataset["train"]))))
        
        if evaluate_teacher:
            logger = logging.getLogger(__name__)
            test_teacher_dir = os.path.join(output_dir,"test_teacher")
            test_student_dir = os.path.join(output_dir,"test_student")
            if not os.path.exists(test_teacher_dir):
                os.makedirs(test_teacher_dir)
            if not os.path.exists(test_student_dir):
                os.makedirs(test_student_dir)
            
            teacher_result = predict(teacher_model,eval_datasets=eval_datasets, step=0,epoch=0,logger=logger, output_dir=test_teacher_dir, task_name=self.task_name, local_rank=-1,predict_batch_size=128, device=device, data_collator=self.data_collator, do_train_eval=True,
                                train_dataset=train_dataset)
            print(teacher_result)
            student_result = predict(student_model,eval_datasets=eval_datasets,step=0,epoch=0,logger=logger,output_dir=test_student_dir,task_name=self.task_name,local_rank=-1,predict_batch_size=128,device=device, data_collator=self.data_collator, do_train_eval=True,
                                train_dataset=train_dataset)
            print(student_result)
        
        intermediate_matches = []
        if use_matches:
            for match in match_list_L6_cos:
                intermediate_matches += matches[match]
        
        distill_config = DistillationConfig(intermediate_matches=intermediate_matches,kd_loss_weight=0.5,debias_loss_type="mlm_db", debias_loss_weight=0.5)    

        train_config = TrainingConfig(device=device, output_dir = output_dir + "models/", log_dir="./tensorboard_logs")


        local_rank = -1
        predict_batch_size = 128
        device = device
        do_train_eval = True

        
        callback_func = partial(predict_and_early_stopping, eval_datasets=eval_datasets,
                                output_dir=output_dir+"results/",
                                task_name=self.task_name, local_rank=local_rank,
                                predict_batch_size=predict_batch_size,
                                device=device, early_stopping_patience=4, do_train_eval=do_train_eval,
                                train_dataset=train_dataset, data_collator=self.data_collator)

        distiller = GeneralDistiller(
            train_config=train_config, distill_config=distill_config,
            model_T=teacher_model, model_S=student_model, 
            adaptor_T=debias_adaptor, adaptor_S=debias_adaptor)


        with distiller:
            distiller.train(optimizer, self.lm_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=callback_func)
        
        clean_saved_models(output_dir)
        best_model_path = os.path.join(output_dir, "models", get_best_model_name(output_dir))
        student_model.load_state_dict(torch.load(best_model_path))
        self.student_model = student_model
        
def clean_saved_models(output_dir):
    best_model_filename = get_best_model_name(output_dir)

    files = os.listdir(os.path.join(output_dir, "models"))
    files.remove(best_model_filename)
    print("Kept best model file: ", best_model_filename)

    for file_name in files:
        file_path = os.path.join(output_dir, "models", file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print("Removed ", file_name)


def get_best_model_name(output_dir):

    output_eval_file = os.path.join(output_dir, "results","eval_results.json")
    with open(output_eval_file) as json_file:
        result_json = json.load(json_file)

    best_step = result_json["best_result"]["step"]
    return "gs" + str(best_step) + ".pkl"

def get_best_model_score(output_dir):
    output_eval_file = os.path.join(output_dir, "results","eval_results.json")
    with open(output_eval_file) as json_file:
        result_json = json.load(json_file)

    best_score = result_json["best_result"]["score"]
    return best_score

def get_best_model_epoch(output_dir):
    output_eval_file = os.path.join(output_dir, "results","eval_results.json")
    with open(output_eval_file) as json_file:
        result_json = json.load(json_file)
    
    return result_json["best_result"]["epoch"]

def get_output_dir(init_layers, seed, output_dir_prefix, base_model_name, task_name, num_hidden_layers, hidden_size, use_matches, temperature, batch_size):
    init_layers_string = ""
    if init_layers:
        for layers_pair in init_layers:
            init_layers_string += "_" + "il" + str(layers_pair["layer_S"]) + "-" + str(layers_pair["layer_T"]) + layers_pair["type"]

    seed_string = ""
    if seed:
        seed_string = str(seed) + "/"
    output_dir = output_dir_prefix + base_model_name  + "/" + seed_string + task_name + "/" + "hl"+ str(num_hidden_layers) + "_hs" +  str(hidden_size) + "_um" + str(use_matches)+ "_tp" + str(temperature) + "_bs" + str(batch_size) + init_layers_string + "/"
    
    return output_dir
        
def get_bias_based_init_layers(teacher_dir, test_number, num_hidden_layers_student, num_hidden_layers_teacher):
    effect_sizes = []
    weat_dir = os.path.join(teacher_dir,"weat")
    for hl in range(1, num_hidden_layers_teacher + 1):
        file_name = os.path.join(weat_dir, f"weat{test_number}_n{hl-1}_m{hl}_result.json")
        with open(file_name, "r") as f:
            data = json.load(f)
        effect_sizes.append(data["effect_size"])
    effect_sizes = np.array(effect_sizes)
    least_biased_layers = np.argsort(effect_sizes)[:num_hidden_layers_student]
    least_biased_layers.sort()
    init_layers = [{"layer_S": 0, "layer_T": 0, "type": "all"}]
    for i in range(0,num_hidden_layers_student):
        init_layers.append({"layer_S":i+1, "layer_T":least_biased_layers[i]+1, "type":"all"})
    return init_layers