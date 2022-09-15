from transformers import EarlyStoppingCallback
from evaluation_framework import *



lrs = [5e-5, 3e-5,2e-5]
bts = [32,16]
task_name = "mnli"
model_name_or_path="bert-base-uncased"
metric_for_best_model = "combined_score" if task_name == "stsb" else "accuracy"
RANDOM_SEED = 43154364

epochs = 10
max_seq_length = 128

for learning_rate in lrs:
    for batch_size in bts:

        if model_name_or_path.startswith("eval_out/"):
            output_dir = model_name_or_path
        else:
            output_dir = "eval_out/" + model_name_or_path +"/"+ str(RANDOM_SEED)+"/"+ task_name + "/" + "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_epochs"+str(epochs)

        ds_args = DatasetArguments(task_name = task_name, max_seq_length=max_seq_length)

        m_args = ModelArguments(model_name_or_path=model_name_or_path, random_init=False)
        t_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir = True, do_train=True,
                                   do_eval = True, do_predict = True, learning_rate = learning_rate,
                                   per_device_train_batch_size = batch_size, per_device_eval_batch_size = 32,
                                   num_train_epochs = epochs, load_best_model_at_end = True,
                                   metric_for_best_model = metric_for_best_model, evaluation_strategy = "epoch",
                                   save_strategy = "epoch", save_steps = 10000, logging_steps=1000,
                                   save_total_limit = 1, no_cuda =False, seed=RANDOM_SEED, report_to="none")


        e = Evaluation(ds_args, m_args, t_args, bias_eval = False, weat_eval=False, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
        e.set_up_logging()
        e.detect_checkpoint()
        e.load_data()
        model, tokenizer = e.load_model_tokenizer()
        e.preprocess_data()
        e.train()


lrs = [5e-5, 3e-5,2e-5]
bts = [32,16]
task_name = "stsb"
model_name_or_path="bert-base-uncased"
metric_for_best_model = "combined_score" if task_name == "stsb" else "accuracy"
RANDOM_SEED = 43154364

epochs = 10
max_seq_length = 128

for learning_rate in lrs:
    for batch_size in bts:

        if model_name_or_path.startswith("eval_out/"):
            output_dir = model_name_or_path
        else:
            output_dir = "eval_out/" + model_name_or_path +"/"+ str(RANDOM_SEED)+"/"+ task_name + "/" + "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_epochs"+str(epochs)

        ds_args = DatasetArguments(task_name = task_name, max_seq_length=max_seq_length)

        m_args = ModelArguments(model_name_or_path=model_name_or_path, random_init=False)
        t_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir = True, do_train=True,
                                   do_eval = True, do_predict = True, learning_rate = learning_rate,
                                   per_device_train_batch_size = batch_size, per_device_eval_batch_size = 32,
                                   num_train_epochs = epochs, load_best_model_at_end = True,
                                   metric_for_best_model = metric_for_best_model, evaluation_strategy = "epoch",
                                   save_strategy = "epoch", save_steps = 10000, logging_steps=1000,
                                   save_total_limit = 1, no_cuda =False, seed=RANDOM_SEED, report_to="none")


        e = Evaluation(ds_args, m_args, t_args, bias_eval = False, weat_eval=False, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
        e.set_up_logging()
        e.detect_checkpoint()
        e.load_data()
        model, tokenizer = e.load_model_tokenizer()
        e.preprocess_data()
        e.train()
