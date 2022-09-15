

from evaluation_framework import *
import distill
import os

def evaluate(num_hidden_layers, hidden_size, weat, bias_eval):
    task_name = "mnli"
    base_model_name = "bert-base-uncased"
    output_dir_prefix="eval_out/distilled/"
    #num_hidden_layers = 1
    #hidden_size = 768
    use_matches = True
    tps = [4,8]
    bss = [64, 128]
    best_output_dir = ""
    best_best_model_name = ""
    best_best_score = 0
    for temperature in tps:
        for batch_size in bss:
            output_dir = output_dir_prefix + base_model_name + "/" + task_name + "/" + "hl"+ str(num_hidden_layers) + "_hs" +  str(hidden_size) + "_um" + str(use_matches)+ "_tp" + str(temperature) + "_bs" + str(batch_size) +  "/"
            best_model_name = distill.get_best_model_name(output_dir)
            best_model_score = distill.get_best_model_score(output_dir)
            if best_model_score > best_best_score:
                best_best_score = best_model_score
                best_best_model_name = best_model_name
                best_output_dir = output_dir
    print(best_best_score)
    print(best_best_model_name)
    print(best_output_dir)
    model_name_or_path = os.path.join(best_output_dir, "models", best_best_model_name)
    metric_for_best_model = "combined_score" if task_name == "stsb" else "accuracy"

    learning_rate = 3e-5
    epochs = 1
    batch_size = 32
    max_seq_length = 128




    #output_dir = "eval_out/distilled/" + config_name +"/" + task_name + "/" + "kd_test_config"

    ds_args = DatasetArguments(task_name = task_name, max_seq_length=max_seq_length)
                               #,max_train_samples = 2000, max_eval_samples=2000, max_predict_samples = 500)

    m_args = ModelArguments(model_name_or_path=model_name_or_path, config_name = base_model_name,
                            tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)

    t_args = TrainingArguments(output_dir=best_output_dir, overwrite_output_dir = True, do_train=False,
                               do_eval = True, do_predict = True, per_device_eval_batch_size = 32,
                               logging_steps=1000, no_cuda =False, seed=1909)

    if weat:
        weat_eval = list(range(3,11))
    else:
        weat_eval = None
    e = Evaluation(ds_args, m_args, t_args, bias_eval = bias_eval, weat_eval=weat_eval)
    e.set_up_logging()
    e.detect_checkpoint()
    e.load_data()
    model, tokenizer = e.load_model_tokenizer()
    e.preprocess_data()
    e.train()
