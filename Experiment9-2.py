from evaluation_framework import *
import result_extraction as re
import distill
from init_layers import init_layers_dict

hls = range(1,7)
for num_hidden_layers in hls:        
    RANDOM_SEED = 11052022
    task_name = "stsb"
    base_model_name = "bert-base-uncased"
    output_dir_prefix="eval_out/distilled/new_token/"
    #num_hidden_layers = 1
    hidden_size = 768
    use_matches = False

    kd_task_name = "mlm"
    init_layers=init_layers_dict[str(num_hidden_layers)]
    tps = [4,8]
    bss = [64,128]
    best_score, best_model_name,best_model_dir, best_model_params = re.get_best_model_kd_series(tps, bss, output_dir_prefix, base_model_name, kd_task_name, num_hidden_layers, hidden_size, use_matches, init_layers)
    print(best_score)
    print(best_model_params)
    temperature = best_model_params[0]
    kd_batch_size = best_model_params[1]
    best_model_name = distill.get_best_model_name(best_model_dir)
    model_name_or_path = os.path.join(best_model_dir, "models", best_model_name)
    #model_name_or_path = "/work/mhessent/TextBrewer/examples/notebook_examples/outputs/bert-base-uncased/mlm/hl6_hs768/models/gs16843.pkl"
    metric_for_best_model = "combined_score" if task_name == "stsb" else "accuracy"

    lrs = [5e-5, 3e-5,2e-5]
    bts = [32,16]

    #learning_rate = 3e-5
    epochs = 10
    #batch_size = 32
    max_seq_length = 128
    init_layers_string = ""
    if init_layers:
        for layers_pair in init_layers:
            init_layers_string += "_" + "il" + str(layers_pair["layer_S"]) + "-" + str(layers_pair["layer_T"]) + layers_pair["type"]




    for learning_rate in lrs:
        for batch_size in bts:
            output_dir = output_dir_prefix + base_model_name +"/"+ "mlm"  + "/" + "hl"+ str(num_hidden_layers) + "_hs" +  str(hidden_size) + "_um" + str(use_matches)+ "_tp" + str(temperature) + "_bs" + str(kd_batch_size) + init_layers_string+  "/"+ str(RANDOM_SEED)+"/"+  task_name + "/"+ "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_epochs"+str(epochs) + "/"

            ds_args = DatasetArguments(task_name = task_name, max_seq_length=max_seq_length)


            m_args = ModelArguments(model_name_or_path=model_name_or_path, config_name = base_model_name,
                                    tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)


            t_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir = True, do_train=True,
                                       do_eval = True, do_predict = True, learning_rate = learning_rate,
                                       per_device_train_batch_size = batch_size, per_device_eval_batch_size = 32,
                                       num_train_epochs = epochs, load_best_model_at_end = True,
                                       metric_for_best_model = metric_for_best_model, evaluation_strategy = "epoch",
                                       save_strategy = "epoch", save_steps = 10000, logging_steps=1000,
                                       save_total_limit = 1, no_cuda =False, seed=RANDOM_SEED, report_to="none") 



            if weat:
                weat_eval = list(range(3,11))
            else:
                weat_eval = None
            e = Evaluation(ds_args, m_args, t_args, bias_eval = False, weat_eval=None, seat_eval=None)
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            model, tokenizer = e.load_model_tokenizer()
            e.preprocess_data()
            e.train()
            
hls = range(1,7)
for num_hidden_layers in hls:        
    RANDOM_SEED = 43154364
    task_name = "stsb"
    base_model_name = "bert-base-uncased"
    output_dir_prefix="eval_out/distilled/new_token/"
    #num_hidden_layers = 1
    hidden_size = 768
    use_matches = False

    kd_task_name = "mlm"
    init_layers=init_layers_dict[str(num_hidden_layers)]
    tps = [4,8]
    bss = [64,128]
    best_score, best_model_name,best_model_dir, best_model_params = re.get_best_model_kd_series(tps, bss, output_dir_prefix, base_model_name, kd_task_name, num_hidden_layers, hidden_size, use_matches, init_layers)
    print(best_score)
    print(best_model_params)
    temperature = best_model_params[0]
    kd_batch_size = best_model_params[1]
    best_model_name = distill.get_best_model_name(best_model_dir)
    model_name_or_path = os.path.join(best_model_dir, "models", best_model_name)
    #model_name_or_path = "/work/mhessent/TextBrewer/examples/notebook_examples/outputs/bert-base-uncased/mlm/hl6_hs768/models/gs16843.pkl"
    metric_for_best_model = "combined_score" if task_name == "stsb" else "accuracy"

    lrs = [5e-5, 3e-5,2e-5]
    bts = [32,16]

    #learning_rate = 3e-5
    epochs = 10
    #batch_size = 32
    max_seq_length = 128
    init_layers_string = ""
    if init_layers:
        for layers_pair in init_layers:
            init_layers_string += "_" + "il" + str(layers_pair["layer_S"]) + "-" + str(layers_pair["layer_T"]) + layers_pair["type"]




    for learning_rate in lrs:
        for batch_size in bts:
            output_dir = output_dir_prefix + base_model_name +"/"+ "mlm"  + "/" + "hl"+ str(num_hidden_layers) + "_hs" +  str(hidden_size) + "_um" + str(use_matches)+ "_tp" + str(temperature) + "_bs" + str(kd_batch_size) + init_layers_string+  "/"+ str(RANDOM_SEED)+"/"+  task_name + "/"+ "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_epochs"+str(epochs) + "/"

            ds_args = DatasetArguments(task_name = task_name, max_seq_length=max_seq_length)


            m_args = ModelArguments(model_name_or_path=model_name_or_path, config_name = base_model_name,
                                    tokenizer_name=base_model_name, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)


            t_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir = True, do_train=True,
                                       do_eval = True, do_predict = True, learning_rate = learning_rate,
                                       per_device_train_batch_size = batch_size, per_device_eval_batch_size = 32,
                                       num_train_epochs = epochs, load_best_model_at_end = True,
                                       metric_for_best_model = metric_for_best_model, evaluation_strategy = "epoch",
                                       save_strategy = "epoch", save_steps = 10000, logging_steps=1000,
                                       save_total_limit = 1, no_cuda =False, seed=RANDOM_SEED, report_to="none") 



            if weat:
                weat_eval = list(range(3,11))
            else:
                weat_eval = None
            e = Evaluation(ds_args, m_args, t_args, bias_eval = False, weat_eval=None, seat_eval=None)
            e.set_up_logging()
            e.detect_checkpoint()
            e.load_data()
            model, tokenizer = e.load_model_tokenizer()
            e.preprocess_data()
            e.train()