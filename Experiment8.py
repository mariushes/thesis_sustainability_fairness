import distill
import result_extraction as re
import os



d = distill.Distillation(task_name="mnli", base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/" )



d.load_tokenizer_dataset_preprocess()

RANDOM_SEED = 43154364
best_model_dir, best_score, best_model_params = re.get_best_model_ef_series(epochs=10, bss=[16,32], lrs=[5e-5, 3e-5,2e-5], model_name="bert-base-uncased", task_name="mnli",seed=RANDOM_SEED)
teacher_model = os.path.join(best_model_dir,"torch_state_dict.pt")


d.load_tokenizer_dataset_preprocess()



tps = [4,8]
bss = [64,128]

num_hidden_layers = 4

hidden_size = 96

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=True, seed=RANDOM_SEED)

hidden_size = 192

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=True, seed=RANDOM_SEED)

hidden_size = 384

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=True, seed=RANDOM_SEED)

hidden_size = 576

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=True, seed=RANDOM_SEED)
