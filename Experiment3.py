
import distill
from init_layers import init_layers_dict
import result_extraction as re
import os

d = distill.Distillation(task_name="mnli", base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/" )



d.load_tokenizer_dataset_preprocess()


RANDOM_SEED = 43154364
best_model_dir, best_score, best_model_params = re.get_best_model_ef_series(epochs=10, bss=[16,32], lrs=[5e-5, 3e-5,2e-5], model_name="bert-base-uncased", task_name="mnli",seed=RANDOM_SEED)
teacher_model = os.path.join(best_model_dir,"torch_state_dict.pt")


tps = [4,8]
bss = [64,128]

hidden_size = 768

num_hidden_layers = 1

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)], seed=RANDOM_SEED)


num_hidden_layers = 2

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)], seed=RANDOM_SEED)

num_hidden_layers = 3

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)], seed=RANDOM_SEED)


num_hidden_layers = 4

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)], seed=RANDOM_SEED)


num_hidden_layers = 5

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)], seed=RANDOM_SEED)

num_hidden_layers = 6

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers_dict[str(num_hidden_layers)], seed=RANDOM_SEED)
