import distill
from init_layers import init_layers_dict

d = distill.MLMDistillation(num_samples=5000, base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/" )

d.load_tokenizer_dataset_preprocess()



tps = [4,8]
bss = [64,128]


num_hidden_layers = 6

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path="bert-base-uncased", num_epochs=40, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=False,init_layers=init_layers_dict[str(num_hidden_layers)], evaluate_teacher=True)


num_hidden_layers = 5

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path="bert-base-uncased", num_epochs=40, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=False,init_layers=init_layers_dict[str(num_hidden_layers)], evaluate_teacher=True)

num_hidden_layers = 4

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path="bert-base-uncased", num_epochs=40, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=False,init_layers=init_layers_dict[str(num_hidden_layers)], evaluate_teacher=True)

num_hidden_layers = 3

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path="bert-base-uncased", num_epochs=40, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=False,init_layers=init_layers_dict[str(num_hidden_layers)], evaluate_teacher=True)


num_hidden_layers = 2

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path="bert-base-uncased", num_epochs=40, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=False,init_layers=init_layers_dict[str(num_hidden_layers)], evaluate_teacher=True)

num_hidden_layers = 1

for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path="bert-base-uncased", num_epochs=40, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=False,init_layers=init_layers_dict[str(num_hidden_layers)], evaluate_teacher=True)
