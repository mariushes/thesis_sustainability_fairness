import distill
from init_layers import init_layers_dict


d = distill.MLMDistillationDebias(num_samples=5000, base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/debias/" )

d.load_tokenizer_dataset_preprocess()



tps = [4,8]
bss = [64,128]

hls = range(5,7)
for num_hidden_layers in hls:
    for temperature in tps:
        for batch_size in bss:
            d.distill(teacher_model_path="bert-base-uncased", num_epochs=40, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=False,init_layers=init_layers_dict[str(num_hidden_layers)], evaluate_teacher=True)
