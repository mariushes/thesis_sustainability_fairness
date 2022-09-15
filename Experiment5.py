#!/usr/bin/env python
# coding: utf-8

# In[1]:


import distill
from init_layers import init_layers_dict
import result_extraction as re
import os

task_name = "mnli"

# In[2]:
RANDOM_SEED = 43154364

d = distill.Distillation(task_name=task_name, base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/" )


# In[3]:


d.load_tokenizer_dataset_preprocess()


best_model_dir, best_score, best_model_params = re.get_best_model_ef_series(epochs=10, bss=[16,32], lrs=[5e-5, 3e-5,2e-5], model_name="bert-base-uncased", task_name=task_name,seed=RANDOM_SEED)
teacher_model = os.path.join(best_model_dir,"torch_state_dict.pt")
# In[4]:





all_init_layers = [{"layer_S": 0, "layer_T": 0, "type": "all"}, {"layer_S": 1, "layer_T": 3, "type": "all"}, {"layer_S": 2, "layer_T": 6, "type": "all"}, {"layer_S": 3, "layer_T": 9, "type": "all"}, {"layer_S": 4, "layer_T": 12, "type": "all"}]

tps = [4,8]
bss = [64,128]

num_hidden_layers = 4

hidden_size = 768
for init_layers in all_init_layers:
    init_layers = [init_layers]
    for temperature in tps:
        for batch_size in bss:
            d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers, seed=RANDOM_SEED)

init_layers = all_init_layers[1:5]
for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers, seed=RANDOM_SEED)




RANDOM_SEED = 1909

d = distill.Distillation(task_name=task_name, base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/" )


# In[3]:


d.load_tokenizer_dataset_preprocess()


best_model_dir, best_score, best_model_params = re.get_best_model_ef_series(epochs=10, bss=[16,32], lrs=[5e-5, 3e-5,2e-5], model_name="bert-base-uncased", task_name=task_name,seed=RANDOM_SEED)
teacher_model = os.path.join(best_model_dir,"torch_state_dict.pt")
# In[4]:





all_init_layers = [{"layer_S": 0, "layer_T": 0, "type": "all"}, {"layer_S": 1, "layer_T": 3, "type": "all"}, {"layer_S": 2, "layer_T": 6, "type": "all"}, {"layer_S": 3, "layer_T": 9, "type": "all"}, {"layer_S": 4, "layer_T": 12, "type": "all"}]

tps = [4,8]
bss = [64,128]

num_hidden_layers = 4

hidden_size = 768
for init_layers in all_init_layers:
    init_layers = [init_layers]
    for temperature in tps:
        for batch_size in bss:
            d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers, seed=RANDOM_SEED)

init_layers = all_init_layers[1:5]
for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers, seed=RANDOM_SEED)


RANDOM_SEED = 11052022

d = distill.Distillation(task_name=task_name, base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/" )


# In[3]:


d.load_tokenizer_dataset_preprocess()


best_model_dir, best_score, best_model_params = re.get_best_model_ef_series(epochs=10, bss=[16,32], lrs=[5e-5, 3e-5,2e-5], model_name="bert-base-uncased", task_name=task_name,seed=RANDOM_SEED)
teacher_model = os.path.join(best_model_dir,"torch_state_dict.pt")
# In[4]:





all_init_layers = [{"layer_S": 0, "layer_T": 0, "type": "all"}, {"layer_S": 1, "layer_T": 3, "type": "all"}, {"layer_S": 2, "layer_T": 6, "type": "all"}, {"layer_S": 3, "layer_T": 9, "type": "all"}, {"layer_S": 4, "layer_T": 12, "type": "all"}]

tps = [4,8]
bss = [64,128]

num_hidden_layers = 4

hidden_size = 768
for init_layers in all_init_layers:
    init_layers = [init_layers]
    for temperature in tps:
        for batch_size in bss:
            d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers, seed=RANDOM_SEED)

init_layers = all_init_layers[1:5]
for temperature in tps:
    for batch_size in bss:
        d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, temperature=temperature, batch_size=batch_size, use_matches=False, init_layers=init_layers, seed=RANDOM_SEED)
