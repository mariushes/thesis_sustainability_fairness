#!/usr/bin/env python
# coding: utf-8

# In[1]:


import distill


# In[2]:


d = distill.Distillation(task_name="mnli", base_model_name = "bert-base-uncased", output_dir_prefix="eval_out/distilled/" )


# In[3]:


teacher_model = "/work/mhessent/master_thesis/eval_out/bert-base-uncased/43154364/mnli/lr2e-05_bs16_epochs10/torch_state_dict.pt"
RANDOM_SEED = 43154364


d.load_tokenizer_dataset_preprocess()




nhls = list(range(1,13))
tps = [8]
bss = [64,128]

for num_hidden_layers in nhls:
    for temperature in tps:
        for batch_size in bss:
            d.distill(teacher_model_path=teacher_model, num_epochs=60, num_hidden_layers=num_hidden_layers, temperature=temperature, batch_size=batch_size, use_matches=True, seed=RANDOM_SEED)
