# Code submission for the master thesis "Sustainability and Fairness in Pretrained Language Models: Analysis and Mitigation of Bias when Distilling BERT" by Marius A. Hessenthaler

We will give an overview on the files and folders in the following.

## Experiments

### ExperimentX.py
The files for running the experiments are named after the experiment where X is the experiment number.

### results
Contains the data for the reported and additional experiments.

### result_extraction.py
Contains code for running tests and gathering results for experiments and saving them as csv.

### evaluate_experiments.py
Contains code run evaluation in result_extraction.py for different experiments.

### Plots_Tables.ipynb
Contains code to generate Plots and Tables for experiments.

## Fine-tuning and evaluation

### evaluation_framework.py
Contains the code for fine-tuning models and evaluating them.

### finetune_bert_teacher.py
Contains the code to fine-tune BERT techer models.

## Knowledge distillation

### src
Contains the code for the adapted TextBrewer framework from https://github.com/airaria/TextBrewer to run knowledge distillation.

### distill.py
Contains code for wrapper classes and utility functions for the distillation with TextBrewer.

### predict_function.py
Contains code to for callback evaluation in KD, adpated from https://github.com/airaria/TextBrewer .

### utils_glue.py
Contains utility functions for GLUE evaluation, adapted from https://github.com/airaria/TextBrewer .

### utils.py
Contains utility functions for KD, adapted from https://github.com/airaria/TextBrewer .

### init_layers.py
Contains the mapping for initializing students by teachers.

### matches.py
Contains the mapping for embeddings loss functions, adapted from https://github.com/airaria/TextBrewer .



## Bias measures

### crows_pair
Contains the code adapted from https://github.com/nyu-mll/crows-pairs to run CrowS-Pairs Bias tests.

### datasets
Contains the datasets for Bias-NLI and Bias-STS.

### seat
Contains the code adapted from https://github.com/W4ngatang/sent-bias to run SEAT.

### evaluation_framework.py
Contains the code to run and evaluate Bias-NLI and Bias-STS, adapted from https://aclanthology.org/2021.findings-emnlp.411/ .

### weat.py
Code to run and evaluate WEAT, adapted from https://github.com/anlausch/XWEAT .
