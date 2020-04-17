# Robustnness of Deep Learning Models to Low-Level Adversarial Attacks

## Requirements

`tensorflow==2.0.0`

`transformers==2.5.1`





## code/models
contains the models being used in this work

TFRobertaForMultiLabelClassification


## Run roberta train/eval/predict

The following describes how to train/evaluate/predict RoBERTa
This behavior is the same for all three tasks, you just need to replace the run_<task>.py file

For detailed description about the command line flags consult the respective python file (e.g. run_tc.py).

### Training
`
python run_tc.py
--data_dir="data/datasets/tc" 
--model_type=roberta 
--model_name_or_path=roberta-base 
--output_dir="models/RoBERTa/tc" 
--max_seq_length=256
--num_train_epochs=3
--per_device_train_batch_size=28 
--seed=1
--do_train
`

### Evaluation
`
python run_tc.py
--data_dir="data/datasets/tc" 
--model_type=roberta 
--model_name_or_path=roberta-base 
--output_dir="models/RoBERTa/tc" 
--max_seq_length=256
--do_eval
`

### Prediction
`
python run_tc.py
--data_dir="data/datasets/tc" 
--model_type=roberta 
--model_name_or_path=roberta-base 
--output_dir="models/RoBERTa/tc" 
--max_seq_length=256
--do_eval
`