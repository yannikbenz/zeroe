# Robustnness of Deep Learning Models to Low-Level Adversarial Attacks

Experiments:

**Model**: RoBERTa

**Perturbers**:
* full-shuffle
* inner-shuffle
* intruders
* disemvoweling
* truncate
* segmentation
* keyboard-typos
* natural-typos
* phonetic
* visual

**Tasks (Datasets)**:
* POS tagging (Universal Dependencies)
* Natural Language Inference (SNLI)
* Toxic Comment Classification (kaggle challenge)

## 1. Requirements

We use conda to setup our python environment.

We freezed our environment into the [environment.yml](environment.yml) file 
([further docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)).

Restore it with the following command:

`conda env create -f environment.yml`

The fact that some packages are not available in the conda repository makes 
it necessary to install them manually:

`pip install transformers==2.5.1`

The full requirements are given in the [requirements.txt](requirements.txt)
You can also install them via:
`pip install -r requirements.txt`

````
`conda install numpy pandas scitkit-learn nltk`  
`conda install -c fastai fastprogress`  
`conda install tensorflow-gpu==2.0.0`  (if GPU is available else: `tensorflow==2.0.0`)  
`pip install transformers==2.5.1`  
````

## 2. code/models
contains the models being used in this work


## 3. data

In order to perturb the data we preprocessed each tasks dataset by all our 10 perturbers.


## 4. Run roberta train/eval/predict (experiments)

The following describes how to train/evaluate/predict RoBERTa
This behavior is the same for all three tasks, you just need to replace the run_task.py file

For detailed description about the command line flags consult the respective python file (e.g. run_tc.py).

### Training
```` shell script
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
````

### Evaluation
```` shell script
python run_tc.py
--data_dir="data/datasets/tc" 
--model_type=roberta 
--model_name_or_path=roberta-base 
--output_dir="models/RoBERTa/tc" 
--max_seq_length=256
--do_eval
````

### Prediction
```` shell script
python run_tc.py
--data_dir="data/datasets/tc" 
--model_type=roberta 
--model_name_or_path=roberta-base 
--output_dir="models/RoBERTa/tc" 
--max_seq_length=256
--do_eval
````
