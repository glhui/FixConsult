
#  Leveraging External Defect Knowledge Driven Natural Language Suggestions Generation for Software Vulnerability Repair

## Requirements
+ importlib-metadata<5.0  
+ spacy==3.0.6  
+ bert-score==0.3.11  
+ datasets==2.5.1  
+ gym==0.21.0
+ jsonlines==3.0.0
+ nltk==3.7
+ pandas==1.3.5
+ rich==12.0.0
+ stable-baselines3==1.5.1a5
+ torch==1.11.0
+ torchvision==0.12.0
+ tqdm==4.64.0
+ transformers==4.18.0
+ wandb==0.12.15
+ jsonlines==3.0.0
+ rouge_score==0.0.4
+ sacrebleu==2.2.0
+ py-rouge==1.1
+ datasets
+ evaluate


## Dataset
We provide the constructed dataset in the directory `data`, which includes 18,517 pairs of vulnerable functions and suggestions, along with the patches. We randomly split them into `train`, `valid`, and `test` datasets with the fraction of 8:1:1. As the filename indicates, for example, files `train.csv` and `valid.csv` contain samples for training and validation respectively. In each file, there are three columns "id", "function", "suggestion", and "patch", which are the vulnerable functions, the corresponding suggestions, and the patches respectively.

Our label data set in the dir data, named with gpt-3.5-turbo.zip, gpt-4o-mini.zip, qwen-max.zip。 and the original data is data.zip

## How to Run
1. `bash train_main.sh` for preprocessing, training and fine-tuning;
2. `bash tira_infer.sh` for inference of our model;
3. `bash evaluation.sh` for getting the results of all metrics.
4. `python gen_vul_labes.py` for label the function with LLM. you should use your own API key.

### Note: modify the data file path
