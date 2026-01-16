# steering-eval
Evaluating steering vectors across diverse tasks.


This repository contains the code and data from [Comparing Bottom-Up and Top-Down Steering Approaches on In-Context Learning Tasks](https://arxiv.org/abs/2310.15213). In particular, this code supports evaluations of Function Vectors [Todd et al.](https://github.com/ericwtodd/function_vectors) and In-Context Vectors [Liu et al.](ttps://github.com/shengliu66/ICV).

## Setup
We suggest creating a conda environment to manage the required packages:
```bash
conda create --name steer-eval python
conda activate steer-eval
pip install -r requirements.txt
```

## Data
All data used in our experiments can be found in `/function_vectors/dataset_files/abstractive`. We broadly categorize datasets into two main categories: functional and behavioral. [See our paper for more information](https://arxiv.org/abs/2310.15213) on tasks we test.

## Evaluation
The scripts used to evaluate steering vectors are `functional_experiment.py` for vanilla functional tasks (zero-shot and shuffle-label few-shot), `natural_text_experiment.py` for out-of-distribution natural text functional tasks, `behavioral_experiment.py` for behavioral tasks, and `ablation_experiment.py` for our ablation experiments. All take the following arguments:
- `--model_name`: Huggingface model name. Supports Llama and Pythia models.
    - `--hf_token`: Optional, huggingface token for gated models
- `--dataset_name`: Dataset file name
- `--run_clean`: Set this flag to evaluate and save clean model performance
- `--run_icv`: Set this flag to evaluate and save ICV-steered performance
- `--run_fv`: Set this flag to evaluate and save FV-steered performance
- `--save_path_root`: Directory to store results
- `--device`: Which device to use (cpu or cuda)
- `--save_cie`: Set this flag to save causal indirect effect calculated for FV extraction
- `--demos`: List of demonstration dataset sizes to use for extracting in-context vectors
    - Usage: `--demos 5 50 100 150`
- `--alphas`: List of in-context vector strengths to evaluate
    - Usage: `--alphas 1.0 1.5`
- `--seeds`: Number of times to run experiment with random seed
- `--specific_seeds`: List of specific seeds to run

You can pass in custom experiment parameters for each type of evaluation script as follows:
- `natural_text_experiment.py`:
    - `--n_eval_templates`: Number of natural text prompt formats to test. Code samples templates randomly without replacement. Max 5 (3 for country-capital task)
- `functional_experiment.py`:
    - `--shots`: List of shots to evaluate over
        - Usage: `--shots 0 3`
- `ablation_experiment.py`: 
    - `--find_layers`: Keyword for layers to edit. Options are:
        "first": first layer of the model
        "last": last layer
        "all": all layers
        "middle": middle layer of model
        "middle2": middle two layers of model
        "middle4": middle four layers of model