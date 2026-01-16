"""
Code for running ablation experiments on methods: adjusting edit locations and
number of edit layers 

"""
import os, json
import numpy as np
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm

from function_vectors.src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from function_vectors.src.utils.prompt_utils import load_dataset
from icv.common import setup_env

from utils.eval_utils import avg_scores, evaluate, send_dataset_name
from utils.experiment_utils import (
    setup_icvs,
    setup_icvs_natural_text,
    setup_fv,
    collect_icv_results,
    make_valid_path_name,
    save_results
)
from utils.ablation_utils import find_edit_layers, function_vector_intervention
from utils.setup_utils import setup_test_prompts

torch.set_grad_enabled(False)

def run_ablation_experiment(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name, 
                            icv_path, fv_path, edit_layers, demos, shots, alphas, metrics, n_shots_eval, n_shot_prompt_paths, 
                            seed, find_layer, run_fv, run_icv): 
    model_name_short = model_name.split('/')[-1]
    folder_name = os.path.join(model_name_short, dataset_name, f'seed_{seed}')
    save_path = os.path.join(save_path_root, folder_name)
    os.makedirs(save_path, exist_ok=True)

    icvs_list = None
    if run_icv:
        if icv_path:
            icvs_list, demos = torch.load(icv_path)
        else:
            if dataset_name != 'detoxification' or dataset_name != 'sentiment_transfer':
                icvs_list = setup_icvs(dataset, model, model_config, tokenizer, demos, seed)
            else:
                icvs_list = setup_icvs_natural_text(dataset, model, model_config, tokenizer, demos, seed)
    
    fv = None
    if fv_path:
        fv, _ = torch.load(fv_path)
    elif run_fv:
        fv, edit_layer = setup_fv(dataset, model, model_config, tokenizer, model_name)
        torch.save((fv, edit_layer), os.path.join(save_path, 'fv_and_edit_layer.pt'))

    if not n_shot_prompt_paths:
        if n_shots_eval:
            for n_shots in shots:
                folder_name = os.path.join(model_name_short, dataset_name, f'seed_{seed}', f'{n_shots}shot')
                save_path = os.path.join(save_path_root, folder_name)
                os.makedirs(save_path, exist_ok=True)

                prompt_and_target = setup_test_prompts(dataset, n_shots, model_config=model_config)
                with open(make_valid_path_name(save_path, 'prompts_and_target'), 'w+') as prompts_file:
                    json.dump(prompt_and_target, prompts_file, indent=2)
                
                if icvs_list is not None:
                    #can try multiple but also several at once
                    for n_demos, icvs in zip(demos, icvs_list):
                        collect_icv_results(model, tokenizer, save_path, prompt_and_target, icvs=icvs, alphas=alphas, n_shots=0, n_demos=n_demos, metrics=metrics, edit_layers=edit_layers, seed=seed)
                if fv is not None:
                    collect_fv_results(model, model_config, tokenizer, save_path, prompt_and_target, fv, n_shots, edit_layers, metrics, find_layer)
 
    else: 
        if n_shots_eval:
            #results will be saved in the parent directory of the prompts file 
            for path in n_shot_prompt_paths:
                with open(path, 'r') as prompts_file:
                    prompt_and_target = json.load(prompts_file)
                
                save_path = os.path.join(*path.split('/')[:-1])
                os.makedirs(save_path, exist_ok=True)
                n_shots = path.split('/')[-2]
                n_shots = n_shots[:n_shots.find('shot')]

                if icvs_list is not None:
                    for n_demos, icvs in zip(demos, icvs_list):
                        collect_icv_results(model, tokenizer, save_path, prompt_and_target, icvs=icvs, alphas=alphas, n_shots=0, n_demos=n_demos, metrics=metrics, edit_layers=edit_layers, seed=seed)
                if fv is not None:
                    collect_fv_results(model, model_config, tokenizer, save_path, prompt_and_target, fv, n_shots, edit_layers, metrics, find_layer)
         

def collect_fv_results(model, model_config, tokenizer, save_path, prompt_and_target, fv, n_shots, edit_layers, metrics, find_layer):
    print(f'{n_shots} shot, fv')
    fv_generations, fv_scores, fv_parsed = [], defaultdict(list), []
    for sentence, target in tqdm(prompt_and_target):
        _, fv_intervention_output = function_vector_intervention( #ablation utils override
            sentence, target=target, edit_layers=edit_layers,
            function_vector=fv, model=model, model_config=model_config, 
            tokenizer=tokenizer, compute_nll=False, generate_str=True
        )
        fv_generations.append(fv_intervention_output)
        evaluate(fv_intervention_output, target, metrics, fv_scores, fv_parsed)

    fv_avgs = avg_scores(fv_scores)
    save_results(f'fv_{find_layer}_layers', save_path, fv_avgs, fv_generations, fv_parsed, fv_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--hf_token', help='Huggingface token for gated models', type=str, required=False)
    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results')
    parser.add_argument('--find_layers', help='Target intervention layer to find', nargs="+", required=False, type=str, default=['middle'])
    parser.add_argument('--seeds', help='Number of randomized seeds to average over', type=int, required=False, default=1)
    parser.add_argument('--device', help='Device to run on', type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--demos', help="Demonstration sizes to test", nargs="+", type=int, required=False, default=[50, 100, 150, 5])
    parser.add_argument('--shots', help="List of n shots to test", nargs="+", type=int, required=False, default=[0, 3])
    parser.add_argument('--metrics', help="Metrics to use when evaluating generated string. Choices: f1_score, distinct, first_word_score, present_score, n_gram_entropy, toxicity_score", nargs="+", type=str, required=False, default=['first_word_score', 'f1_score', 'n_gram_entropy', 'distinct'])
    parser.add_argument('--alphas', help="In-context vector strengths to test", nargs="+", type=float, required=False, default=[0.7, 1.0, 1.5])
    parser.add_argument('--specific_seeds', help="Specific seeds to run with", nargs="+", type=int, required=False) 
    parser.add_argument('--icv_path', help="Path to existing ICV list", required=False)
    parser.add_argument('--fv_path', help="Path to existing FV", required=False)
    parser.add_argument('--n_shots_eval', help="Run n-shot evaluations (for rerun)", action='store_true')
    parser.add_argument('--n_shot_prompt_paths', help="Paths to existing prompt_and_target files", nargs="+", type=str, required=False)
    parser.add_argument('--run_icv', help="Run ICV experiments", action='store_true')
    parser.add_argument('--run_fv', help="Run FV experiments", action='store_true')

    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    save_path_root = args.save_path_root
    num_seeds = args.seeds
    device = args.device
    find_layers = args.find_layers
    demos = args.demos
    shots = args.shots
    metrics = args.metrics
    alphas = args.alphas
    hf_token = args.hf_token
    specific_seeds = args.specific_seeds
    n_shots_eval = args.n_shots_eval
    icv_path = args.icv_path
    fv_path = args.fv_path
    n_shot_prompt_paths = args.n_shot_prompt_paths
    run_fv = args.run_fv
    run_icv = args.run_icv

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, hf_token=hf_token)
    
    if specific_seeds is not None:
        seeds = specific_seeds
    else:
        seeds = [np.random.randint(0, 100000) for _ in range(num_seeds)]
    

    group_edit_layers = []
    if find_layers:
        for layer_group in find_layers:
            print('Finding edit layer')
            edit_layers = find_edit_layers(model, layer_group)
            group_edit_layers.append(edit_layers)

    for seed in seeds:
        for find_layer, edit_layers in zip(find_layers, group_edit_layers):
            setup_env(gpu_s=1, seed=seed)
            set_seed(seed)
            send_dataset_name(dataset_name)
            dataset = load_dataset(dataset_name, root_data_dir=r'./function_vectors/dataset_files', seed=seed)

            print(find_layer, edit_layers)
            print(f'Running ablation experiment on {model_name} at layers {edit_layers}')
            run_ablation_experiment(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name, 
                            icv_path, fv_path, edit_layers, demos, shots, alphas, metrics, n_shots_eval, n_shot_prompt_paths, 
                            seed, find_layer, run_fv, run_icv)