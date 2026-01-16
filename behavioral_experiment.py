import os, json
import numpy as np
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm

from function_vectors.src.utils.intervention_utils import function_vector_intervention
from function_vectors.src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from function_vectors.src.utils.prompt_utils import load_dataset
from icv.common import setup_env

from utils.experiment_utils import (
    setup_icvs_natural_text,
    setup_fv,
    collect_clean_results,
    collect_fv_results,
    collect_icv_results,
    make_valid_path_name,
)
from utils.setup_utils import setup_test_prompts
from utils.eval_utils import send_dataset_name

torch.set_grad_enabled(False)

def run_evaluation(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name, demos, alphas, metrics, seed, run_clean, run_icl, run_icv, run_fv, last_token_only, save_indirect_effect):
    model_name_short = model_name.split('/')[-1]
    folder_name = os.path.join(model_name_short, dataset_name, f'seed_{seed}')
    save_path = os.path.join(save_path_root, folder_name)
    os.makedirs(save_path, exist_ok=True)

    if run_icl:
        icl_prompts = setup_test_prompts(dataset, n_shots=50, model_config=model_config, shuffle=False)
        with open(make_valid_path_name(save_path, 'icl_prompts'), 'w+') as prompts_file:
            json.dump(icl_prompts, prompts_file, indent=2)
        
        collect_clean_results(model, tokenizer, save_path, icl_prompts, metrics)
        return

    if run_icv:
        icvs_list = setup_icvs_natural_text(dataset, model, model_config, tokenizer, demos, seed)
        torch.save((icvs_list, demos), os.path.join(save_path, 'icvs_and_info.pt'))
    
    if run_fv or save_indirect_effect:
        if save_indirect_effect:
            indirect_effect_path = save_path
        fv, edit_layer = setup_fv(dataset, model, model_config, tokenizer, model_name, last_token_only, indirect_effect_path)
        torch.save((fv, edit_layer), os.path.join(save_path, 'fv_and_edit_layer.pt'))

    prompt_and_target = setup_test_prompts(dataset, n_shots=0, model_config=model_config)
    with open(make_valid_path_name(save_path, 'prompts_and_target'), 'w+') as prompts_file:
        json.dump(prompt_and_target, prompts_file, indent=2)
    
    if run_clean: 
        collect_clean_results(model, tokenizer, save_path, prompt_and_target, metrics)

    if run_icv:
        for n_demos, icvs in zip(demos, icvs_list):
            collect_icv_results(model, tokenizer, save_path, prompt_and_target, metrics=metrics, icvs=icvs, alphas=alphas, n_demos=n_demos)

    if run_fv:
        collect_fv_results(model, model_config, tokenizer, save_path, prompt_and_target, metrics, fv=fv, edit_layer=edit_layer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--hf_token', help='Huggingface token for gated models', type=str, required=False)
    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results')
    parser.add_argument('--seeds', help='Number of randomized seeds to average over', type=int, required=False, default=1)
    parser.add_argument('--device', help='Device to run on', type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--demos', help="Demonstration sizes to test", nargs="+", type=int, required=False, default=[50, 100, 150, 5])
    parser.add_argument('--alphas', help="In-context vector strengths to test", nargs="+", type=float, required=False, default=[-1.0, 0.7, 1.0, 1.3, 1.5])
    parser.add_argument('--run_clean', help="Run clean experiments", action='store_true')
    parser.add_argument('--run_icl', help="Run 100-shot ICL baseline", action='store_true')
    parser.add_argument('--run_icv', help="Run ICV experiments", action='store_true')
    parser.add_argument('--run_fv', help="Run FV experiments", action='store_true')
    parser.add_argument('--save_cie', help="Save FV causal indirect effect", action='store_true')
    parser.add_argument('--specific_seeds', help="Specific seeds to run with", nargs="+", type=int, required=False) 

    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    save_path_root = args.save_path_root
    num_seeds = args.seeds
    device = args.device
    demos = args.demos
    alphas = args.alphas
    hf_token = args.hf_token
    run_clean = args.run_clean
    run_icl = args.run_icl
    run_icv = args.run_icv
    run_fv = args.run_fv
    save_cie = args.save_cie
    specific_seeds = args.specific_seeds

    
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, hf_token=hf_token)
    
    if specific_seeds is not None:
        seeds = specific_seeds
    else:
        seeds = [np.random.randint(0, 100000) for _ in range(num_seeds)]

    metrics = ['n_gram_entropy', 'distinct']
    if dataset_name == 'sentiment_transfer':
        metrics.insert(0, 'sentiment_transfer')
    if dataset_name == 'detoxification':
        metrics.insert(0, 'toxicity_score')

    for seed in seeds:
        setup_env(gpu_s=1, seed=seed)
        set_seed(seed)
        send_dataset_name(dataset_name)
        dataset = load_dataset(dataset_name, root_data_dir=r'./function_vectors/dataset_files', seed=seed)

        run_evaluation(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name, 
                       demos=demos, alphas=alphas, metrics=metrics, seed=seed, 
                       run_clean=run_clean, run_icl=run_icl, run_icv=run_icv, run_fv=run_fv, last_token_only=True, save_indirect_effect=save_cie)