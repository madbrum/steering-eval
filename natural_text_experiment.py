import os, json
import numpy as np
import argparse

import torch

from function_vectors.src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from function_vectors.src.utils.prompt_utils import load_dataset
from icv.common import setup_env

from utils.experiment_utils import (
    get_random_natural_text_template,
    setup_icvs,
    setup_fv,
    collect_clean_results,
    collect_fv_results,
    collect_icv_results,
    make_valid_path_name,
)
from utils.setup_utils import setup_natural_prompts
from utils.eval_utils import send_dataset_name


torch.set_grad_enabled(False)

def run_natural_text_eval(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name_short, seed, 
                         n_eval_templates, demos, icvs_list, fv, alphas, metrics, run_clean, run_icl, run_icv, run_fv, edit_layer):
    '''
    Formats given dataset into natural text questions, evaluates steering methods on dataset,
    and saves results evaluated on given metrics.

    Parameters:
        See run_evaluation.
        icvs_list: list of icvs, corresponding to each demo set size in demos
        fv: task-specific function vector
        edit_layer: model-specific intervention layer for FVs
    '''
    
    folder_name = os.path.join(model_name_short, dataset_name, f'seed_{seed}', 'natural_text_eval')
    save_path = os.path.join(save_path_root, folder_name)
    os.makedirs(save_path, exist_ok=True)

    natural_templates = get_random_natural_text_template(n_eval_templates, dataset_name)
    templates = []

    for i, template in enumerate(natural_templates):
        templates.append(template)

        if run_icl:
            icl_prompts = setup_natural_prompts(dataset, n_shots=50, model_config=model_config, template=template)
            with open(make_valid_path_name(save_path, 'icl_prompts'), 'w+') as prompts_file:
                json.dump(icl_prompts, prompts_file, indent=2)
            
            collect_clean_results(model, tokenizer, save_path, icl_prompts, metrics=metrics)
            continue

        prompt_and_target_natural = setup_natural_prompts(dataset, n_shots=0, model_config=model_config, template=template)
        with open(make_valid_path_name(save_path, f'prompts_and_target_natural{i}'), 'w+') as natural_prompts_file:
            json.dump(prompt_and_target_natural, natural_prompts_file, indent=2)

        if run_clean: 
            collect_clean_results(model, tokenizer, save_path, prompt_and_target_natural, metrics)

        if run_icv:
            for n_demos, icvs in zip(demos, icvs_list):
                collect_icv_results(model, tokenizer, save_path, prompt_and_target_natural, icvs=icvs, alphas=alphas, n_shots=0, n_demos=n_demos, metrics=metrics, seed=seed)

        if run_fv:
            collect_fv_results(model, model_config, tokenizer, save_path, prompt_and_target_natural, fv, n_shots=0, n_trials=100, edit_layer=edit_layer, metrics=metrics, seed=seed)

    with open(make_valid_path_name(save_path, 'templates'), 'w+') as templates_file:
        json.dump(templates, templates_file, indent=2)

def run_evaluation(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name, demos, alphas, metrics, n_eval_templates, seed, run_clean, run_icl, run_icv, run_fv, last_token_only, save_indirect_effect):
    '''
    Evaluates steering methods on given dataset and saves results evaluated on given metrics.

    Parameters:
        model: Huggingface model
        model_config: dict of model config variables
        dataset: dict containing train/test/validation splits of input/output pair data
        tokenizer: Huggingface tokenizer
        save_path_root: directory to save results
        dataset_name: dataset file name, for saving results
        model_name: model name as it appears on Huggingface, for saving results
        demos: list of demonstration dataset sizes to test
        shots: list of numbers of shots to use in prompt
        alphas: list of ICV vector strengths to use
        metrics: list of evaluation metric names
        n_eval_templates: number of natural text templates to evaluate over
        seed: random seed
        run_clean: whether to run clean model baseline
        run_icl: whether to run UNSHUFFLED few-shot icl baseline (WIP, can't be run with other evals)
        run_icv: whether to run ICV-steered model
        run_fv: whether to run FV-steered model
        save_indirect_effect: whether to save CIE
    '''
    model_name_short = model_name.split('/')[-1]
    folder_name = os.path.join(model_name_short, dataset_name, f'seed_{seed}')
    save_path = os.path.join(save_path_root, folder_name)
    os.makedirs(save_path, exist_ok=True)

    icvs_list = []
    if run_icv:
        icvs_list = setup_icvs(dataset, model, model_config, tokenizer, demos, seed)
        torch.save((icvs_list, demos), os.path.join(save_path, 'icvs_and_info.pt'))
    
    fv, edit_layer = None, None
    if run_fv or save_indirect_effect:
        if save_indirect_effect:
            indirect_effect_path = save_path
        fv, edit_layer = setup_fv(dataset, model, model_config, tokenizer, model_name, last_token_only, indirect_effect_path)
        torch.save((fv, edit_layer), os.path.join(save_path, 'fv_and_edit_layer.pt'))

    run_natural_text_eval(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name_short, seed, n_eval_templates, demos, icvs_list, fv, alphas, metrics, run_clean, run_icl, run_icv, run_fv, edit_layer)


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
    parser.add_argument('--n_eval_templates', help="Number of templates to use for template tests", type=int, required=False, default=1)
    parser.add_argument('--run_clean', help="Run clean experiments", action='store_true')
    parser.add_argument('--run_icl', help="Run 100-shot ICL baseline", action='store_true')
    parser.add_argument('--run_icv', help="Run ICV experiments", action='store_true')
    parser.add_argument('--run_fv', help="Run FV experiments", action='store_true')
    parser.add_argument('--save_cie', help="Save FV causal indirect effect", action='store_true')
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    save_path_root = args.save_path_root
    num_seeds = args.seeds
    device = args.device
    demos = args.demos
    alphas = args.alphas
    n_eval_templates = args.n_eval_templates
    hf_token = args.hf_token
    run_clean = args.run_clean
    run_icl = args.run_icl
    run_icv = args.run_icv
    run_fv = args.run_fv
    specific_seeds = args.specific_seeds
    save_cie = args.save_cie

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, hf_token=hf_token)
    
    if specific_seeds is not None:
        seeds = specific_seeds
    else:
        seeds = [np.random.randint(0, 100000) for _ in range(num_seeds)]

    metrics = ['present_score', 'n_gram_entropy', 'distinct']
    
    for seed in seeds:
        setup_env(gpu_s=1, seed=seed)
        set_seed(seed)
        send_dataset_name(dataset_name)
        dataset = load_dataset(dataset_name, root_data_dir=r'./function_vectors/dataset_files', seed=seed)
        run_evaluation(model, model_config, dataset, tokenizer, save_path_root, dataset_name, model_name, demos=demos, 
                        alphas=alphas, metrics=metrics, n_eval_templates=n_eval_templates, seed=seed, run_clean=run_clean,
                        run_icl=run_icl, run_icv=run_icv, run_fv=run_fv, last_token_only=True, save_indirect_effect=save_cie)
    