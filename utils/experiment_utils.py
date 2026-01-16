import itertools
from collections import defaultdict
import json
import os
from tqdm import tqdm
import numpy as np
from function_vectors.src.utils.intervention_utils import function_vector_intervention

from utils.eval_utils import avg_scores, batched_generate, evaluate
from utils.icv_utils import add_in_context_vector, remove_in_context_vector
from utils.setup_utils import get_fv_params, setup_function_vector, setup_in_context_vector


def get_random_natural_text_template(n_eval_templates, dataset_name):
    """
    Get random natural text templates for prompting.
    Derived from https://github.com/ericwtodd/function_vectors.
    """
    
    if dataset_name in ['country-capital']:
        sentences = ["A couple years ago I visited {X}, and",
                     "If you ever travel to {X}, you have to visit",   
                     "When you think of {X},"]
    else:
        sentences = ["The word \"{X}\", means", 
                     "When I think of the word \"{X}\", it usually means",
                     "When I think of \"{X}\", I usually",
                     "While reading a book, I came across the word \"{X}\". I looked it up in a dictionary and it turns out that it means",
                     "The word \"{X}\" can be understood as a synonym for"]
    return np.array(sentences)[np.random.choice(np.arange(len(sentences)), n_eval_templates, replace=False)]


def get_random_templates(n_eval_templates):
    """
    Used for portability experiment for QA prompting, not included in paper.
    Derived from https://github.com/ericwtodd/function_vectors.
    """
    all_prefixes = [{'input': 'A:', 'output': 'B:', 'instructions': ''},
                    {'input': 'input:', 'output': 'output:', 'instructions': ''},
                    {'input': 'Input:', 'output': 'Output:', 'instructions': ''},
                    {'input': 'In:', 'output': 'Out:', 'instructions': ''},
                    {'input': 'question:', 'output': 'answer:', 'instructions': ''},
                    {'input': 'Question:', 'output': 'Answer:', 'instructions': ''},
                    {'input': '', 'output': ' ->', 'instructions': ''},
                    {'input': '', 'output': ' :', 'instructions': ''},
                    {'input': 'text:', 'output': 'label:', 'instructions': ''},
                    {'input': 'x:', 'output': 'f(x):', 'instructions': ''},
                    {'input': 'x:', 'output': 'y:', 'instructions': ''},
                    {'input': 'X:', 'output': 'Y:', 'instructions': ''}]

    all_separators=[{'input': ' ', 'output': '', 'instructions': ''},
                    {'input': ' ', 'output': '\n', 'instructions': ''},
                    {'input': ' ', 'output': '\n\n', 'instructions': ''},
                    {'input': '\n', 'output': '\n', 'instructions': ''},
                    {'input': '\n', 'output': '\n\n', 'instructions': ''},
                    {'input': '\n\n', 'output': '\n\n', 'instructions': ''},
                    {'input': ' ', 'output': '|', 'instructions': ''},
                    {'input': '\n', 'output': '|', 'instructions': ''},
                    {'input': '|', 'output': '\n', 'instructions': ''},
                    {'input': '|', 'output': '\n\n', 'instructions': ''}]

    all_combinations = list(itertools.product(all_prefixes, all_separators))
    random_combos = [list(x) for x in np.array(all_combinations)[np.random.choice(np.arange(len(all_combinations)), n_eval_templates, replace=False)]]
    return random_combos


def setup_icvs(dataset, model, model_config, tokenizer, demos, seed):
    """
    Extracts and returns list of ICVs, one for each demonstration dataset size. 

    Args:
        dataset: dict containing train/test/validation splits of input/output pair data
        model: Huggingface model
        model_config: dict of model config variables
        tokenizer: Huggingface tokenizer
        demos: List of demonstration dataset sizes to test
        seed: Random seed
    
    Returns:
        list of ICVs corresponding to demonstration set sizes 
    """
    icvs_list = []
    for n_demos in demos:
        icvs = setup_in_context_vector(dataset, model, model_config, tokenizer, n_demos, seed)
        icvs_list.append(icvs)
    return icvs_list


def setup_icvs_natural_text(dataset, model, model_config, tokenizer, demos, seed):
    """
    Extracts and returns list of ICVs from natural text demonstrations,
    one for each demonstration dataset size. 

    Args:
        dataset: dict containing train/test/validation splits of input/output pair data
        model: Huggingface model
        model_config: dict of model config variables
        tokenizer: Huggingface tokenizer
        demos: List of demonstration dataset sizes to test
        seed: Random seed
    
    Returns:
        list of ICVs corresponding to demonstration set sizes 
    """
    icvs_list = []
    for n_demos in demos:
        icvs = setup_in_context_vector(dataset, model, model_config, tokenizer, n_demos, seed, preformatted=True)
        icvs_list.append(icvs)
    return icvs_list


def setup_fv(dataset, model, model_config, tokenizer, model_name, indirect_effect_path=None):
    """
    Extracts and returns function vector and L/3 intervention layer.

    Parameters: 
        dataset: dict containing train/test/validation splits of input/output pair data
        model: Huggingface model
        model_config: dict of model config variables
        tokenizer: Huggingface tokenizer
        model_name: model name as it appears on Huggingface
        indirect_effect_path: Optional, location to save CIE
    
    Returns:
        tuple containing function vector, model-specific intervention layer number
    """
    edit_layer, n_top_heads = get_fv_params(model_name, model_config)
    fv = setup_function_vector(dataset, model, model_config, tokenizer, n_top_heads=n_top_heads, indirect_effect_path=indirect_effect_path)
    return fv, edit_layer


def make_valid_path_name(root: str, file_name: str):
    """
    Returns an updated path name if given name already exists.
    From https://github.com/ericwtodd/function_vectors.
    """
    counter = 1
    extension = '.json'
    path = os.path.join(root, file_name + extension)

    while os.path.exists(path):
        path = os.path.join(root, file_name + f"_({counter})" + extension)
        counter += 1

    return path


def save_results(file_name, save_path, avgs, generations, parsed, scores):
    """
    Saves all results to file.
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        with open(make_valid_path_name(save_path, file_name), 'w+') as f:
            json.dump({'avgs': avgs, 'generations': generations, 'parsed': parsed, 'scores': scores}, f, indent=2)
    except IOError as e:
        print(f"Error saving results to {file_name}: {e}")


def process_batch(input_batch, targets_batch, generation_func, metrics):
    """Generate outputs given a batch of inputs and evaluate."""
    outputs = generation_func(input_batch)
    scores = defaultdict(list)
    parsed = []
    for output, target in zip(outputs, targets_batch):
        evaluate(output, target, metrics, scores, parsed)
    return outputs, scores, parsed


def collect_clean_results(model, tokenizer, save_path, prompt_and_target, n_shots, metrics):
    """Evaluate clean model generations over dataset and save results (batched implementation)"""
    batch_size = 32
    sentences_targets = list(zip(*prompt_and_target))
    targets = sentences_targets[1]
    sentences = sentences_targets[0]

    print(f'{n_shots} shot, clean')
    clean_generations, clean_scores, clean_parsed = [], defaultdict(list), []
    for j in tqdm(range(0, len(sentences_targets[0]), batch_size), total=len(sentences_targets[0])//batch_size):
        gen, scores, parsed = process_batch(
            sentences[j:j+batch_size], 
            targets[j:j+batch_size], 
            lambda x: batched_generate(x, model, tokenizer),
            metrics
        )
        clean_generations.extend(gen)
        for k, v in scores.items():
            clean_scores[k].extend(v)
        clean_parsed.extend(parsed)
    
    clean_avgs = avg_scores(clean_scores)
    save_results('clean', save_path, clean_avgs, clean_generations, clean_parsed, clean_scores)


def collect_icv_results(model, tokenizer, save_path, prompt_and_target, icvs, alphas, n_shots, n_demos, metrics):
    """Evaluate ICV-steered model generations over dataset and save results (batched implementation)"""
    batch_size = 32
    sentences_targets = list(zip(*prompt_and_target))
    targets = sentences_targets[1]
    sentences = sentences_targets[0]

    for alpha in alphas:
        print(f'{n_shots} shot, {alpha} icv')
        add_in_context_vector(model, icvs, alpha=alpha)
        
        icv_generations, icv_scores, icv_parsed = [], defaultdict(list), []
        for j in tqdm(range(0, len(sentences_targets[0]), batch_size), total=len(sentences_targets[0])//batch_size):
            gen, scores, parsed = process_batch(
                sentences[j:j+batch_size], 
                targets[j:j+batch_size], 
                lambda x: batched_generate(x, model, tokenizer),
                metrics
            )
            icv_generations.extend(gen)
            for k, v in scores.items():
                icv_scores[k].extend(v)
            icv_parsed.extend(parsed)

        remove_in_context_vector(model)
        icv_avgs = avg_scores(icv_scores)
        save_results(f'icv_{alpha}_{n_demos}_demos', save_path, icv_avgs, icv_generations, icv_parsed, icv_scores)


def collect_fv_results(model, model_config, tokenizer, save_path, prompt_and_target, fv, n_shots, n_trials, edit_layer, metrics):
    """Evaluate FV-steered model generations over dataset and save results"""
    print(f'{n_shots} shot, fv')
    fv_generations, fv_scores, fv_parsed = [], defaultdict(list), []
    for sentence, target in tqdm(prompt_and_target):
        _, fv_intervention_output = function_vector_intervention(
            sentence, target=target, edit_layer=edit_layer,
            function_vector=fv, model=model, model_config=model_config, 
            tokenizer=tokenizer, compute_nll=False, generate_str=True
        )
        fv_generations.append(fv_intervention_output)
        evaluate(fv_intervention_output, target, metrics, fv_scores, fv_parsed)

    fv_avgs = avg_scores(fv_scores)
    save_results(f'fv_{n_trials}_trials', save_path, fv_avgs, fv_generations, fv_parsed, fv_scores)
