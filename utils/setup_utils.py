from function_vectors.src.compute_indirect_effect import compute_indirect_effect
from function_vectors.src.utils.extract_utils import compute_function_vector, get_mean_head_activations
from function_vectors.src.utils.prompt_utils import word_pairs_to_prompt_data, create_fewshot_primer, create_prompt
from icv.tasks import load_task
import torch
import os
import numpy as np
import math
import json
from tqdm import tqdm


def get_fv_params(model_name, model_config):
  """
  Calculates hyperparameters to use for FV extraction/intervention:
  (1) intervention layer and (2) number of top causally implicated attention heads to sum over.
  Default values are 11 and 20, based on FV paper.
  
  Args:
    model_name (str): The name of the model to use.
    model_config (dict): The configuration of the model.

  Returns:
    tuple (int, int): intervention layer and the number of top causally implicated attention heads.
  """
  
  if model_name == 'meta-llama/Llama-2-7b-chat-hf':
    return 11, 20
  else:
    edit_layer = math.ceil(model_config['n_layers'] / 3)
    n_top_heads = math.ceil(model_config['n_layers'] * model_config['n_heads'] / 51.2) #factor calculated from FV paper
    return edit_layer, n_top_heads


def setup_function_vector(dataset, model, model_config, tokenizer, n_top_heads=20, n_trials=100, last_token_only=False, indirect_effect_path=None):
  """
  Extracts function vector. Optionally saves CIE calculated as intermediate step.
  
  Args:
    dataset (dict): The dataset to use.
    model (model): Huggingface model to use.
    model_config (dict): The configuration of the model.
    tokenizer (tokenizer): Huggingface tokenizer to use.
    n_top_heads (int): The number of top causally implicated attention heads to sum over.
        If None, all heads are used. Default is 20.
    n_trials (int): The number of trials to run. Default is 100.
    last_token_only (bool): Whether to only use the last token for CIE calculation.
        If True, only the last token of the input is used for CIE calculation.
        Default is False.
    indirect_effect_path (str): The path to save the CIE. Default is None.
  """
  mean_activations = get_mean_head_activations(dataset, model, model_config, tokenizer, N_TRIALS=n_trials)
  indirect_effect = compute_indirect_effect(dataset, mean_activations, model, model_config, tokenizer, last_token_only=last_token_only) #averages over 25 shuffled-label examples 
  if indirect_effect_path:
    torch.save(indirect_effect, os.path.join(indirect_effect_path, 'causal_indirect_effect.pt'))
  fv, top_heads = compute_function_vector(mean_activations, indirect_effect, model, model_config, n_top_heads=n_top_heads)
  return fv


def setup_in_context_vector(dataset, model, model_config, tokenizer, n_demos, seed, preformatted=False):
  """
  Extracts in-context vector. 
  
  Args:
    dataset (dict): The dataset to use.
    model (model): Huggingface model to use.
    model_config (dict): The configuration of the model.
    tokenizer (tokenizer): Huggingface tokenizer to use.
    n_demos (int): The number of demonstrations (contrast pairs) to use.
    seed (int): The seed to use.

  Returns:
    list[torch.Tensor]: The in-context vector.
  """
  TaskHandler = load_task('demo')
  task_agent = TaskHandler('default')
  task_agent.set_seed(seed) #i think this seed doesn't matter here as long as the ds passed in is just the train split, make sure it's same as above tho
  neg_ds = []
  with open('all_truncated_outputs.json', 'r') as dsfile:
    neg_ds = json.load(dsfile)
  if preformatted:
    demo_set = create_demonstration_set_natural_text(dataset['train'], n_demos=n_demos)
  else:
    demo_set = create_demonstration_set(dataset['train'], neg_ds, n_demos=n_demos, model_config=model_config)
  icv = task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, demo_set)) 
  return [icv]


def setup_test_prompts(dataset, n_shots, model_config, shuffle=True, prefixes=None, separators=None):
  """
  Creates prompts for model evaluation from dataset.
  
  Args:
    dataset (dict): The dataset to use.
    n_shots (int): The number of shots to use for few-shot prompt formatting.
    model_config (dict): The configuration of the model.
    shuffle (bool): Whether to shuffle the labels for few-shot prompts. Default is True.
    prefixes (list[str]): Special prefixes to use. Default is None.
    separators (list[str]): Special separators to use. Default is None.

  Returns:
    list[tuple[str, str]]: List of tuples in form (prompt, correct label).
  """
  prompt_and_target = []

  for j in tqdm(range(len(dataset['test'])), total=len(dataset['test'])):
    
    word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)] 
    word_pairs_test = dataset['test'][j]
    if prefixes is not None and separators is not None:
      prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prefixes=prefixes, separators=separators, prepend_bos_token=not model_config['prepend_bos'], shuffle_labels=shuffle)
    else:
      prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=not model_config['prepend_bos'], shuffle_labels=shuffle)
    target = prompt_data['query_target']['output']
    sentence = create_prompt(prompt_data)

    prompt_and_target.append((sentence, target))

  return prompt_and_target


def setup_natural_prompts(dataset, n_shots, model_config, template, shuffle=True):
  """
  Creates prompts in natural text form.

  Args:
    dataset (dict): The dataset to use.
    n_shots (int): The number of shots to use for few-shot prompt formatting (can be 0).
    model_config (dict): The configuration of the model.
    template (str): The template to use for natural text formatting.
    shuffle (bool): Whether to shuffle the labels for few-shot prompts. Default is True.

  Returns:
    list[tuple[str, str]]: List of tuples in form (prompt, correct label).
  """
  prompt_and_target = []

  for j in tqdm(range(len(dataset['test'])), total=len(dataset['test'])):
    if n_shots < 1: #0-shot setting
      q_pair = dataset['test'][j]       
      if isinstance(q_pair['input'], list):
          q_pair['input'] = q_pair['input'][0]
      if isinstance(q_pair['output'], list):
          q_pair['output'] = q_pair['output'][0]

      sentence = template.replace('{X}', f"{q_pair['input']}")       
      prompt_and_target.append((sentence, q_pair['output']))

    else: #few-shot
      word_pairs_raw = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
      templatized_word_pairs = {'input': [template.replace('{X}', f"{w1}") for w1 in word_pairs_raw['input']],
        'output': [w2 for w2 in word_pairs_raw['output']]}
      word_pairs_test = dataset['test'][j]
      templatized_test_pair = {'input': template.replace('{X}', f"{word_pairs_test['input']}"), 'output': word_pairs_test['output']}
      
      prompt_data = word_pairs_to_prompt_data(templatized_word_pairs, query_target_pair = templatized_test_pair,
          prepend_bos_token=not model_config['prepend_bos'], shuffle_labels=shuffle)
      target = prompt_data['query_target']['output']
      sentence = create_prompt(prompt_data)

      prompt_and_target.append((sentence, target))

  return prompt_and_target


def create_demonstration_set(train_ds, neg_ds, n_demos, model_config, seed=None):
  """
  Create demonstration dataset for ICV extraction.
  Returns list of contrast pairs of form (negative example, positive example)
  
  Args:
    train_ds (dict): The training dataset to use.
    neg_ds (dict): The negative dataset to use.
    n_demos (int): The number of demonstrations to create.
    model_config (dict): The configuration of the model.
    seed (int): The seed to use.

  Returns:
    list[tuple[str, str]]: List of tuples in form (negative example, positive example).
  """
  demonstrations = []
  for i in range(n_demos):
    icl_word_pairs = train_ds[np.random.choice(len(train_ds), 1, replace=False)] #num shots is 10
    prompt_data = word_pairs_to_prompt_data(icl_word_pairs, prepend_bos_token=not model_config['prepend_bos'], shuffle_labels=False)
    
    pos = create_fewshot_primer(prompt_data)
    for ex in prompt_data['examples']:
      ex['output'] = neg_ds[np.random.choice(len(neg_ds), replace=False)]
    neg = create_fewshot_primer(prompt_data)
    demonstrations.append((neg, pos))
  return demonstrations


def create_demonstration_set_natural_text(train_ds, n_demos, seed=None):
  """ 
  Create demonstration dataset for ICV extraction when parent dataset is 
  already formatted in natural text.

  Args:
    train_ds (dict): The training dataset to use.
    n_demos (int): The number of demonstrations to create.
    seed (int): The seed to use.

  Returns:
    list[tuple[str, str]]: List of tuples in form (negative example, positive example).
  """
  demonstrations = []
  for i in range(n_demos):
    icl_word_pairs = train_ds[np.random.choice(len(train_ds), 1, replace=False)] #num shots is 10
    demonstrations.append((icl_word_pairs['input'][0], icl_word_pairs['output'][0]))
  return demonstrations


def tokenize_each_demonstration(tok, demonstration_list):
  """
  Tokenizes each demonstration in the list, stripping whitespace and punctuation.
  From https://github.com/shengliu66/ICV/.

  Args:
    tok (tokenizer): Huggingface tokenizer to use.
    demonstration_list (list[tuple[str, str]]): List of tuples in form (negative example, positive example).

  Returns:
    list[tuple[torch.Tensor, torch.Tensor]]: List of tuples in form (tokenized negative example, tokenized positive example).
  """
  tokenized_demonstration_list = []
  for exp_id in range(len(demonstration_list)):
    demonstration_list[exp_id] = (demonstration_list[exp_id][0].strip(" .").strip("."), demonstration_list[exp_id][1].strip(" .").strip("."))

    e_original = tok(demonstration_list[exp_id][0]) 
    e_rewrite = tok(demonstration_list[exp_id][1])
    tokenized_demonstration_list.append((e_original, e_rewrite)) 
  return tokenized_demonstration_list