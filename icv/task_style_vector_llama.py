import gc
import json
import logging
import os
import textwrap


import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anchor import logger_root
from common import setup_env, mk_parser, AdvantageLogger
from models import build_model_signature, build_tokenizer, build_model
from models.meta_optimizer import AttnOptimWrapper
from tasks import load_task
from utils.logger import setup_logger, tabular_pretty_print
from utils.tools import ensure_folder
from utils.pca import PCA
import numpy as np

logger = logging.getLogger("task")

class AdapterLayer(torch.nn.Module):

    """
    AdapterLayer implementation.
    """

    def __init__(self, directions, directions_reading, alpha):
        super(AdapterLayer, self).__init__()
        self.directions = directions
        self.directions_reading = directions_reading
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
        if self.directions is not None:
            X = x.float()
            Y = self.directions
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            alpha = self.alpha
            directions_all = []
            y = 0
            for i in range(len(self.directions)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x.float(), self.directions[i][None,None,:], dim=-1)).unsqueeze(-1)
                y -= alpha[i] * lambda_sim * F.normalize(self.directions[i], dim=-1).repeat(1,x.shape[1],1)
            y = y/len(self.directions)
            
            x = F.normalize(F.normalize(x.float(),dim=-1) +  0.1 * y, dim=-1) * norm
            return x.half()
        else:
            return x

class GPT2_with_adapter(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

        self.end_point = 0


        # Embed adapter layers into the transformer blocks 
        
    def get_model(self, directions, directions_reading, alpha):
        for i in range(0, len(self.model.model.layers) + self.end_point):
            directions_ = directions[i]
            directions_reading_ =directions_reading[i]
            self.model.model.layers[i].mlp = torch.nn.Sequential(self.model.model.layers[i].mlp, AdapterLayer(directions_, directions_reading_, alpha))
        return self.model

    def remove_adapter(self):
        
        weight_all = []
        
        for i in range(0, len(self.model.model.layers) + self.end_point):
            weight_all.append(self.model.model.layers[i].mlp[1].weight_all)
            self.model.model.layers[i].mlp = self.model.model.layers[i].mlp[0]
        return weight_all
            

if __name__ == "__main__":
    DEBUG = False
    parser = mk_parser()
    if DEBUG:
        fake_cmd = (
            "--prompt default "
            "--dataset paradetox "
            "--exemplar_method stratified --num_k_shots 1 "
            "--model_type opt --model_size 125m "
            "--batch_size 16 "
            "--seed 0 "
            "--gpus 0 "
            "--in_8bit true"
        )
        args = parser.parse_args(fake_cmd.strip().split())
    else:
        args = parser.parse_args()

    if DEBUG:
        logger_root = logger_root.joinpath("DEBUG")

    logger_root = logger_root.joinpath("main")
    dataset_name = args.dataset
    logger_folder = logger_root.joinpath(dataset_name)

    task_name = f"seed{args.seed}"
    task_name += f"_{args.prompt_version}"
    task_name += f"_{args.model_type}_{args.model_size}"
    task_name += f"_{args.exemplar_method}{'' if args.exemplar_method == 'written' else args.num_k_shots}"
    task_name += f"_stylestrength{args.alpha}"
    task_name += f"_rank{args.rank}"
    
    setup_env(gpu_s=args.gpus, seed=args.seed)
    ensure_folder(logger_folder, parents=True)
    setup_logger(
        logger_folder,
        log_file_name=f"{task_name}.log",
        console_output=not args.no_console,
    )

    logger.info(f"Task Prepared: {task_name}")
    logger.info(f"\tDataset: {dataset_name}")
    logger.info(f"\tLogger save at {logger_folder}")

    # 1. load model, tokenizer
    model_signature = build_model_signature(args.model_type, args.model_size)

    if args.model_type in ['falcon']:
        padding_side = 'right'
    else:
        padding_side = 'right'
    tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side=padding_side)

    model = build_model(args.model_type, args.model_size, args.in_8bit)
    torch.autograd.set_grad_enabled(False)
    logger.info(f"Model loaded: {model_signature}")

    # 2. load dataset (with demonstrations)
    TaskHandler = load_task(dataset_name)
    task_agent = TaskHandler(args.prompt_version)
    task_agent.set_seed(args.seed)
    task_agent.do_load()

    if dataset_name == 'shakespeare' or dataset_name == 'format' or dataset_name == 'emotive':
        prefix = 'Please paraphrase the following sentence. '
    elif dataset_name == 'paradetox' or dataset_name == 'formality':
        prefix='Please paraphrase the following sentence. '
    elif dataset_name =='sentiment':
        prefix='Please paraphrase the following sentence. '
    else:
        raise NotImplementedError

    dataset = task_agent.mk_result_dataset(tokenizer, no_padding=True, prefix=prefix+'. ')

    if args.exemplar_method == "written":
        exemplar_str = task_agent.handcrafted_exemplars()
    elif args.exemplar_method == "random":
        exemplar_str = task_agent.random_selected_exemplars(args.num_k_shots, prefix=prefix)

    elif args.exemplar_method == "stratified":
        exemplar_str = task_agent.stratified_sampling(args.num_k_shots)
    else:
        raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

    text_width = 168
    exemplar_showcase = [["Line", "Text"]]
    for line_idx, line in enumerate(exemplar_str.split("\n")):
        if len(line) > text_width:
            splitted_lines = textwrap.wrap(line, text_width)
            exemplar_showcase.append([str(line_idx + 1), splitted_lines[0]])
            for remained in splitted_lines[1:]:
                exemplar_showcase.append(["", remained])
        else:
            exemplar_showcase.append([str(line_idx + 1), line])

    exemplar_showcase[-1][-1] += "<query starts from here>"
    for line in tabular_pretty_print(exemplar_showcase):
        logger.info(line)


    direction, direction_reading, _, _ = task_agent.obtain_direction(model, dataset.tokenize_each_demonstration(task_agent._cached_ex_list, dataset_name), rank=args.rank)

    direction = direction[1:]
    direction_reading = direction_reading[1:]
    

    logger.info(f"Caculated style vector from context example")
    while True:
        try:
            GPT2_with_adapter(model).remove_adapter()
            print('Style vector is removed\n')
        except:
            print('All style vectors have been removed!\n')    
            break
    updated_wrapper = GPT2_with_adapter(model)
    _ = updated_wrapper.get_model(torch.stack([direction],dim=1).cuda(), torch.stack([direction_reading],dim=1).cuda(), alpha = [args.alpha])
    print('Style vectors have been added!\n')  


    logger.info(f"Selected batch_size: {args.batch_size}")


    loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=2)

    logger.info("Running ...")

    eos_id = tokenizer.encode('only.')[-1]


    with torch.no_grad():
        with open(logger_folder.joinpath(task_name + '.json') , 'w') as f:
            for batch_input in tqdm(loader, desc=f"Evaluation"):
                batch_input_ids = batch_input[0]
                batch_masks = batch_input[1]
                batch_reference = batch_input[2]

                # try:
                generation_output = model.generate(
                    input_ids=batch_input_ids.cuda(),
                    attention_mask=batch_masks.cuda(),
                    max_new_tokens=128,
                    # do_sample=True,
                    # temperature=0.7,
                    # repetition_penalty=1/0.85,
                    # # top_p=0.95,
                    # top_k=40,
                    temperature=0.7,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    num_return_sequences=1,
                    early_stopping=True,

                    eos_token_id=[1642, 13492, 26036, 29908],
                )


                original_input = tokenizer.decode(batch_input_ids[0])

                generation_output = tokenizer.decode(generation_output[0][len(batch_input_ids[0]):], skip_special_tokens=True)

                if len(batch_reference) > 1:
                    ref = batch_reference
                else:
                    ref = batch_reference[0]
                logger.info(f'input: {original_input}, generation: {generation_output}, gold: {ref} \n')
                json.dump({'generation': generation_output,'gold': ref}, f)
                f.write("\n")
                # except:
                #     pass
            f.truncate()

                
    while True:
        try:
            GPT2_with_adapter(model).remove_adapter()
            print('Style vector is removed\n')
        except:
            print('All style vectors have been removed!\n')    
            break
