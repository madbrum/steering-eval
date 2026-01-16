import torch, numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM

from icv.common import setup_env, mk_parser
from icv.models import build_model_signature, build_tokenizer, build_model
from icv.tasks import load_task
from icv.utils.logger import tabular_pretty_print
from icv.utils.tools import ensure_folder
from icv.utils.pca import PCA
from icv.utils.llm_layers import get_layers
import numpy as np

class AdapterLayer(torch.nn.Module):
    """
    Adapter layer for adding in-context vector to layer output hidden state.
    From https://github.com/shengliu66/ICV 
    """
    
    def __init__(self, icvs, alpha):
        super(AdapterLayer, self).__init__()
        self.icvs = icvs
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
        input_dtype = x.dtype
        if self.icvs is not None:
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            alpha = self.alpha
            icv_all_tasks = 0
            for i in range(len(self.icvs)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x.float(), self.icvs[i][None,None,:], dim=-1)).unsqueeze(-1)
                icv_all_tasks -= alpha[i] * lambda_sim * F.normalize(self.icvs[i], dim=-1).repeat(1,x.shape[1],1)
            icv_all_tasks = 0.1 * icv_all_tasks/len(self.icvs)
            
            x = F.normalize(F.normalize(x.float(),dim=-1) +  icv_all_tasks, dim=-1) * norm
            return x.type(input_dtype)
        else:
            return x

class model_with_adapter(torch.nn.Module):
    """
    Base model with adapter layers added, for applying in-context vector to desired layers.
    From https://github.com/shengliu66/ICV 
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, icvs, alpha):
        if isinstance(self.model, GPTNeoXForCausalLM):
            inner_model = self.model.gpt_neox
        else:
            inner_model = self.model.model
        
        for i in range(0, len(inner_model.layers)):
            icvs_ = icvs[i]
            inner_model.layers[i].mlp = torch.nn.Sequential(inner_model.layers[i].mlp, AdapterLayer(icvs_, alpha))
        return self.model

    def remove_adapter(self):
        if isinstance(self.model, GPTNeoXForCausalLM):
            inner_model = self.model.gpt_neox
        else:
            inner_model = self.model.model
        
        weight_all = []
        
        for i in range(0, len(inner_model.layers)):
            weight_all.append(inner_model.layers[i].mlp[1].weight_all)
            inner_model.layers[i].mlp = inner_model.layers[i].mlp[0]
        return weight_all


def remove_in_context_vector(model):
    """
    Remove in-context vector from adapted model.
    From https://github.com/shengliu66/ICV 

    Args:
        model (model_with_adapter): The adapter model wrapper.
        edit_layers (list[int]): The list of layer adapters to remove.
    """
    while True:
        try:
            model_with_adapter(model).remove_adapter()
            print('ICV vector is removed\n')
        except:
            print('All ICV vectors have been removed!\n')    
            break

def add_in_context_vector(model, icvs, alpha):
    """
    Add in-context vector to adapted model.
    From https://github.com/shengliu66/ICV 
    
    Args:
        model (model_with_adapter): The adapter model wrapper.
        icvs (list[torch.Tensor]): List of ICVs to apply to the model.
        edit_layers (list[int]): The list of layers to edit.
    """
    remove_in_context_vector(model)
    updated_wrapper = model_with_adapter(model)
    _ = updated_wrapper.get_model(torch.stack(icvs,dim=1).cuda(), alpha = [alpha])
    print('Style vectors have been added!\n') 