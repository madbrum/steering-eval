from transformers import GPTNeoXForCausalLM
from baukit import TraceDict


def add_function_vector(edit_layers, fv_vector, device, idx=-1):
    """
    Adds a vector to the outputs of multiple layers in the model
    Adapted from https://github.com/ericwtodd/function_vectors
    to support multiple edit layers.

    Parameters:
    edit_layers: the layers to perform the FV intervention
    fv_vector: the function vector to add as an intervention
    device: device of the model (cuda gpu or cpu)
    idx: the token index to add the function vector at

    Returns:
    add_act: a fuction specifying how to add a function vector to a layer's output hidden state
    """
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layers:
            if isinstance(output, tuple):
                output[0][:, idx] += fv_vector.to(device)
                return output
            else:
                return output
        else:
            return output

    return add_act


def function_vector_intervention(sentence, target, edit_layers, function_vector, model, model_config, tokenizer, compute_nll=False,
                                  generate_str=False):
    """
    Runs the model on the sentence and adds the function_vector to the output of all layers in edit_layers as a model intervention, predicting a single token.
    Returns the output of the model with and without intervention.
    Adapted from https://github.com/ericwtodd/function_vectors to support multiple edit layers.

    Parameters:
    sentence: the sentence to be run through the model
    target: expected response of the model (str, or [str])
    edit_layers: layers at which to add the function vector
    function_vector: torch vector that triggers execution of a task
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced completion (used to compute perplexity (PPL))
    generate_str: whether to generate a string of tokens or predict a single token

    Returns:
    fvi_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
        nll_targets[:,:-target_len] = -100  # This is the accepted value to skip indices when computing loss (see nn.CrossEntropyLoss default)
        output = model(**nll_inputs, labels=nll_targets)
        clean_nll = output.loss.item()
        clean_output = output.logits[:,original_pred_idx,:]
        intervention_idx = -1 - target_len
    elif generate_str:
        MAX_NEW_TOKENS = 16
        output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, top_p=0.9, temperature=0.1,
                                max_new_tokens=MAX_NEW_TOKENS)
        clean_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        intervention_idx = -1
    else:
        clean_output = model(**inputs).logits[:,-1,:]
        intervention_idx = -1

    # Perform Intervention
    intervention_fn = add_function_vector(edit_layers, function_vector.reshape(1, model_config['resid_dim']), model.device, idx=intervention_idx)
    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:,original_pred_idx,:]
        elif generate_str:
            output = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, top_p=0.9, temperature=0.1,
                                    max_new_tokens=MAX_NEW_TOKENS)
            intervention_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            intervention_output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
    
    fvi_output = (clean_output, intervention_output)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)
    
    return fvi_output 


def find_edit_layers(model, keyword='middle'):
    """Finds edit layer indices given a keyword.
    Options are "first", "last", "all", "middle", "middle2", and "middle4".
    """

    if isinstance(model, GPTNeoXForCausalLM):
        inner_model = model.gpt_neox
    else:
        inner_model = model.model
    
    if keyword == 'first':
        edit_layer = [0]
    elif keyword == 'last':
        edit_layer = [len(inner_model.layers) - 1]
    elif keyword == 'all':
        edit_layer = [i for i in range(len(inner_model.layers))]
    elif keyword == 'middle2':
        middle = len(inner_model.layers) // 2
        edit_layer = [i for i in range(middle - 1, middle + 1)]
    elif keyword == 'middle4':
        middle = len(inner_model.layers) // 2
        edit_layer = [i for i in range(middle - 2, middle + 2)]
    elif keyword == 'middle':
        edit_layer = [len(inner_model.layers) // 2]
    return edit_layer