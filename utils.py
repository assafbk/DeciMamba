import json
import numpy as np
import torch
import os
import warnings
import wandb
import random

from torch.utils.data import Sampler, Dataset

import matplotlib.pyplot as plt
import matplotlib
import PIL

# used to sample a subset of the val set during training
class SubsetSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))

    def __len__(self):
        return len(self.mask)

class ListDataset(Dataset):
     def __init__(self, original_list):
        self.original_list = original_list
     def __len__(self):
        return len(self.original_list)

     def __getitem__(self, i):
        return self.original_list[i]
     
     
def load_config(args):
    
    if args.eval != -1:
        path = f'./configs/eval_ssm_config_{args.eval}.json'
    else:
        path = './configs/finetune_ssm_config.json'

    f = open(path)
    json_data = json.load(f)
    f.close()

    if args.device != 'None':
        json_data['model_device'] = f'cuda:{args.device}'

    return json_data

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def calc_grad_norm(model):
    grad_sum_sqrd = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_sum_sqrd += torch.sum((param.grad.detach().clone().flatten())**2)

    norm = torch.sqrt(grad_sum_sqrd)
    return norm

'''for each prediction, calculates the entropy (of the predicted distribution over all tokens), and then calculates the mean over all predictions in the batch'''
def calc_mean_entropy(predicted_logits):
    vocab_size = predicted_logits.shape[2]
    probabilities = torch.softmax(predicted_logits.reshape(-1, vocab_size), axis=1)
    prob_zeros_mask = probabilities == 0.
    tmp = probabilities * torch.log2(probabilities) # when a probability equals 0 this gives 0*-inf and torch returns nan. by the entropy definition it should equal 0, so we fix that
    tmp[prob_zeros_mask] = 0.
    if torch.any(torch.isnan(tmp)):
        warnings.warn("Warning: entropy calculation (metric) has nans in it")

    entropy = -torch.sum(tmp, axis=1)
    return torch.mean(entropy)

def init_grad_flow_data(model):
    grad_flow_data = {}
    grad_flow_data['steps'] = []
    for module_name, module in model.named_children():
        layers = []
        for n, p in module.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                if n.startswith('layers.'):
                    layer_num = n.split('.')[1]
                    layer_num = layer_num if len(layer_num)==2 else f'0{layer_num}'
                    layer_name = f'layer_{layer_num}'
                    if layer_name not in layers:
                        grad_flow_data[f'{module_name}/{layer_name}'] = []
                        layers.append(layer_name)
                else:        
                    grad_flow_data[f'{module_name}/{n}'] = []
                    layers.append(n)
    return grad_flow_data

def get_grad_flow_log_format(model, step, grad_flow_data):
    log_dict = {}
    grad_flow_data['steps'].append(step)
    for module_name, module in model.named_children():
        cur_layers, cur_avg_grads, cur_max_grads = _calc_grad_flow(module.named_parameters(), module_name)
        if len(cur_layers) == 0:
            continue
        
        keys = []
        y_vals = []
        for i in range(len(cur_layers)):
            layer_name = cur_layers[i]
            avg_grads = cur_avg_grads[i]
            max_grads = cur_max_grads[i]
            
            # update db
            grad_flow_data[f'{module_name}/{layer_name}'].append(avg_grads)
            # grad_flow_data[f'max_grad/{module_name}/{layer_name}'].append(max_grads)

            # for wandb
            keys.append(layer_name)
            y_vals.append(grad_flow_data[f'{module_name}/{layer_name}'])
        
        # save in wandb structure
        log_dict[module_name] = wandb.plot.line_series(
                       xs=grad_flow_data['steps'], 
                       ys=y_vals,
                       keys=keys,
                       title=f'{module_name} grad flow (normalized by weight values)',
                       xname="steps")

    return log_dict, grad_flow_data

def _calc_grad_flow(named_parameters, module_name, epsilon=1e-13):
    avg_grads = []
    avg_weights = []
    max_grads= []
    layers = []
    norm = 'l2' # 'l1' / 'l2'
    # num_elements_in_layer = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            p_grad = p.grad.cpu()
            p_weight = p.detach().clone().cpu()
            if n.startswith('layers.'): # block x (layers.x) has a few components, we aggregate them
                layer_num = n.split('.')[1]
                layer_num = layer_num if len(layer_num)==2 else f'0{layer_num}'
                if f'layer_{layer_num}' not in layers:
                    layers.append(f'layer_{layer_num}')
                    avg_grads.append(0.)
                    avg_weights.append(0.)
                    max_grads.append(0.)
                    # num_elements_in_layer.append(0)
                # num_elements_in_layer[-1]+=len(p_grad.flatten())
                if norm == 'l2':
                    avg_grads[-1]+=p_grad.square().sum()
                    avg_weights[-1]+=p_weight.square().sum()
                else:
                    avg_grads[-1]+=p_grad.abs().sum()
                    avg_weights[-1]+=p_weight.abs().sum()
                max_grads[-1] = torch.max(torch.Tensor([max_grads[-1], p_grad.abs().max()]))
                
            else:        
                layers.append(n)
                if norm == 'l2':
                    avg_grads.append(p_grad.square().sum())
                    avg_weights.append(p_weight.square().sum())
                else:
                    avg_grads.append(p_grad.abs().sum())
                    avg_weights.append(p_weight.abs().sum())
                max_grads.append(p_grad.abs().max())
                # num_elements_in_layer.append(len(p_grad.flatten()))

    # avg_grads = [avg_grads[i]/num_elements_in_layer[i] for i in range(len(avg_grads))]
    if norm == 'l2':
        avg_grads = [torch.sqrt(avg_grads[i]/(avg_weights[i]+epsilon)) for i in range(len(avg_grads))] # no need to divide by num_elements_in_layer, it cancels out in avg_grad/avg_weight
    else:
        avg_grads = [avg_grads[i]/(avg_weights[i]+epsilon) for i in range(len(avg_grads))] # no need to divide by num_elements_in_layer, it cancels out in avg_grad/avg_weight

    return layers, avg_grads, max_grads

def init_entropy_per_layer_data():
    entropy_data = {}
    entropy_data['steps'] = []
    for layer_idx in range(24):
        layer_num = layer_idx if len(str(layer_idx))==2 else f'0{layer_idx}'
        entropy_data[f'layer_{layer_num}'] = []
    return entropy_data

def get_log_format_for_per_layer_entropy(step, mean_entropy_per_layer, entropy_data):
    log_dict = {}
    entropy_data['steps'].append(step)
    
    keys = []
    y_vals = []
    for layer_idx in range(len(mean_entropy_per_layer)):
        layer_num = layer_idx if len(str(layer_idx))==2 else f'0{layer_idx}'
        layer_name = f'layer_{layer_num}'
        avg_entropy = mean_entropy_per_layer[layer_idx]
        # max_grads = cur_max_grads[i]
        
        # update db
        entropy_data[layer_name].append(avg_entropy)

        # for wandb
        keys.append(layer_name)
        y_vals.append(entropy_data[layer_name])
    
    # save in wandb structure
    log_dict['mean_entropy_per_layer'] = wandb.plot.line_series(
                    xs=entropy_data['steps'], 
                    ys=y_vals,
                    keys=keys,
                    title=f'mean entropy mem activity per layer',
                    xname="steps")

    return log_dict, entropy_data

def convert_niah_array_to_img(niah_array, config):
    fig=plt.figure()
    plt.xticks(list(range(len(config['niah_context_lens_eval']))), config['niah_context_lens_eval'])
    plt.yticks(list(range(len(config['niah_needle_depths_eval']))), config['niah_needle_depths_eval'])
    plt.xlabel('context length [toks]')
    plt.ylabel('needle depth w.r.t context length')
    plt.title('niah map')
    cmap = matplotlib.colors.ListedColormap(['tomato', 'lightgreen'])
    context_len_train = config['niah_context_len_train']
    plt.imshow(niah_array, interpolation='none', cmap=cmap)
    index_train_context_len = config['niah_context_lens_eval'].index(context_len_train)
    plt.axvline(x=index_train_context_len, color='black', linewidth=3)
    plt.annotate(f'train context len = {context_len_train//1000}k',
                xy=(index_train_context_len, 0.8), xycoords='data',
                horizontalalignment='right', verticalalignment='top', rotation=90, fontsize=12)
    fig.canvas.draw()
    niah_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    return niah_img