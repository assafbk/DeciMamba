from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, DecimatingMambaModel

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import json
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import pickle
import argparse
from tabulate import tabulate

from submodules.babilong.babilong_utils import TaskDatasetCustom, SentenceSampler, NoiseInjectionDataset

from utils import *
from custom_datasets.pg19 import *

def clean_up(start_datetime_str):
    print('\nrunning clean up\n')
    tmp_dir_path = f'./tmp/{start_datetime_str}'
    if os.path.exists(tmp_dir_path):
        shutil.rmtree(tmp_dir_path)

def collate_fn_squad(data):
    batch = {}
    batch['size'] = len(data)
    batch['inputs'] = [f'{elem["question"]}\n\n{elem["context"]}' for elem in data]
    batch['outputs'] = [elem['answers']['text'] for elem in data]
    batch['ids'] = [elem['id'] for elem in data]
    batch['titles'] = [elem['title'] for elem in data]
    batch['questions'] = [elem["question"] for elem in data]
    batch['contexts'] = [elem["context"] for elem in data]
    return batch

def collate_fn_niah(data):
    batch = {}
    batch['size'] = len(data)
    batch['question'] = [elem['question'] for elem in data]
    batch['answer'] = [elem['answer'] for elem in data]
    batch['question_tokens'] = [torch.tensor([elem['question_tokens']]) for elem in data]
    batch['context_tokens'] = [torch.tensor([elem['input_tokens']]) for elem in data]
    batch['target_tokens'] = [torch.tensor([elem['target_tokens']]) for elem in data]
    batch['needle_position'] = [elem['needle_position'] for elem in data]
    return batch

'''this collate function uses a dataset that was tokenized before training'''
def collate_fn_ppl_test_3(data, seq_len=-1, pred_len=1):
    batch = {}
    batch['size'] = len(data)
    begin_locs = [random.sample(range(elem['input_ids'].shape[1] - seq_len),1)[0] if elem['input_ids'].shape[1]>=seq_len else 0 for elem in data] # we will just take the whole thing if we dont have more than seq_len words
    batch['inputs'] = [elem['input_ids'][:,begin_locs[i]:begin_locs[i]+seq_len] for i, elem in enumerate(data)]
    batch['outputs'] = [elem[:,1:] for elem in batch['inputs']]
    return batch

def get_lr_scheduler(config, optimizer, train_set_len, batch_size):
    num_steps_in_epoch = train_set_len/batch_size
    if config['lr_sched_type'] == "cosine_annealing":
        lr_sched = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=5*num_steps_in_epoch, cycle_mult=1.0, max_lr=config['lr'], min_lr=1e-5, warmup_steps=num_steps_in_epoch, gamma=1.0)
    elif config['lr_sched_type'] == "const":
        lambda1 = lambda epoch: 1
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise(f'lr_sched_type {config["lr_sched_type"]} not supported')
    
    return lr_sched

def save_model(config, model, model_processor, epoch, step, start_datetime_str, best_model=False):
    output_dir = os.path.join(config['output_dir'], f'{start_datetime_str}_{config["mamba_arch"]}_{config["model_type"]}_{config["dataset"]}_refs_seed_{config["seed"]}/')
    if best_model:
        ckpt = os.path.join(output_dir, f'best_model')
    else:
        ckpt = os.path.join(output_dir, f'epoch_{epoch}_step_{step}')
    model.save_pretrained(ckpt)
    model_processor.save_pretrained(ckpt)
    with open(os.path.join(config['output_dir'],ckpt,'fintune_ssm_config.json'), 'w') as f:
        json.dump(config, f)

def get_data_loaders(config, model_processor=None, final_eval_mode=False):
    if config['dataset'] == 'niah_custom':
        return get_data_loaders_babilong(config, model_processor, final_eval_mode)

    if config['dataset'] == 'ppl_test':
        return get_data_loaders_ppl_test(config, final_eval_mode)

    if config['dataset'].startswith('squad'):
        return get_data_loaders_squad(config, final_eval_mode)

def get_data_loaders_squad(config, final_eval_mode=False):
    with open('./custom_datasets/multidoc_squad/has_answer_indices_train.pkl', 'rb') as f:
        train_indices_list = pickle.load(f)
    with open('./custom_datasets/multidoc_squad/has_answer_indices_val.pkl', 'rb') as f:
        validation_indices_list = pickle.load(f)

    USER_AGENT = get_datasets_user_agent()
    dataset = load_dataset("rajpurkar/squad_v2", cache_dir=config['cache_dir'])
    dataset['validation'] = dataset['validation'].select(validation_indices_list)

    if final_eval_mode:
        data_loader_val = DataLoader(dataset['validation'], collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0)
        return data_loader_val
    
    else:
        dataset['train'] = dataset['train'].select(train_indices_list)
        dataset_smaller_val_split = dataset['validation'].train_test_split(test_size=config["eval_set_size"]/dataset['validation'].num_rows, seed=111) # we set the same seed because we do want to shuffle the dataset before selecting the val set, but want it to be consistent every time.
        
        data_loader_train = DataLoader(dataset['train'], collate_fn=collate_fn_squad, batch_size=config["grad_accum_steps"], shuffle=True, num_workers=0)
        data_loader_val = DataLoader(dataset_smaller_val_split['test'], collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0)
        return data_loader_train, data_loader_val

def get_noise_data_loader_squad(config):
    USER_AGENT = get_datasets_user_agent()
    with open('./custom_datasets/multidoc_squad/noise_docs_indices_train.pkl', 'rb') as f:
        noise_dataset_shuffled_indices_train = pickle.load(f)
    dataset_train = load_dataset("rajpurkar/squad_v2", split="train", cache_dir=config['cache_dir'])
    shuffled_noise_dataset_train = dataset_train.select(noise_dataset_shuffled_indices_train.tolist()*config['multidoc_num_noise_docs_train']*3)
    noise_data_loader_train = DataLoader(shuffled_noise_dataset_train, collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0).__iter__()
    return noise_data_loader_train

def squad_recovery_run_loaders_to_cp(config, step_in_epoch, data_loader_train, squad_noise_data_loader):
    if config['recover_step'] > step_in_epoch:
        if step_in_epoch == 0:
            print(f'starting recovery to step {config["recover_step"]}')
        [squad_noise_data_loader.__next__() for i in range(config['multidoc_num_noise_docs_train'])]
        return True # advance the train dataloader outside the function
    else:
        return False

def get_data_loaders_ppl_test(config, final_eval_mode=False):
    dataset_train, _ = get_pg19() 

    data_loader_val = None
    if final_eval_mode:
        return data_loader_val
    
    data_loader_train = DataLoader(dataset_train, collate_fn=lambda d: collate_fn_ppl_test_3(d, config['ppl_test_context_len_train'], config['ppl_test_context_len_train']-1), batch_size=config["grad_accum_steps"], shuffle=True, num_workers=0)
    return data_loader_train, data_loader_val
    
def get_data_loaders_babilong(config, model_processor, final_eval_mode=False):
    USER_AGENT = get_datasets_user_agent()
    if config['dataset'] == 'niah_custom':
        train_path = "submodules/babilong/data/codes/codes_train.txt"
        test_path = "submodules/babilong/data/codes/codes_test.txt"
        noise_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=config['cache_dir'])
        noise_sampler_train = SentenceSampler(noise_dataset['train'], tokenizer=model_processor)
        noise_sampler_test = SentenceSampler(noise_dataset['test'], tokenizer=model_processor)
        context_lens_train = config['niah_context_len_train']               # max number of tokens in sample

    else:
        raise(f'{config["dataset"]} dataset is unsupported')
    
    niah_datasets_val = []
    pct_delta = 0.1
    for needle_depth in config['niah_needle_depths_eval']:
        for context_len in config['niah_context_lens_eval']:
            cur_task_dataset_test = TaskDatasetCustom(test_path, max_len=1) # 1 try per depth, we init in each loop so we can get a random init of the key
            niah_datasets_val.append(NoiseInjectionDataset(task_dataset=cur_task_dataset_test,
                                noise_sampler=noise_sampler_test,
                                tokenizer=model_processor,
                                sample_size=context_len,
                                task_start_pct = max(0,needle_depth-pct_delta),
                                task_end_pct = min(1, needle_depth+pct_delta)))
    
    dataset_val = torch.utils.data.ConcatDataset(niah_datasets_val)
    
    if final_eval_mode:
        data_loader_val = DataLoader(dataset_val, collate_fn=collate_fn_niah, batch_size=1, shuffle=False, num_workers=0)
        return data_loader_val
    
    else:
        task_dataset_train = TaskDatasetCustom(train_path, max_len=config['niah_train_set_size'])
        dataset_train = NoiseInjectionDataset(task_dataset=task_dataset_train,
                                              noise_sampler=noise_sampler_train,
                                              tokenizer=model_processor,
                                              sample_size=context_lens_train)
        
        data_loader_train = DataLoader(dataset_train, collate_fn=collate_fn_niah, batch_size=config["grad_accum_steps"], shuffle=True, num_workers=0)
        data_loader_val = DataLoader(dataset_val, collate_fn=collate_fn_niah, batch_size=1, shuffle=False, num_workers=0)
        return data_loader_train, data_loader_val

def load_model(config):

    wanted_dtype = torch.float32
    model_processor = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if config['mamba_arch'] == 'deci':
        mamba_model_class = DecimatingMambaModel
    elif config['mamba_arch'] == 'vanilla':
        mamba_model_class = MambaLMHeadModel
    else:
        raise(f'bad mamba architecture: {config["mamba_arch"]}')
    
    decimation_config = get_decimation_config(config)

    if config['load_cp'] is not None:
        print(f'loading model from checkpoint: {config["load_cp"]}')
        model = mamba_model_class.from_pretrained(config['load_cp'], device=config['model_device'], dtype=wanted_dtype, cache_dir=config['cache_dir'], decimation_config=decimation_config)
    else:
        model = mamba_model_class.from_pretrained(f'state-spaces/{config["model_type"]}', device=config['model_device'], dtype=wanted_dtype, cache_dir=config['cache_dir'], decimation_config=decimation_config)

    return model_processor, model

def run_squad_retrieve_evaluator(pred_dicts, config, start_datetime_str):
    scores_per_num_noise_docs = []
    for i, pred_dict in enumerate(pred_dicts):
        cur_score = 0
        for pred in pred_dict['results']:
            if pred['pred'] == pred['gt']:
                cur_score += 1
        scores_per_num_noise_docs.append(cur_score/len(pred_dict['results']))

    return {'score': np.mean(scores_per_num_noise_docs), 'scores_per_num_noise_docs': scores_per_num_noise_docs}

def inject_noise_to_context(config, golden_doc, noise_data_loader, idx, num_noise_docs, query, is_eval=False):
    
    # get golden doc location
    if config['multidoc_noise_injection_policy'] == 'random_loc':
        if is_eval:
            with open('./custom_datasets/multidoc_squad/random_indices_for_num_docs_before_golden_val.pkl', 'rb') as f:
                noise_dataset_shuffled_indices = pickle.load(f)
        else:
            with open('./custom_datasets/multidoc_squad/random_indices_for_num_docs_before_golden_train.pkl', 'rb') as f:
                noise_dataset_shuffled_indices = pickle.load(f)
        num_docs_before_golden = noise_dataset_shuffled_indices[idx] % (num_noise_docs+1)
    else:
        num_docs_before_golden = 0

    # sample noise and inject doc
    noise_docs_before_golden = []
    for s in range(num_docs_before_golden):
        noise_docs_before_golden.append(noise_data_loader.__next__()['contexts'][0])
    
    noise_docs_after_golden = []
    for s in range(num_noise_docs-num_docs_before_golden):
        noise_docs_after_golden.append(noise_data_loader.__next__()['contexts'][0])
    
    all_docs = noise_docs_before_golden + [golden_doc] + noise_docs_after_golden

    noisy_context = ''
    doc_ids = random.sample(range(0, 1000), num_noise_docs+1)
    for i_doc, doc in enumerate(all_docs):
        noisy_context += f' <|Query|> {query} <|Document {doc_ids[i_doc]}|> {doc}'
    
    return noisy_context, doc_ids[i_doc]

def squad_update_id(pred, i, config):
    pred['id'] = f'{pred["id"]}_{config["multidoc_num_noise_docs_eval"][i]}'
    return pred

def run_niah_evaluator(responses, gts, config):
    res_flat = []
    for i in range(len(responses)):
        cur_response = responses[i].split('<|endoftext|>')[0].split(' ')
        cur_score = gts[i] in cur_response
        res_flat.append(cur_score)
    score = np.sum(res_flat)/len(res_flat)
    niah_map = np.reshape(res_flat, [len(config['niah_needle_depths_eval']),len(config['niah_context_lens_eval'])])
    niah_map_str = '\n'.join('\t'.join(f'{"v" if x else "-"}' for x in y) for y in niah_map)
    
    score2str = np.vectorize(lambda x: 'v' if x else '-')
    print(tabulate(np.hstack([np.array([config['niah_needle_depths_eval']]).T, score2str(niah_map)]), headers=['Depth / Ctx Len'] + [f'{x//1000}K' for x in config['niah_context_lens_eval']] , tablefmt='pretty'))
    return {'score':score, 'niah_map': niah_map_str}

def get_input_ids_and_labels_train(batch, i, model_processor, config, epoch, noise_data_loader=None):
    if config["dataset"].startswith('squad'):
        noisy_context, golden_doc_id = inject_noise_to_context(config, batch["contexts"][i], noise_data_loader, i, config['multidoc_num_noise_docs_train'], batch["questions"][i])
        whole_sequence = f'{noisy_context} <|Answer|> <|Document {golden_doc_id}|>'
        inputs = model_processor(text=whole_sequence, return_tensors="pt")
        labels = model_processor(text=f'{golden_doc_id}|>', return_tensors="pt")
        input_tokens = inputs['input_ids'].to(config['model_device'])
        labels_tokens = labels['input_ids'][0].to(config['model_device'])
    
    if config["dataset"].startswith('niah'):
        question_tokens = batch['question_tokens'][i]
        context_tokens = batch['context_tokens'][i]
        question_post_context_tokens = model_processor(text='\nAnswer: ', return_tensors="pt").input_ids
        labels_tokens = batch['target_tokens'][i]
        input_tokens = torch.cat([question_tokens, context_tokens, question_post_context_tokens, labels_tokens], dim=1).to(config['model_device'])
        labels_tokens = labels_tokens[0].to(config['model_device'])
    
    if config["dataset"].startswith('ppl_test'):
        input_tokens = batch['inputs'][i].to(config['model_device'])
        labels_tokens = batch['outputs'][i][0].to(config['model_device'])
    
    return input_tokens, labels_tokens

def get_input_ids_eval_squad(batch, model_processor, config, noise_data_loader, num_noise_docs, i):
    prompt, golden_doc_id = inject_noise_to_context(config, batch["contexts"][0], noise_data_loader, i, num_noise_docs, batch["questions"][0], is_eval=True)
    prompt = prompt + ' <|Answer|> <|Document '
    input_ids = model_processor(text=prompt, return_tensors="pt").input_ids.to(config['model_device'])
    return input_ids, prompt, golden_doc_id

def get_input_ids_eval(batch, model_processor, config):

    if config["dataset"].startswith('niah'):
        prompt = None
        question_tokens = batch['question_tokens'][0]
        context_tokens = batch['context_tokens'][0]
        question_post_context_tokens = model_processor(text='\nAnswer: ', return_tensors="pt").input_ids
        input_ids = torch.cat([question_tokens, context_tokens, question_post_context_tokens], dim=1).to(config['model_device'])
    
    if config["dataset"].startswith('ppl_test'):
        prompt = None
        input_ids = None

    return input_ids, prompt

def update_results_eval(pred_dict, samples_df_list, batch, idx, epoch, cur_step, response, prompt, squad_num_noise_docs=None):
    if config["dataset"] == 'squad_retrieve':
        pred_dict['results'].append({'id': batch["ids"][0], 'pred': response, 'gt': batch['outputs'][0]})
        samples_df_list.append({'id':batch["ids"][0], 'epoch':epoch, 'step':cur_step, 'num_noise_docs':squad_num_noise_docs, 'prompt':prompt[0:200], 'response':response, 'gt':batch['outputs'][0]})
    if config["dataset"].startswith('niah'):
        pred_dict[idx] = response
        ctx_len = batch['context_tokens'][0].shape[1]
        needle_depth = batch['needle_position'][0]/ctx_len
        samples_df_list.append({'id':idx, 'epoch':epoch, 'step':cur_step, 'response':response, 'gt':batch['answer'][0], 'ctx_len':ctx_len, 'needle_depth':f'{needle_depth:.0%}'})
    
    return pred_dict, samples_df_list

def run_evaluator(pred_dict, samples_df, config, start_datetime_str):
    if config["dataset"].startswith('niah'):
        evaluator_response = run_niah_evaluator(samples_df['response'].to_list(), samples_df['gt'].to_list(), config)
    if config["dataset"] == 'squad_retrieve':
        evaluator_response = run_squad_retrieve_evaluator(pred_dict, config, start_datetime_str)
    
    return evaluator_response

def evaluate_validation_set_ppl_test(model, model_processor, data_loader_val, config, epoch, cur_step, start_datetime_str, num_samples_to_log=None):
    minimal_stride = 10
    max_amount_of_windows = config['ppl_test_num_windows_per_context_len_eval']
    ce_loss = CrossEntropyLoss()

    dataset_val = get_pg19(val_only=True) 
    context_lengths = config['ppl_test_context_lens_eval']
    ppl_per_context_length = []
    for i_ctx_len, window_size in enumerate(context_lengths):
        nlls = []
        trg_len = config['ppl_test_pred_len']
        print(f'testing perplexity with context length of {window_size}, windows per sample = {max_amount_of_windows}, {trg_len} labels per window')
        for i, sample in enumerate(tqdm(dataset_val)):    
            seq_len = sample['input_ids'].size(1)
            if seq_len < window_size:
                print(f'skipping sample {i}, seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')
            
            stride = (seq_len-window_size)//max_amount_of_windows
            if stride < minimal_stride:
                stride = minimal_stride

            for begin_loc in range(0, seq_len-window_size, stride):
                end_loc = begin_loc + window_size
                input_ids = sample['input_ids'][:, begin_loc:end_loc].to(config['model_device'])
                target_ids = input_ids.clone()

                with torch.no_grad():
                    target_ids = target_ids[:, -trg_len:]
                    outputs, params_for_debug = model(input_ids, num_last_tokens=trg_len+1) # FIXME i added the +1 here, see if it makes sense
                    logits = outputs.logits
                    neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())

                nlls.append(neg_log_likelihood)

                if end_loc == seq_len:
                    break

        ppl = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
        print(f'calculated perplexity: {ppl:.2f}')
        ppl_per_context_length.append(ppl)

    val_log = {}
    val_log['score'] = np.mean(ppl_per_context_length)
    ppl_per_context_length_str = '\t'.join(f'{x:.2f}' for x in ppl_per_context_length)
    val_log['ppl_per_ctx_len'] = {'epoch':epoch, 'step':cur_step, 'ppl_per_context_length': ppl_per_context_length_str}
    samples_df = []
    print(tabulate([['score:'] + [f'{x:.2f}' for x in ppl_per_context_length]], headers=['ctx len:'] + [f'{x//1000}K' for x in context_lengths] , tablefmt='pretty'))
    return samples_df, val_log

def evaluate_validation_set_squad(model, model_processor, data_loader_val, config, epoch, cur_step, start_datetime_str, num_samples_to_log=None):
    
    # prepare noise dataset
    with open('./custom_datasets/multidoc_squad/noise_docs_indices_val.pkl', 'rb') as f:
        noise_dataset_shuffled_indices_val = pickle.load(f)
    dataset_val = load_dataset("rajpurkar/squad_v2", split="validation", cache_dir=config['cache_dir'])
    shuffled_val_dataset = dataset_val.select(noise_dataset_shuffled_indices_val.tolist()*10)

    samples_df_list = []
    val_log = {}
    pred_dicts = []
    params_for_debug_per_example = []
    mean_token_counts = []
    for num_noise_docs in config['multidoc_num_noise_docs_eval']:
        cur_mean_token_count = 0
        print(f'Evaluating with {num_noise_docs} noise documents, noise injection policy: {config["multidoc_noise_injection_policy"]}')
        cur_pred_dict = init_pred_dict(config)
        noise_data_loader = DataLoader(shuffled_val_dataset, collate_fn=collate_fn_squad, batch_size=1, shuffle=False, num_workers=0).__iter__() # a bit hacky but we reset the DataLoader in every loop so we would not run out of noise documents
        for idx, batch in enumerate(tqdm(data_loader_val)):
            
            input_ids, prompt, golden_doc_id = get_input_ids_eval_squad(batch, model_processor, config, noise_data_loader, num_noise_docs, idx)
            batch['outputs'][0] = f'{golden_doc_id}|>'
            output, params_for_debug = model.generate(input_ids, max_length=len(input_ids[0])+config['eval_max_len'], eos_token_id=model_processor.eos_token_id)
            params_for_debug_per_example.append(params_for_debug)            
            response = model_processor.decode(output[0][len(input_ids[0]):])
            response = response.split('|>')[0] + '|>'
            cur_pred_dict, samples_df_list = update_results_eval(cur_pred_dict, samples_df_list, batch, idx, epoch, cur_step, response, prompt, num_noise_docs)
            cur_mean_token_count += input_ids.shape[1]
        
        pred_dicts.append(cur_pred_dict)
        mean_token_counts.append(cur_mean_token_count/len(data_loader_val))
        cur_res = run_evaluator([cur_pred_dict], [], config, start_datetime_str)
        print(cur_res['score'])
    
    samples_df = pd.DataFrame(samples_df_list)
    evaluator_response = run_evaluator(pred_dicts, samples_df, config, start_datetime_str)
    val_log['score'] = evaluator_response['score']
    for i_num_noise_docs, num_noise_docs in enumerate(config['multidoc_num_noise_docs_eval']):
        val_log[f'score_{num_noise_docs}_noise_docs'] = evaluator_response['scores_per_num_noise_docs'][i_num_noise_docs]
        val_log[f'mean_token_count_{num_noise_docs}_noise_docs'] = mean_token_counts[i_num_noise_docs]
    val_log['score'] = evaluator_response['score']
    if num_samples_to_log is not None:
        samples_to_log = [i for i in range(len(samples_df)) if i%config["eval_set_size"] in np.arange(config["eval_samples_to_log"])]
        samples_df = samples_df.iloc[samples_to_log]
    
    print(tabulate([['score:'] + evaluator_response["scores_per_num_noise_docs"]], headers=['num noise docs:'] + config['multidoc_num_noise_docs_eval'] , tablefmt='pretty'))
    return samples_df, val_log

def evaluate_validation_set(model, model_processor, data_loader_val, config, epoch, cur_step, start_datetime_str, num_samples_to_log=None):
    
    print(f'\n Evaluating over {config["dataset"]}, epoch: {epoch}, step: {cur_step}')

    if config["dataset"] == "ppl_test":
        samples_df, val_log = evaluate_validation_set_ppl_test(model, model_processor, data_loader_val, config, epoch, cur_step, start_datetime_str, num_samples_to_log=None)
        return samples_df, val_log
    if config["dataset"].startswith("squad"):
        samples_df, val_log = evaluate_validation_set_squad(model, model_processor, data_loader_val, config, epoch, cur_step, start_datetime_str, num_samples_to_log)
        return samples_df, val_log
    
    samples_df_list = []
    val_log = {}
    pred_dict = init_pred_dict(config)
    params_for_debug_per_example = []
    for idx, batch in enumerate(tqdm(data_loader_val)):
        input_ids, prompt = get_input_ids_eval(batch, model_processor, config)
        output, params_for_debug = model.generate(input_ids, max_length=len(input_ids[0])+config['eval_max_len'], eos_token_id=model_processor.eos_token_id)
        if config["dataset"] == "niah_custom":
            params_for_debug['needle_position'] = batch['needle_position'][0]
        params_for_debug_per_example.append(params_for_debug)
        response = model_processor.decode(output[0][len(input_ids[0]):])
        pred_dict, samples_df_list = update_results_eval(pred_dict, samples_df_list, batch, idx, epoch, cur_step, response, prompt)
    
    samples_df = pd.DataFrame(samples_df_list)
    evaluator_response = run_evaluator(pred_dict, samples_df, config, start_datetime_str)
    val_log['score'] = evaluator_response['score']
    if config["dataset"].startswith('niah'):
        val_log['niah_map'] = {'epoch':epoch, 'step':cur_step, 'niah_map': evaluator_response['niah_map']}
    if num_samples_to_log is not None:
        samples_df = samples_df.iloc[:num_samples_to_log]
        
    return samples_df, val_log

def get_decimation_config(config):
    decimation_config = {}
    activate_decimation = config['activate_decimation'] and config['mamba_arch'] == 'deci'
    decimation_config['activate'] = activate_decimation
    decimation_config['record_debug_params'] = config['record_debug_params']

    if activate_decimation:
        decimation_config['beta'] = config['decimation_beta']
        decimation_config['min_seq_len'] = config['decimation_min_seq_len']
        decimation_config['type'] = config['decimation_type']
        decimation_config['L_base'] = config['decimation_max_p_L_base']
        decimation_config['decimating_layers'] = config['decimating_layers']
        
    else:
        decimation_config['beta'] = 1
        decimation_config['min_seq_len'] = 0
        decimation_config['type'] = 'max_p'
        decimation_config['L_base'] = torch.inf
        decimation_config['decimating_layers'] = []
    
    return decimation_config

def validate_config(config):
    if not (config['mamba_arch'] == 'deci' and config['dataset'] == 'ppl_test'):
        config['deci_num_chunks'] = 1
    
    if not config['mamba_arch'] == 'deci':
        config['activate_decimation'] = False

    return config

def update_best_score(cur_score, best_score, config):
    if config["dataset"] == "ppl_test":
        return cur_score < best_score
    if config["dataset"].startswith('niah'):
        return cur_score > best_score
    if config["dataset"] == 'squad_retrieve':
        return cur_score > best_score
    
    raise(f'bad dataset {config["dataset"]}')

def init_best_score(config):
    if config["dataset"] == "ppl_test":
        return np.inf
    if config["dataset"].startswith('niah'):
        return -np.inf
    if config["dataset"] == "squad_retrieve":
        return -np.inf
    
    raise(f'bad dataset {config["dataset"]}')

def init_pred_dict(config):
    pred_dict = {}
    if config['dataset'] == 'squad_retrieve':
        pred_dict['results'] = []
    
    return pred_dict

'''used for ppl test, for other sequences do nothing.'''
def chunk_train_sequence(i_chunk, input_tokens, labels_tokens,  config):
    if config['dataset'] == 'ppl_test':
        num_toks_in_cur_seq = (i_chunk + 1) * input_tokens.shape[1] // config['deci_num_chunks']
        input_tokens_cur_seq = input_tokens[:,:num_toks_in_cur_seq]
        labels_start_in_cur_seq = i_chunk * input_tokens.shape[1] // config['deci_num_chunks']
        labels_end_in_cur_seq = (i_chunk + 1) * input_tokens.shape[1] // config['deci_num_chunks'] - 1
        num_labels_in_cur_seq = labels_end_in_cur_seq - labels_start_in_cur_seq
        labels_cur_seq = labels_tokens[labels_start_in_cur_seq:labels_end_in_cur_seq]
        return input_tokens_cur_seq, labels_cur_seq, num_labels_in_cur_seq
    
    else:
        return input_tokens, labels_tokens, len(labels_tokens)

def update_deci_layer(model, deci_layer):
    num_layers = len(model.backbone.layers)
    model.decimation_config['decimating_layers'] = [deci_layer]
    for layer in range(num_layers):
        model.backbone.layers[layer].decimation_config['decimating_layers'] = [deci_layer]
    return

def find_deci_layer(model, model_processor, data_loader_val, config):
    num_layers = len(model.backbone.layers)
    find_deci_layers_config = config
    find_deci_layers_config['ppl_test_context_lens_eval'] = [8000]
    find_deci_layers_config['ppl_test_num_windows_per_context_len_eval'] = 3
    score_per_layer = []
    for layer in range(8, min(num_layers,25), 1):
        update_deci_layer(model, layer)
        _, res = evaluate_validation_set(model, model_processor, data_loader_val, find_deci_layers_config, 0, 0, '', num_samples_to_log=0)
        score_per_layer.append(res['score'])
        print(f'score for layer {layer}: {res["score"]}')
    if config['dataset'] == 'ppl_test':
        deci_layer = np.argmin(score_per_layer)
    else:
        deci_layer = np.argmax(score_per_layer)
    print(f'deci layer: {deci_layer}')
    print(score_per_layer)
    return deci_layer

def run_train_loop(config, start_datetime_str):
    
    set_seed(config['seed'])
    config = validate_config(config)
    model_processor, model = load_model(config)
    data_loader_train, data_loader_val = get_data_loaders(config, model_processor=model_processor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_sched = get_lr_scheduler(config, optimizer, len(data_loader_train), data_loader_train.batch_size)
    ce_loss = CrossEntropyLoss()
    df_val = pd.DataFrame()
    if config["dataset"].startswith('niah'):
        df_niah = pd.DataFrame()
    if config["dataset"].startswith('ppl_test'):
        df_ppl_test = pd.DataFrame()

    if config['activate_logging']:
        wandb.login()
        wandb_run = wandb.init(project="icl_ssm", config=config, dir=config['wandb_dir'], name=f'{config["mamba_arch"]} {config["model_type"]} {config["dataset"]} lr {config["lr"]} batch size {config["grad_accum_steps"]} {config["run_name_addon"]}')
        os.environ["WANDB_CACHE_DIR"] = config['wandb_dir']
    grad_flow_data = init_grad_flow_data(model)

    # find deci layer automatically and update it
    if config['mamba_arch'] == 'deci' and config['find_deci_layer'] == True:
        deci_layer = find_deci_layer(model, model_processor, data_loader_val, config)
        update_deci_layer(model, deci_layer)

    # initial performance
    model.eval()
    cur_df_val, val_log_step_0 = evaluate_validation_set(model, model_processor, data_loader_val, config, 0, 0, start_datetime_str, num_samples_to_log=config['eval_samples_to_log'])
    df_val = df_val._append(cur_df_val,ignore_index=True)
    print(f'\nValidation Set - Initial Result | Score: {val_log_step_0["score"]:.3f}\n')
    if config['activate_logging']:
        wandb_table_val = wandb.Table(data=df_val)
        val_log_step_0['validation_data'] = wandb_table_val
        if config["dataset"].startswith('niah'):
            df_niah = df_niah._append(val_log_step_0['niah_map'],ignore_index=True)
            val_log_step_0.pop('niah_map')
            niah_table_val = wandb.Table(data=df_niah)
            val_log_step_0['niah_val'] = niah_table_val
        if config["dataset"].startswith('ppl_test'):
            df_ppl_test = df_ppl_test._append(val_log_step_0['ppl_per_ctx_len'],ignore_index=True)
            val_log_step_0.pop('ppl_per_ctx_len')
            ppl_test_table_val = wandb.Table(data=df_ppl_test)
            val_log_step_0['ppl_val'] = ppl_test_table_val
        wandb.log(val_log_step_0, step=0)

    if config['eval_mode']:
        return

    # train
    model.train()
    best_score = init_best_score(config)
    squad_noise_data_loader = None
    for epoch in range(config["epochs"]):
        if config['dataset'].startswith('squad'):
            squad_noise_data_loader = get_noise_data_loader_squad(config)

        for idx, batch in enumerate(data_loader_train):
            if epoch == 0 and config['recover_step'] is not None and squad_recovery_run_loaders_to_cp(config, idx, data_loader_train, squad_noise_data_loader):
                continue

            step = idx + epoch*len(data_loader_train)
            if (step) > config["max_step"]:
                break
            
            loss=0
            mean_input_len=0
            optimizer.zero_grad()
            for i in tqdm(range(batch["size"])):
                input_tokens, labels_tokens = get_input_ids_and_labels_train(batch, i, model_processor, config, epoch, squad_noise_data_loader)
                if input_tokens.shape[1] > config['max_train_input_len']:
                    print(f'input length {input_tokens.shape[1]} exceeds max train input length {config["max_train_input_len"]}, dropping sample')
                    continue
                if config['mamba_arch'] == 'deci' and input_tokens.shape[1]//config['deci_num_chunks'] < config['decimation_min_seq_len'] :
                    print(f'input length {input_tokens.shape[1]} cannot be chunked into {config["deci_num_chunks"]} chunks, dropping sample')
                    continue
                
                for i_chunk in range(config['deci_num_chunks']):
                    input_tokens_cur_seq, labels_cur_seq, num_labels_in_cur_seq = chunk_train_sequence(i_chunk, input_tokens, labels_tokens,  config)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out, params_for_debug = model(input_ids=input_tokens_cur_seq, num_last_tokens=num_labels_in_cur_seq+1)
                        logits = out.logits
                        cur_loss = ce_loss(logits[0,:-1,:], labels_cur_seq) / batch['size'] / config['deci_num_chunks'] # better as long as grad accum steps = batch size  # div by deci_num_chunks because the ce loss does a mean for a 1/deci_num_chunks for every chunk, so we should fix that

                    cur_loss.backward()
                    loss += cur_loss.detach().clone()
                    mean_input_len += input_tokens.shape[1] / batch['size'] / config['deci_num_chunks']# better as long as grad accum steps = batch size

            if step % config["grad_flow_steps"] == 0 and config['activate_logging']:
                log_grad_flow, grad_flow_data = get_grad_flow_log_format(model, step, grad_flow_data)

            if config['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_max_norm'])
            optimizer.step()
            lr_sched.step(epoch + idx/len(data_loader_train))
            cur_lr = optimizer.param_groups[0]["lr"]
            
            # metrics
            grad_norm = calc_grad_norm(model)       
            print(f'Epoch: {epoch} | Step In Epoch: {idx + 1} | Loss: {loss:.3e}, | Grad Norm: {grad_norm:.3e} | Mean Input Length: {mean_input_len:.3e}')
            log_cur_step = {"loss": loss, "grad_norm": grad_norm, "lr": cur_lr, "mean_input_len": mean_input_len} 
                
            # log validation samples, train samples and train data 
            if step % config['eval_steps'] == 0 and step > 0:     
                model.eval()
                cur_df_val, val_log_cur_step = evaluate_validation_set(model, model_processor, data_loader_val, config, epoch, step, start_datetime_str, num_samples_to_log=config['eval_samples_to_log'])
                
                if step % config['log_eval_predictions_steps'] == 0:
                    df_val = df_val._append(cur_df_val,ignore_index=True)
                
                cur_score = val_log_cur_step["score"]
                print(f'\nValidation Set - Epoch: {epoch} | Step In Epoch: {idx + 1} | Score: {cur_score:.3f}\n')

                if config["dataset"].startswith('niah'):
                    df_niah = df_niah._append(val_log_cur_step['niah_map'],ignore_index=True)
                    val_log_cur_step.pop('niah_map')
                
                if config["dataset"].startswith('ppl_test'):
                    df_ppl_test = df_ppl_test._append(val_log_cur_step['ppl_per_ctx_len'],ignore_index=True)
                    val_log_cur_step.pop('ppl_per_ctx_len')

                model.train()
                if config['activate_logging']:
                    log_all = {**log_cur_step, **val_log_cur_step}
                    if step % config["grad_flow_steps"] == 0:
                        log_all = {**log_all, **log_grad_flow}

                    if step % config['log_eval_predictions_steps'] == 0:
                        wandb_table_val = wandb.Table(data=df_val)
                        log_all['validation_data'] = wandb_table_val

                    if config["dataset"].startswith('niah'):
                        niah_table_val = wandb.Table(data=df_niah)
                        log_all['niah_val'] = niah_table_val
                    
                    if config["dataset"].startswith('ppl_test'):
                        ppl_test_table_val = wandb.Table(data=df_ppl_test)
                        log_all['ppl_val'] = ppl_test_table_val
                    
                    wandb.log(log_all, step=step)
                
                # save best model
                if update_best_score(cur_score, best_score, config):
                    best_score = cur_score
                    print(f'New best score: {best_score}, saving model')
                    save_model(config, model, model_processor, epoch, step, start_datetime_str, best_model=True)
            
            # log train data 
            else:
                if config['activate_logging']:
                    log_all = {**log_cur_step}
                    if step % config["grad_flow_steps"] == 0:
                        log_all = {**log_all, **log_grad_flow}
                    wandb.log(log_all, step=step)
            
            # save model
            if step % config['save_steps'] == 0 and step > 0:
                save_model(config, model, model_processor, epoch, step, start_datetime_str)
    
    if config['activate_logging']:
        wandb.finish()
    clean_up(start_datetime_str)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=int, default=-1)
    parser.add_argument("--device", type=str, default='None')
    args = parser.parse_args()
    config = load_config(args)
    start_datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_train_loop(config, start_datetime_str)