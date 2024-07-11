import torch
from tqdm import tqdm
from datasets import load_dataset
import argparse
from transformers import AutoTokenizer
model_processor = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def main(args):
    if args.eval_only:
        splits = ['test', 'validation']
    else:
        splits = ['test', 'validation', 'train']
    
    for split in splits:
        print(f'Tokenizing {split} split')
        cur_dataset = load_dataset("deepmind/pg19", cache_dir='./hf_cache', split=split)
        i=0
        tokenized_ds = []
        for sample in tqdm(cur_dataset):
            cur_sample = {}
            cur_sample['short_book_title'] = sample['short_book_title']
            cur_sample['input_tokens'] = model_processor(text=sample['text'], return_tensors="pt").input_ids
            tokenized_ds.append(cur_sample)

            i+=1
            if i%1000 == 0 and i>0:
                print(f'saving checkpoint after {i} examples')
                torch.save(tokenized_ds, f'./artifacts/ppl_test/pg19/{split}_set.pt')

        torch.save(tokenized_ds, f'./artifacts/ppl_test/pg19/{split}_set.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    main(args)